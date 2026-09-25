"""
[A01] repo_ingest - static agent (no LLM)

Input : ctx["source"], any of:
          https://github.com/owner/repo                     (git clone)
          https://github.com/owner/repo.git                 (git clone)
          https://github.com/owner/repo/archive/main.zip    (zip download)
          G:\\Downloads\\repo-main.zip                       (local zip, "Download ZIP" on GitHub)
Output: ctx["repo"]  structured JSON the analyzer (A03) reads

Every path in "tree" becomes a valid evidence source_path. head_sha (and each
SHA in "commits") becomes a valid source_ref. I05 rejects anything else.

Zip inputs have no git history. GitHub writes the commit SHA into the zip's
comment, so head_sha is still a real commit. If that comment is missing, head_sha
falls back to a SHA-256 of the zip bytes (source_ref_kind = "content-hash").

Zip extraction is hardened: no path traversal (zip slip), no symlinks,
capped file count and uncompressed size (zip bomb).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from config.settings import CACHE_DIR
from core.base_agent import AgentError, BaseAgent

GIT_URL_RE = re.compile(r"github\.com[/:]([\w.-]+)/([\w.-]+?)(?:\.git)?/?$")
ZIP_URL_RE = re.compile(r"github\.com/([\w.-]+)/([\w.-]+)/(?:archive|zipball)/")
SHA40_RE = re.compile(r"^[0-9a-f]{40}$")

SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "env",
             "dist", "build", ".mypy_cache", ".pytest_cache", ".idea", ".vscode"}
DEP_FILES = {"requirements.txt", "pyproject.toml", "setup.py", "Pipfile",
             "package.json", "go.mod", "Cargo.toml", "docker-compose.yml",
             "docker-compose.yaml", "Dockerfile"}
SOURCE_EXT = {".py", ".js", ".ts", ".go", ".rs", ".java", ".sh", ".ps1",
              ".yaml", ".yml", ".toml", ".json", ".md"}
ENTRY_HINTS = ("main", "app", "server", "cli", "run", "api", "config")

MAX_FILES = 500              # tree entries kept (tree_truncated=True beyond this)
MAX_COMMITS = 30             # history depth for git sources
README_CHARS = 8_000
DEP_CHARS = 4_000
KEY_FILE_CHARS = 6_000       # per file
KEY_FILES_BUDGET = 60_000    # total source chars handed to the analyzer

ZIP_MAX_DOWNLOAD = 100 * 1024 * 1024       # 100 MB compressed
ZIP_MAX_UNCOMPRESSED = 300 * 1024 * 1024   # 300 MB after extraction
ZIP_MAX_MEMBERS = 20_000
CACHE_VERSION = 3            # bump when the output shape changes; old cache files are ignored


# ---------- helpers ----------

def git(args: list[str], cwd: Path | None = None) -> str:
    try:
        r = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=180)
    except FileNotFoundError:
        raise AgentError("git is not installed or not on PATH")
    except subprocess.TimeoutExpired:
        raise AgentError(f"git {args[0]} timed out after 180 s")
    if r.returncode != 0:
        err = r.stderr.strip()
        if "could not read Username" in err or "not found" in err.lower():
            raise AgentError("repo not found or private - check the URL")
        raise AgentError(f"git {args[0]} failed: {err[:300]}")
    return r.stdout


def _force_remove(func, path, _exc):
    # Windows: files inside .git are read-only and rmtree fails without this
    os.chmod(path, stat.S_IWRITE)
    func(path)


def rmtree(path: Path) -> None:
    if sys.version_info >= (3, 12):
        shutil.rmtree(path, onexc=_force_remove)
    else:
        shutil.rmtree(path, onerror=_force_remove)


def read_text(p: Path, limit: int) -> str:
    try:
        return p.read_text(encoding="utf-8", errors="replace")[:limit]
    except OSError:
        return ""


# ---------- source detection ----------

def classify(source: str) -> dict:
    """Work out what kind of input this is. Returns a dict with a 'kind' key."""
    s = source.strip().strip('"')
    if not s:
        raise AgentError("source is empty")

    if s.lower().endswith(".zip") and not s.lower().startswith(("http://", "https://")):
        p = Path(s).expanduser()
        if not p.is_file():
            raise AgentError(f"zip file not found: {s}")
        return {"kind": "zip_file", "path": p, "owner": "local", "repo": None}

    if s.lower().startswith("http://"):
        raise AgentError("only https:// URLs are accepted")

    m = ZIP_URL_RE.search(s)
    if m or (s.lower().startswith("https://") and s.lower().endswith(".zip")):
        owner, repo = (m.group(1), m.group(2)) if m else ("remote", None)
        return {"kind": "zip_url", "url": s, "owner": owner, "repo": repo}

    m = GIT_URL_RE.search(s)
    if m:
        return {"kind": "git", "owner": m.group(1), "repo": m.group(2),
                "url": f"https://github.com/{m.group(1)}/{m.group(2)}.git"}

    raise AgentError(f"not a GitHub URL or .zip file: {source}")


def parse_url(url: str) -> tuple[str, str]:
    """Kept for callers that only need owner/repo from a git URL."""
    c = classify(url)
    if c["kind"] != "git":
        raise AgentError(f"not a GitHub repo URL: {url}")
    return c["owner"], c["repo"]


# ---------- zip handling ----------

def download_zip(url: str, dest: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "content-os"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
            total = 0
            while chunk := r.read(1 << 16):
                total += len(chunk)
                if total > ZIP_MAX_DOWNLOAD:
                    raise AgentError("zip download is larger than 100 MB")
                f.write(chunk)
    except AgentError:
        raise
    except Exception as e:
        raise AgentError(f"could not download zip: {type(e).__name__}: {e}")


def safe_extract(zip_path: Path, dest: Path) -> zipfile.ZipFile:
    """Extract with zip-slip, symlink and zip-bomb protection."""
    try:
        zf = zipfile.ZipFile(zip_path)
    except zipfile.BadZipFile:
        raise AgentError("file is not a valid zip archive")

    infos = zf.infolist()
    if len(infos) > ZIP_MAX_MEMBERS:
        raise AgentError(f"zip has too many entries ({len(infos)})")
    if sum(i.file_size for i in infos) > ZIP_MAX_UNCOMPRESSED:
        raise AgentError("zip expands to more than 300 MB")

    for info in infos:
        name = info.filename.replace("\\", "/")
        parts = PurePosixPath(name).parts
        if name.startswith("/") or ".." in parts or (parts and ":" in parts[0]):
            raise AgentError(f"unsafe path in zip: {info.filename}")
        if stat.S_ISLNK(info.external_attr >> 16):
            continue                          # never materialise symlinks
        zf.extract(info, dest)
    return zf


def zip_root(extracted: Path) -> Path:
    """GitHub zips wrap everything in one folder, e.g. repo-main/. Step into it."""
    entries = [p for p in extracted.iterdir() if p.name != "__MACOSX"]
    if len(entries) == 1 and entries[0].is_dir():
        return entries[0]
    return extracted


def zip_identity(zf: zipfile.ZipFile, zip_path: Path) -> tuple[str, str]:
    """(head_sha, kind). GitHub/git-archive zips store the commit SHA as the zip comment."""
    comment = zf.comment.decode("ascii", "ignore").strip().lower()
    if SHA40_RE.match(comment):
        return comment, "commit"
    digest = hashlib.sha256(zip_path.read_bytes()).hexdigest()
    return digest[:40], "content-hash"


def repo_name_from_folder(folder: Path, fallback: str) -> str:
    name = folder.name if folder.name else fallback
    return re.sub(r"-(main|master|[0-9a-f]{7,40})$", "", name) or fallback


# ---------- extraction (pure functions: easy to test offline) ----------

def walk_tree(root: Path) -> tuple[list[dict], bool]:
    """All files as repo-relative POSIX paths. Returns (files, truncated)."""
    files = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        for f in sorted(filenames):
            p = Path(dirpath) / f
            files.append({"path": p.relative_to(root).as_posix(), "size": p.stat().st_size})
            if len(files) >= MAX_FILES:
                return files, True
    return files, False


def find_readme(root: Path) -> str:
    for p in sorted(root.iterdir()):
        if p.is_file() and p.name.lower().startswith("readme"):
            return read_text(p, README_CHARS)
    return ""


def find_deps(root: Path, tree: list[dict]) -> dict[str, str]:
    """Dependency files at the root or one folder down."""
    return {f["path"]: read_text(root / f["path"], DEP_CHARS)
            for f in tree
            if Path(f["path"]).name in DEP_FILES and f["path"].count("/") <= 1}


def read_commits(root: Path) -> list[dict]:
    out = git(["log", f"-n{MAX_COMMITS}", "--date=short",
               "--pretty=format:%H%x1f%an%x1f%ad%x1f%s"], cwd=root)
    commits = []
    for line in out.splitlines():
        sha, author, date, subject = line.split("\x1f", 3)
        commits.append({"sha": sha, "author": author, "date": date, "subject": subject})
    return commits


CODE_EXT = {".py", ".js", ".ts", ".go", ".rs", ".java", ".sh", ".ps1"}
LOW_VALUE_DIRS = ("docs/", "examples/", "benchmarks/")


def pick_key_files(root: Path, tree: list[dict]) -> dict[str, str]:
    """Source files the analyzer reads, within budget. Order:
    real code before config/docs, entry points first, hidden dirs (.github, .pre-commit)
    and tests last."""
    candidates = [f for f in tree
                  if Path(f["path"]).suffix in SOURCE_EXT
                  and not Path(f["path"]).name.lower().startswith("readme")
                  and Path(f["path"]).name != "package-lock.json"
                  and f["size"] > 0]

    def priority(f):
        path = f["path"]
        name = Path(path).stem.lower()
        is_hidden = any(part.startswith(".") for part in path.split("/"))
        is_test = "test" in path.lower()
        is_low = path.startswith(LOW_VALUE_DIRS)
        is_code = Path(path).suffix in CODE_EXT
        is_entry = any(h in name for h in ENTRY_HINTS)
        return (is_hidden, is_test, is_low, not is_code, not is_entry, path.count("/"), f["size"])

    picked, used = {}, 0
    for f in sorted(candidates, key=priority):
        if used >= KEY_FILES_BUDGET:
            break
        text = read_text(root / f["path"], KEY_FILE_CHARS)
        picked[f["path"]] = text
        used += len(text)
    return picked


def language_mix(tree: list[dict]) -> dict[str, int]:
    c = Counter(Path(f["path"]).suffix.lower() or "(none)" for f in tree)
    return dict(c.most_common(10))


def extract(root: Path, meta: dict, commits: list[dict]) -> dict:
    tree, truncated = walk_tree(root)
    return {
        **meta,   # source_type, owner, repo, url, head_sha, source_ref_kind
        "ingested_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "file_count": len(tree),
        "tree_truncated": truncated,
        "languages": language_mix(tree),
        "readme": find_readme(root),
        "dependencies": find_deps(root, tree),
        "commits": commits,
        "tree": tree,
        "key_files": pick_key_files(root, tree),
    }


def cache_path(source_type: str, owner: str, repo: str, head_sha: str) -> Path:
    """One cache file per source type, so a zip run never returns a git run's data."""
    return CACHE_DIR / f"v{CACHE_VERSION}__{source_type}__{owner}__{repo}__{head_sha[:12]}.json"


# ---------- agent ----------

class RepoIngest(BaseAgent):
    agent_id = "A01"
    name = "repo_ingest"
    requires = ("source",)
    produces = "repo"
    uses_llm = False

    def run(self, ctx: dict) -> dict:
        src = classify(ctx["source"])
        return self._from_git(src) if src["kind"] == "git" else self._from_zip(src)

    # --- git clone ---
    def _from_git(self, src: dict) -> dict:
        owner, repo = src["owner"], src["repo"]
        head = git(["ls-remote", src["url"], "HEAD"]).split()
        if not head:
            raise AgentError("repo has no commits yet")
        head_sha = head[0]

        cached = cache_path("git", owner, repo, head_sha)
        if cached.exists():
            return {**json.loads(cached.read_text(encoding="utf-8")), "cache_hit": True}

        tmp = Path(tempfile.mkdtemp(prefix="cos_"))
        try:
            dest = tmp / repo
            git(["clone", f"--depth={MAX_COMMITS}", "--quiet", src["url"], str(dest)])
            meta = {"source_type": "git", "owner": owner, "repo": repo,
                    "url": f"https://github.com/{owner}/{repo}",
                    "head_sha": head_sha, "source_ref_kind": "commit"}
            data = extract(dest, meta, read_commits(dest))
        finally:
            rmtree(tmp)
        return self._save(data)

    # --- zip (local file or URL) ---
    def _from_zip(self, src: dict) -> dict:
        tmp = Path(tempfile.mkdtemp(prefix="cos_"))
        try:
            if src["kind"] == "zip_url":
                zip_path = tmp / "download.zip"
                download_zip(src["url"], zip_path)
                url = src["url"]
            else:
                zip_path = src["path"]
                url = str(zip_path)

            out = tmp / "x"
            zf = safe_extract(zip_path, out)
            head_sha, ref_kind = zip_identity(zf, zip_path)
            zf.close()
            root = zip_root(out)
            owner = src["owner"]
            repo = src["repo"] or repo_name_from_folder(root, zip_path.stem)

            cached = cache_path("zip", owner, repo, head_sha)
            if cached.exists():
                return {**json.loads(cached.read_text(encoding="utf-8")), "cache_hit": True}

            meta = {"source_type": "zip", "owner": owner, "repo": repo, "url": url,
                    "head_sha": head_sha, "source_ref_kind": ref_kind}
            data = extract(root, meta, commits=[])
        finally:
            rmtree(tmp)
        return self._save(data)

    def _save(self, data: dict) -> dict:
        cache_path(data["source_type"], data["owner"], data["repo"], data["head_sha"]).write_text(
            json.dumps(data, indent=2), encoding="utf-8")
        return {**data, "cache_hit": False}


# ---------- manual checkpoint ----------
if __name__ == "__main__":
    source = sys.argv[1] if len(sys.argv) > 1 else "https://github.com/pallets/itsdangerous"
    ctx = {"source": source}
    rec = RepoIngest().execute(ctx)
    print(json.dumps(rec, indent=2))
    if rec["status"] == "success":
        r = ctx["repo"]
        print(f"source      {r['source_type']}  ({r['url']})")
        print(f"repo        {r['owner']}/{r['repo']}")
        print(f"head_sha    {r['head_sha'][:12]}  kind={r['source_ref_kind']}")
        print(f"cache_hit   {r['cache_hit']}")
        print(f"files       {r['file_count']}  truncated={r['tree_truncated']}")
        print(f"languages   {r['languages']}")
        print(f"deps        {list(r['dependencies'])}")
        print(f"commits     {len(r['commits'])}")
        print(f"key files   {len(r['key_files'])}  chars={sum(map(len, r['key_files'].values()))}")
        print(f"readme      {len(r['readme'])} chars")