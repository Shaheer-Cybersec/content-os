"""
[I08] Content OS dashboard - local web UI over the orchestrator (I07).

    python -m dashboard.server            then open http://127.0.0.1:8765
    python -m dashboard.server --port 9000

Standard library only (no FastAPI, nothing to install, nothing to pay for).
Listens on 127.0.0.1 only: nothing on your network can reach it.

What it does (the same actions as run.py, with a UI):
  - start a run from a GitHub URL / zip URL, or by dropping a .zip file
  - show all 13 steps live (A01..A11 + G1/G2) and every stage's output
  - Claude's turn: copy the prompt, paste Claude's JSON answer, resume
  - G1: pick an angle; G2: read the post, approve or reject
  - memory panel: posts in memory, queued angles

Long steps (A01 download, A04 embeddings, LLM resumes) run in a background thread;
the page polls /api/runs/<id> and redraws. Progress is still the state.json on
disk, so the CLI, the dashboard and the runner skill all see the same run.
"""
from __future__ import annotations

import argparse
import json
import re
import threading
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from config import settings
from core import orchestrator as O
from core.base_agent import AgentError
from memory import db

STATIC = Path(__file__).with_name("static")
MAX_UPLOAD = 100 * 1024 * 1024                 # same cap as A01
RUN_ID_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")

STEPS = [  # (tag, name, who, what it does)
    ("A01", "repo_ingest", "code", "Fetch the repo, file tree, commits, key files"),
    ("A02", "memory_retrieve", "code", "Load past posts and queued angles"),
    ("A03", "analyzer", "claude", "Components, decisions, security notes, limits"),
    ("A04", "angle_extractor", "claude", "5-8 post angles, grounded, deduped, ranked"),
    ("G1", "pick angle", "you", "You choose the angle"),
    ("A05", "strategist", "claude", "Format, hook, beats, CTA, hashtags"),
    ("A06", "visual_planner", "claude", "Up to 3 quick screenshots"),
    ("A07", "writer", "claude", "The post, every claim bound to a file"),
    ("A08", "critic", "claude", "Style checks + hostile expert review"),
    ("A09", "reviser", "claude", "Fixes only the flagged lines"),
    ("A10", "packager", "code", "3000-char limit, final evidence check, file"),
    ("G2", "approve", "you", "You approve or reject"),
    ("A11", "memory_writer", "code", "Save post, queue angles, Obsidian note"),
]
PRODUCES = {"A01": "repo", "A02": "memory", "A03": "analysis", "A04": "angle_set",
            "A05": "strategy", "A06": "visual_plan", "A07": "draft", "A08": "critique",
            "A09": "final_draft", "A10": "package", "A11": "memory_write"}

_busy: set[str] = set()
_errors: dict[str, str] = {}
_lock = threading.Lock()


# ---------------- actions (plain functions; the HTTP layer only routes) ----------------

def _check_id(run_id: str) -> str:
    if not RUN_ID_RE.match(run_id or ""):
        raise AgentError("bad run id")
    return run_id


def _in_background(run_id: str, fn, *args) -> None:
    with _lock:
        if run_id in _busy:
            raise AgentError(f"run '{run_id}' is already working, wait for it")
        _busy.add(run_id)
        _errors.pop(run_id, None)

    def work():
        try:
            fn(*args)
        except AgentError as e:
            _errors[run_id] = str(e)
        except Exception as e:                     # a bug: show it, keep the server alive
            _errors[run_id] = f"{type(e).__name__}: {e}"
        finally:
            with _lock:
                _busy.discard(run_id)

    threading.Thread(target=work, daemon=True).start()


def new_run_id(source: str) -> str:
    name = re.sub(r"\.zip$|\.git$", "", source.rstrip("/").split("/")[-1].split("\\")[-1])
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:24] or "run"
    return f"{datetime.now():%m%d-%H%M%S}-{slug}"


def start(source: str, run_id: str | None = None) -> str:
    source = (source or "").strip()
    if not source:
        raise AgentError("give a GitHub URL, a zip URL, or drop a .zip")
    run_id = _check_id(run_id or new_run_id(source))
    if (settings.RUNS_DIR / run_id / "state.json").exists():
        raise AgentError(f"run '{run_id}' already exists")
    _in_background(run_id, O.start, source, run_id)
    return run_id


def save_upload(name: str, data: bytes) -> Path:
    if not name.lower().endswith(".zip"):
        raise AgentError("only .zip files (GitHub: Code -> Download ZIP)")
    if len(data) > MAX_UPLOAD:
        raise AgentError("zip is over 100 MB")
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", Path(name).name)
    d = settings.DATA_DIR / "uploads"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{datetime.now():%Y%m%d-%H%M%S}-{safe}"
    p.write_bytes(data)
    return p


def upload_and_start(name: str, data: bytes) -> str:
    p = save_upload(name, data)
    return start(str(p), new_run_id(name))


def pick(run_id: str, n: int) -> None:
    _in_background(_check_id(run_id), O.pick, run_id, int(n))


def approve(run_id: str) -> None:
    _in_background(_check_id(run_id), O.approve, run_id)


def resume(run_id: str) -> None:
    _in_background(_check_id(run_id), O.resume, run_id)


def reject(run_id: str, reason: str) -> None:
    O.reject(_check_id(run_id), reason or "rejected in dashboard")


def pending_stage(st: dict) -> tuple[str, str] | None:
    """(tag, agent name) of the stage waiting for Claude, if any."""
    if st["status"] not in ("waiting", "failed") or st["pos"] >= len(O.PLAN):
        return None
    tag = O.PLAN[st["pos"]]
    return (tag, dict((t, n) for t, n, _, _ in STEPS)[tag]) if tag.startswith("A") else None


def get_prompt(run_id: str) -> dict:
    st = O.load(_check_id(run_id))
    p = pending_stage(st)
    if not p:
        raise AgentError("no stage is waiting for Claude")
    f = settings.RUNS_DIR / run_id / f"{p[1]}.prompt.md"
    if not f.exists():
        raise AgentError(f"{f.name} not found")
    return {"stage": p[0], "name": p[1], "path": f"data/runs/{run_id}/{f.name}",
            "text": f.read_text(encoding="utf-8")}


def submit_response(run_id: str, text: str) -> None:
    st = O.load(_check_id(run_id))
    p = pending_stage(st)
    if not p:
        raise AgentError("no stage is waiting for Claude")
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)      # tolerate a pasted code fence
    try:
        json.loads(text)
    except json.JSONDecodeError as e:
        raise AgentError(f"that is not valid JSON: {e}")
    (settings.RUNS_DIR / run_id / f"{p[1]}.response.json").write_text(text, encoding="utf-8")
    resume(run_id)


# ---------------- views ----------------

def _trim_repo(r: dict) -> dict:
    return {k: r.get(k) for k in ("source_type", "owner", "repo", "url", "head_sha", "file_count",
                                  "tree_truncated", "languages", "cache_hit", "ingested_at")} | {
        "commits": (r.get("commits") or [])[:8],
        "key_files": [{"path": p, "chars": len(t or "")} for p, t in (r.get("key_files") or {}).items()]
        if isinstance(r.get("key_files"), dict) else [],
        "tree": [f["path"] for f in (r.get("tree") or [])[:400]],
        "readme": (r.get("readme") or "")[:1500],
    }


def stage_view(st: dict, busy: bool) -> list[dict]:
    out = []
    for i, (tag, name, who, what) in enumerate(STEPS):
        rec = st["records"].get(tag)
        status = rec["status"] if rec else "pending"
        if tag in ("G1", "G2"):
            status = "gate" if st["gate"] == tag else ("done" if st["pos"] > i else "pending")
        elif i == st["pos"] and busy:
            status = "running"
        if st["status"] == "rejected" and i >= st["pos"] and status == "pending":
            status = "cancelled"
        out.append({"tag": tag, "name": name, "who": who, "what": what, "status": status,
                    "ms": rec["duration_ms"] if rec else None,
                    "error": rec["error"] if rec and rec["status"] == "failed" else None})
    return out


def run_detail(run_id: str) -> dict:
    st = O.load(_check_id(run_id))
    busy = run_id in _busy
    ctx = st["ctx"]
    outputs = {}
    for tag, key in PRODUCES.items():
        if key in ctx:
            outputs[tag] = _trim_repo(ctx[key]) if key == "repo" else ctx[key]
    if "chosen_angle" in ctx:
        outputs["G1"] = ctx["chosen_angle"]
    pkg_md = None
    if "package" in ctx:
        p = Path(ctx["package"]["path"])
        p = p if p.is_absolute() else settings.ROOT / p
        if not p.exists():
            p = settings.OUTPUTS_DIR / Path(ctx["package"]["path"]).name
        pkg_md = p.read_text(encoding="utf-8") if p.exists() else None
    return {"summary": O.summary(st), "source": st["source"], "busy": busy,
            "error": _errors.get(run_id), "created": st["created"], "updated": st["updated"],
            "stages": stage_view(st, busy), "outputs": outputs,
            "pending": pending_stage(st), "package_md": pkg_md}


def list_runs() -> list[dict]:
    rows = []
    for p in settings.RUNS_DIR.glob("*/state.json"):
        try:
            st = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        repo = st["ctx"].get("repo", {})
        rows.append({"run_id": st["run_id"], "status": st["status"], "gate": st["gate"],
                     "source": st["source"], "updated": st["updated"],
                     "repo": f"{repo.get('owner')}/{repo.get('repo')}" if repo else None,
                     "busy": st["run_id"] in _busy})
    for rid in _busy:                                  # started, state not written yet
        if not any(r["run_id"] == rid for r in rows):
            rows.append({"run_id": rid, "status": "running", "gate": None, "source": "",
                         "updated": "9999", "repo": None, "busy": True})
    return sorted(rows, key=lambda r: r["updated"], reverse=True)


def memory_view() -> dict:
    if not Path(db.DB_PATH).exists():
        return {"posts": [], "queued": [], "counts": {"posts": 0, "queued": 0, "repos": 0}}
    with db.connect() as conn:
        posts = [dict(r) for r in conn.execute(
            "SELECT id, repo_id, substr(text, 1, 160) AS preview, length(text) AS chars, "
            "embedding IS NOT NULL AS embedded, created_at FROM posts ORDER BY id DESC LIMIT 50")]
        queued = [dict(r) for r in conn.execute(
            "SELECT repo_id, title, score FROM angles WHERE status='queued' ORDER BY score DESC LIMIT 50")]
        repos = conn.execute("SELECT count(*) FROM repos").fetchone()[0]
    return {"posts": posts, "queued": queued,
            "counts": {"posts": len(posts), "queued": len(queued), "repos": repos}}


# ---------------- HTTP ----------------

class Handler(BaseHTTPRequestHandler):
    server_version = "ContentOS/1"

    def log_message(self, fmt, *args):             # quiet console
        pass

    def _send(self, code: int, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype + ("; charset=utf-8" if "text" in ctype or "json" in ctype else ""))
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _body(self) -> bytes:
        n = int(self.headers.get("Content-Length") or 0)
        if n > MAX_UPLOAD:
            raise AgentError("request too large")
        return self.rfile.read(n) if n else b""

    def _json(self) -> dict:
        b = self._body()
        return json.loads(b) if b else {}

    def _route(self, method: str):
        u = urlparse(self.path)
        parts = [p for p in u.path.split("/") if p]
        try:
            if method == "GET" and not parts:
                return self._send(200, (STATIC / "index.html").read_bytes(), "text/html")
            if parts[:1] != ["api"]:
                return self._send(404, {"error": "not found"})
            p = parts[1:]
            if method == "GET":
                if p == ["runs"]:
                    return self._send(200, list_runs())
                if p == ["memory"]:
                    return self._send(200, memory_view())
                if len(p) == 2 and p[0] == "runs":
                    return self._send(200, run_detail(p[1]))
                if len(p) == 3 and p[0] == "runs" and p[2] == "prompt":
                    return self._send(200, get_prompt(p[1]))
            if method == "POST":
                if p == ["runs"]:
                    return self._send(202, {"run_id": start(self._json().get("source", ""))})
                if p == ["upload"]:
                    name = parse_qs(u.query).get("name", ["upload.zip"])[0]
                    return self._send(202, {"run_id": upload_and_start(name, self._body())})
                if len(p) == 3 and p[0] == "runs":
                    rid, act = p[1], p[2]
                    if act == "pick":
                        pick(rid, self._json().get("n", 0))
                    elif act == "approve":
                        approve(rid)
                    elif act == "reject":
                        reject(rid, self._json().get("reason", ""))
                    elif act == "resume":
                        resume(rid)
                    elif act == "response":
                        submit_response(rid, self._json().get("text", ""))
                    else:
                        return self._send(404, {"error": "unknown action"})
                    return self._send(202, {"ok": True})
            return self._send(404, {"error": "not found"})
        except AgentError as e:
            return self._send(400, {"error": str(e)})
        except Exception as e:
            return self._send(500, {"error": f"{type(e).__name__}: {e}"})

    def do_GET(self):
        self._route("GET")

    def do_POST(self):
        self._route("POST")


def serve(port: int = 8765) -> ThreadingHTTPServer:
    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Content OS dashboard")
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    httpd = serve(a.port)
    print(f"Content OS dashboard: http://127.0.0.1:{a.port}   (Ctrl+C to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("stopped")
