"""[A01] offline tests: source detection, zip safety, extraction on fake repos.
Live git clones are checked by running the agent module (see checkpoint)."""
import zipfile

import pytest

from agents.repo_ingest import agent as A
from agents.repo_ingest.agent import (RepoIngest, classify, find_deps, parse_url,
                                      pick_key_files, walk_tree)
from core.base_agent import AgentError
from core.evidence import validate

SHA = "672971d66a2ef9f85151e53283113f33d642dabd"


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    """Tests never write into your real data/repo_cache."""
    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setattr(A, "CACHE_DIR", cache)


# ---------- source detection ----------

@pytest.mark.parametrize("url,expected", [
    ("https://github.com/pallets/itsdangerous", ("pallets", "itsdangerous")),
    ("https://github.com/a-b/c.d.git", ("a-b", "c.d")),
    ("git@github.com:owner/repo.git", ("owner", "repo")),
    ("https://github.com/owner/repo/", ("owner", "repo")),
])
def test_git_urls(url, expected):
    assert parse_url(url) == expected

def test_zip_url_detected():
    c = classify("https://github.com/o/r/archive/refs/heads/main.zip")
    assert c["kind"] == "zip_url" and (c["owner"], c["repo"]) == ("o", "r")

@pytest.mark.parametrize("bad", ["hello", "https://gitlab.com/a/b",
                                 "https://github.com/onlyowner", "http://github.com/a/b",
                                 "C:/nowhere/missing.zip", ""])
def test_bad_sources_rejected(bad):
    with pytest.raises(AgentError):
        classify(bad)


# ---------- fake repo on disk ----------

FILES = {
    "README.md": "# demo",
    "requirements.txt": "fastapi\n",
    "app/main.py": "print('entry')\n",
    "app/util.py": "x = 1\n",
    "tests/test_util.py": "def test(): pass\n",
    "node_modules/junk.js": "junk",
}

@pytest.fixture
def fake_repo(tmp_path):
    root = tmp_path / "repo"
    for rel, text in FILES.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    return root

def test_walk_tree_posix_paths_and_skips(fake_repo):
    tree, truncated = walk_tree(fake_repo)
    paths = {f["path"] for f in tree}
    assert "app/main.py" in paths
    assert not any(p.startswith("node_modules") for p in paths)
    assert truncated is False

def test_deps_found(fake_repo):
    tree, _ = walk_tree(fake_repo)
    assert list(find_deps(fake_repo, tree)) == ["requirements.txt"]

def test_key_files_entry_first_tests_last(fake_repo):
    tree, _ = walk_tree(fake_repo)
    order = list(pick_key_files(fake_repo, tree))
    assert order[0] == "app/main.py" and order[-1] == "tests/test_util.py"

def test_code_beats_hidden_config(tmp_path):
    """Regression: .pre-commit-config.yaml and .github/ must not outrank real source."""
    for rel in (".pre-commit-config.yaml", ".github/workflows/ci.yml",
                "src/pkg/signer.py", "docs/conf.py"):
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("x = 1\n")
    tree, _ = walk_tree(tmp_path)
    order = list(pick_key_files(tmp_path, tree))
    assert order[0] == "src/pkg/signer.py"
    assert order.index("docs/conf.py") < order.index(".pre-commit-config.yaml")


# ---------- zip input ----------

def make_zip(path, files, prefix="demo-main/", comment=b""):
    with zipfile.ZipFile(path, "w") as z:
        for rel, text in files.items():
            z.writestr(prefix + rel, text)
        z.comment = comment
    return path

def test_github_style_zip_uses_commit_from_comment(tmp_path):
    zp = make_zip(tmp_path / "demo-main.zip", FILES, comment=SHA.encode())
    ctx = {"source": str(zp)}
    rec = RepoIngest().execute(ctx)
    r = ctx["repo"]
    assert rec["status"] == "success"
    assert r["source_type"] == "zip" and r["repo"] == "demo"
    assert r["head_sha"] == SHA and r["source_ref_kind"] == "commit"
    assert "app/main.py" in {f["path"] for f in r["tree"]}      # wrapper folder stripped

def test_zip_without_comment_uses_content_hash(tmp_path):
    zp = make_zip(tmp_path / "plain.zip", FILES, prefix="")
    ctx = {"source": str(zp)}
    RepoIngest().execute(ctx)
    assert ctx["repo"]["source_ref_kind"] == "content-hash"
    assert len(ctx["repo"]["head_sha"]) == 40

def test_zip_output_works_with_evidence_validator(tmp_path):
    zp = make_zip(tmp_path / "demo-main.zip", FILES, comment=SHA.encode())
    ctx = {"source": str(zp)}
    RepoIngest().execute(ctx)
    ok = validate([{"claim": "entry point", "source_path": "app/main.py",
                    "source_ref": SHA[:7]}], ctx["repo"])
    assert ok["ok"]

def test_second_zip_run_is_cache_hit(tmp_path):
    zp = make_zip(tmp_path / "demo-main.zip", FILES, comment=SHA.encode())
    RepoIngest().execute({"source": str(zp)})
    ctx = {"source": str(zp)}
    RepoIngest().execute(ctx)
    assert ctx["repo"]["cache_hit"] is True

def test_zip_slip_rejected(tmp_path):
    zp = tmp_path / "evil.zip"
    with zipfile.ZipFile(zp, "w") as z:
        z.writestr("../../escaped.txt", "pwned")
    rec = RepoIngest().execute({"source": str(zp)})
    assert rec["status"] == "failed" and "unsafe path" in rec["error"]
    assert not (tmp_path.parent / "escaped.txt").exists()

def test_zip_bomb_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(A, "ZIP_MAX_UNCOMPRESSED", 10)
    zp = make_zip(tmp_path / "big.zip", {"a.txt": "x" * 100})
    rec = RepoIngest().execute({"source": str(zp)})
    assert rec["status"] == "failed" and "300 MB" in rec["error"]

def test_not_a_zip_rejected(tmp_path):
    fake = tmp_path / "fake.zip"
    fake.write_text("not a zip")
    rec = RepoIngest().execute({"source": str(fake)})
    assert rec["status"] == "failed" and "not a valid zip" in rec["error"]


# ---------- agent contract ----------

def test_missing_source_fails_cleanly():
    rec = RepoIngest().execute({})
    assert rec["status"] == "failed" and "source" in rec["error"]

def test_zip_never_returns_git_cache(tmp_path):
    """Regression: a zip and a git clone of the same commit must not share a cache file."""
    import json
    A.cache_path("git", "local", "demo", SHA).write_text(json.dumps({"source_type": "git"}))
    zp = make_zip(tmp_path / "demo-main.zip", FILES, comment=SHA.encode())
    ctx = {"source": str(zp)}
    RepoIngest().execute(ctx)
    assert ctx["repo"]["source_type"] == "zip" and ctx["repo"]["cache_hit"] is False