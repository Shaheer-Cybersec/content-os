"""[I05] checkpoint tests for the evidence validator, against a fake A01 output."""
import pytest

from core.evidence import validate

REPO = {
    "head_sha": "38ae3eb0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6",
    "tree_truncated": False,
    "tree": [{"path": "app/retriever.py", "size": 1200},
             {"path": "README.md", "size": 400}],
    "commits": [{"sha": "38ae3eb0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6", "subject": "add retriever"},
                {"sha": "ceb45f6aaaabbbbccccddddeeeeffff000011112", "subject": "init"}],
}

def ev(path="app/retriever.py", ref="38ae3eb", claim="Top-k is 5"):
    return {"claim": claim, "source_path": path, "source_ref": ref}


def test_real_path_and_short_sha_valid():
    r = validate([ev()], REPO)
    assert r["ok"] and len(r["valid"]) == 1

def test_full_sha_and_older_commit_valid():
    r = validate([ev(ref="ceb45f6aaaabbbbccccddddeeeeffff000011112")], REPO)
    assert r["ok"]

def test_windows_style_path_still_matches():
    assert validate([ev(path=r"app\retriever.py")], REPO)["ok"]

def test_invented_path_rejected():
    r = validate([ev(path="app/guardrails.py")], REPO)
    assert not r["ok"]
    assert "path not in repo" in r["invalid"][0]["reason"]

def test_invented_sha_rejected():
    r = validate([ev(ref="deadbee")], REPO)
    assert "unknown commit" in r["invalid"][0]["reason"]

@pytest.mark.parametrize("ref", ["main", "HEAD", "v1.0", "38ae"])
def test_non_sha_refs_rejected(ref):
    r = validate([ev(ref=ref)], REPO)
    assert "not a commit SHA" in r["invalid"][0]["reason"]

def test_malformed_evidence_rejected():
    r = validate([{"claim": "x", "source_path": "../etc/passwd", "source_ref": "38ae3eb"}], REPO)
    assert "malformed" in r["invalid"][0]["reason"]

def test_truncated_tree_fails_closed():
    repo = {**REPO, "tree_truncated": True}
    r = validate([ev(path="deep/unlisted.py")], repo)
    assert "cannot verify" in r["invalid"][0]["reason"]

def test_mixed_batch_splits_correctly():
    r = validate([ev(), ev(path="fake.py"), ev(ref="deadbee")], REPO)
    assert len(r["valid"]) == 1 and len(r["invalid"]) == 2 and not r["ok"]