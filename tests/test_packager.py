"""[A10] checkpoint tests. Output goes to a temp folder."""
import copy

import pytest

from agents.packager import agent as PK
from agents.packager.agent import FIXTURE, Packager


@pytest.fixture(autouse=True)
def out_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(PK, "OUTPUTS_DIR", tmp_path)
    return tmp_path

def ctx():
    return copy.deepcopy(FIXTURE)

def run(c):
    rec = Packager().execute(c)
    return rec, c.get("package")


def test_packages_valid_draft(out_dir):
    rec, pkg = run(ctx())
    assert rec["status"] == "success"
    assert pkg["evidence_checked"] == 2 and pkg["warnings"] == []
    body = (out_dir / pkg["path"].split("/")[-1]).read_text(encoding="utf-8")
    assert "## Post (copy from here)" in body
    assert "- [ ] retriever.py top_k line" in body
    assert "`app/retriever.py`" in body

def test_final_draft_wins_over_draft():
    c = ctx()
    c["final_draft"] = {"text": "Revised version.", "claims": []}
    _, pkg = run(c)
    assert pkg["post"]["text"] == "Revised version."

def test_invented_file_blocks_packaging(out_dir):
    c = ctx()
    c["draft"]["claims"][0]["source_path"] = "app/guardrails.py"
    rec, pkg = run(c)
    assert rec["status"] == "failed" and "path not in repo" in rec["error"]
    assert pkg is None and list(out_dir.iterdir()) == []       # nothing written

def test_invented_commit_blocks_packaging():
    c = ctx()
    c["draft"]["claims"][1]["source_ref"] = "deadbee"
    rec, _ = run(c)
    assert rec["status"] == "failed" and "unknown commit" in rec["error"]

def test_over_linkedin_limit_fails():
    c = ctx()
    c["draft"]["text"] = "x" * 3001
    rec, _ = run(c)
    assert rec["status"] == "failed" and "3001 chars" in rec["error"]

def test_links_and_hashtags_warn_but_package():
    c = ctx()
    c["draft"]["text"] += "\nhttps://github.com/x/y #a #b #c"
    rec, pkg = run(c)
    assert rec["status"] == "success"
    assert len(pkg["warnings"]) == 2

def test_no_draft_fails_cleanly():
    c = ctx()
    del c["draft"]
    rec, _ = run(c)
    assert rec["status"] == "failed" and "no final_draft or draft" in rec["error"]

def test_post_handed_on_for_approval():
    _, pkg = run(ctx())
    assert set(pkg["post"]) == {"text", "evidence"}