"""[A02] checkpoint tests. Seeds a temp database with A11, then reads it back."""
import copy
import json

import pytest

from agents.memory_retrieve.agent import MemoryRetrieve
from agents.memory_writer import agent as MW
from agents.memory_writer.agent import FIXTURE, MemoryWriter
from memory import db
from memory import embeddings as E

REPO = {"owner": "Shaheer-Cybersec", "repo": "damn-vulnerable-rag"}


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "t.db")
    monkeypatch.setattr(MW, "OBSIDIAN_DIR", tmp_path)
    monkeypatch.setattr(MW, "DATA_DIR", tmp_path)
    monkeypatch.setattr(E, "embed", lambda text: [0.6, 0.8])

def seed(run_id="r1", repo=None):
    c = copy.deepcopy(FIXTURE)
    c["run_id"] = run_id
    if repo:
        c["repo"] = {**c["repo"], **repo}
    assert MemoryWriter().execute(c)["status"] == "success"

def retrieve(repo=REPO):
    ctx = {"repo": repo}
    rec = MemoryRetrieve().execute(ctx)
    assert rec["status"] == "success"
    return ctx["memory"]


def test_empty_database_is_not_an_error():
    m = retrieve()
    assert m["past_posts"] == [] and m["queued_angles"] == [] and m["past_post_count"] == 0

def test_reads_back_what_a11_wrote():
    seed()
    m = retrieve()
    assert m["past_post_count"] == 1
    assert m["past_posts"][0]["angle_title"].startswith("Why top-k")
    assert m["embedded_post_ids"] == [m["past_posts"][0]["id"]]

def test_only_queued_angles_come_back():
    seed()
    titles = [a["title"] for a in retrieve()["queued_angles"]]
    assert len(titles) == 2                         # used + dropped are excluded
    assert "I built a vulnerable RAG app" not in titles
    assert titles[0] == "Building a RAG lab that is vulnerable on purpose"   # highest score first

def test_posts_from_other_repos_included_but_not_their_angles():
    seed()
    seed(run_id="r2", repo={"repo": "llm-redteam-harness",
                            "url": "https://github.com/Shaheer-Cybersec/llm-redteam-harness"})
    m = retrieve()
    assert m["past_post_count"] == 2                        # both repos' posts, for dedup
    assert len(m["queued_angles"]) == 2                     # only this repo's queue

def test_newest_post_first():
    seed("r1")
    seed("r2")
    ids = [p["id"] for p in retrieve()["past_posts"]]
    assert ids == sorted(ids, reverse=True)

def test_context_has_no_bytes():
    seed()
    json.dumps(retrieve())                                  # would raise on raw bytes

def test_post_vectors_helper_for_a04():
    seed()
    m = retrieve()
    with db.connect() as c:
        vecs = db.post_vectors(c, m["embedded_post_ids"])
    assert E.from_blob(next(iter(vecs.values()))) == pytest.approx([0.6, 0.8])

def test_missing_repo_fails_cleanly():
    rec = MemoryRetrieve().execute({})
    assert rec["status"] == "failed" and "repo" in rec["error"]