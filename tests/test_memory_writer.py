
"""[A11] checkpoint tests. Temp database and temp Obsidian folder; embeddings faked."""
import copy
import json

import pytest

from agents.memory_writer import agent as MW
from agents.memory_writer.agent import FIXTURE, MemoryWriter
from memory import db
from memory import embeddings as E


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "t.db")
    notes = tmp_path / "obsidian"
    notes.mkdir()
    monkeypatch.setattr(MW, "OBSIDIAN_DIR", notes)
    monkeypatch.setattr(MW, "DATA_DIR", tmp_path)
    monkeypatch.setattr(E, "embed", lambda text: [0.6, 0.8])   # fast, deterministic
    return notes

def ctx():
    return copy.deepcopy(FIXTURE)

def rows(sql):
    with db.connect() as c:
        return [dict(r) for r in c.execute(sql)]


def test_writes_everything():
    c = ctx()
    rec = MemoryWriter().execute(c)
    assert rec["status"] == "success"
    out = c["memory_write"]
    assert out == {**out, "angles_used": 1, "angles_queued": 2, "angles_dropped": 1, "embedded": True}
    assert len(rows("SELECT * FROM repos")) == 1
    assert rows("SELECT status FROM runs")[0]["status"] == "done"
    assert len(rows("SELECT * FROM posts")) == 1

def test_chosen_angle_linked_to_post():
    c = ctx()
    MemoryWriter().execute(c)
    post = rows("SELECT angle_id FROM posts")[0]
    angle = rows(f"SELECT title, status FROM angles WHERE id={post['angle_id']}")[0]
    assert angle["status"] == "used" and angle["title"].startswith("Why top-k")

def test_embedding_roundtrips_as_blob():
    MemoryWriter().execute(ctx())
    blob = rows("SELECT embedding FROM posts")[0]["embedding"]
    assert E.from_blob(blob) == pytest.approx([0.6, 0.8])

def test_embeddings_off_stores_null_not_fake(monkeypatch):
    monkeypatch.setattr(E, "embed", lambda text: None)
    c = ctx()
    MemoryWriter().execute(c)
    assert c["memory_write"]["embedded"] is False
    assert rows("SELECT embedding FROM posts")[0]["embedding"] is None

def test_run_context_saved_without_bytes():
    MemoryWriter().execute(ctx())
    saved = json.loads(rows("SELECT context_json FROM runs")[0]["context_json"])
    assert saved["run_id"] == "fixture-run-001" and "memory_write" not in saved

def test_obsidian_note_written(isolated):
    MemoryWriter().execute(ctx())
    notes = list(isolated.glob("*.md"))
    assert len(notes) == 1
    body = notes[0].read_text(encoding="utf-8")
    assert "commit: 38ae3eb0c1d2" in body and "`app/retriever.py`" in body

def test_second_run_keeps_one_repo_row():
    MemoryWriter().execute(ctx())
    c2 = ctx()
    c2["run_id"] = "fixture-run-002"
    MemoryWriter().execute(c2)
    assert len(rows("SELECT * FROM repos")) == 1
    assert len(rows("SELECT * FROM posts")) == 2

def test_empty_post_rejected_and_nothing_written():
    c = ctx()
    c["approved_post"]["text"] = "   "
    rec = MemoryWriter().execute(c)
    assert rec["status"] == "failed" and "no text" in rec["error"]
    db.init()
    assert rows("SELECT * FROM posts") == []

def test_missing_inputs_fail_cleanly():
    rec = MemoryWriter().execute({"repo": {}})
    assert rec["status"] == "failed" and "run_id" in rec["error"]