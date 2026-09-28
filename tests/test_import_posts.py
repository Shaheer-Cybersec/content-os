"""[L01] import of hand-written posts into memory (so dedup knows about them)."""
import pytest

from memory import db
from memory import embeddings as E
from memory import import_posts as IP

VOICE = """# Voice samples

## Observed patterns
- short lines

---

## Post 1: first

Line one of post one.
#Tag

---

## Comment: not a post

Should not import.

---

## Post 2: second

Post two body.
"""


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "t.db")
    monkeypatch.setattr(E, "embed", lambda text: [0.6, 0.8])
    p = tmp_path / "voice_samples.md"
    p.write_text(VOICE, encoding="utf-8")
    return p


def test_parse_only_post_sections():
    assert IP.parse_posts(VOICE) == ["Line one of post one.\n#Tag", "Post two body."]

def test_import_then_idempotent(env):
    r = IP.import_posts(env)
    assert (r["found"], r["added"], r["embedded"]) == (2, 2, 2)
    with db.connect() as conn:
        rows = conn.execute("SELECT repo_id, run_id, embedding IS NOT NULL AS e FROM posts").fetchall()
    assert [tuple(r) for r in rows] == [("manual/linkedin", None, 1)] * 2
    assert IP.import_posts(env)["skipped"] == 2

def test_dry_run_writes_nothing(env):
    assert IP.import_posts(env, dry_run=True)["added"] == 2
    with db.connect() as conn:
        assert conn.execute("SELECT count(*) FROM posts").fetchone()[0] == 0