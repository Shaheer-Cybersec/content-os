"""[I03] checkpoint tests for the SQLite store. Uses a temp DB, never your real one."""
import sqlite3

import pytest

from memory import db


@pytest.fixture
def path(tmp_path):
    p = tmp_path / "test.db"
    db.init(p)
    return p

def test_four_tables(path):
    assert db.tables(path) == ["angles", "posts", "repos", "runs"]

def test_init_is_idempotent(path):
    with db.connect(path) as c:
        c.execute("INSERT INTO repos (id, url) VALUES ('a/b', 'https://github.com/a/b')")
    db.init(path)                                   # second init must not wipe data
    with db.connect(path) as c:
        assert c.execute("SELECT COUNT(*) FROM repos").fetchone()[0] == 1

def test_foreign_keys_enforced(path):
    with pytest.raises(sqlite3.IntegrityError):
        with db.connect(path) as c:
            c.execute("INSERT INTO posts (repo_id, text) VALUES ('ghost/repo', 'hi')")

def test_status_values_checked(path):
    with db.connect(path) as c:
        c.execute("INSERT INTO repos (id, url) VALUES ('a/b', 'u')")
    with pytest.raises(sqlite3.IntegrityError):
        with db.connect(path) as c:
            c.execute("INSERT INTO runs (id, repo_id, status) VALUES ('r1', 'a/b', 'maybe')")