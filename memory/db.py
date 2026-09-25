"""
[I03] SQLite store - the pipeline's long-term memory.

Four tables:
  repos   one row per GitHub repo we have ingested
  runs    one row per pipeline run; holds run state so a run can resume
  angles  every angle A04 proposed; unused ones stay 'queued' for later posts
  posts   every approved post, with its embedding for dedup (I04)

init() is idempotent: safe to call on every start.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

from config.settings import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS repos (
    id           TEXT PRIMARY KEY,              -- "owner/repo"
    url          TEXT NOT NULL,
    last_sha     TEXT,
    ingested_at  TEXT
);

CREATE TABLE IF NOT EXISTS runs (
    id           TEXT PRIMARY KEY,              -- e.g. "20260925-0540-ab12"
    repo_id      TEXT NOT NULL REFERENCES repos(id),
    status       TEXT NOT NULL DEFAULT 'running'
                 CHECK (status IN ('running','waiting_on_you','done','failed','rejected')),
    last_stage   TEXT,                          -- last stage that succeeded
    context_json TEXT NOT NULL DEFAULT '{}',    -- full run context, for resume
    created_at   TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at   TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS angles (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    repo_id      TEXT NOT NULL REFERENCES repos(id),
    run_id       TEXT REFERENCES runs(id),
    title        TEXT NOT NULL,
    summary      TEXT,
    score        REAL,
    dedup_status TEXT CHECK (dedup_status IN ('ok','rejected','unchecked')),
    status       TEXT NOT NULL DEFAULT 'queued'
                 CHECK (status IN ('queued','used','dropped')),
    created_at   TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS posts (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    repo_id      TEXT NOT NULL REFERENCES repos(id),
    run_id       TEXT REFERENCES runs(id),
    angle_id     INTEGER REFERENCES angles(id),
    text         TEXT NOT NULL,
    embedding    BLOB,                          -- float32 bytes; never goes into run context
    created_at   TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


def connect(path: Path | str | None = None) -> sqlite3.Connection:
    """Open the database. Rows behave like dicts; foreign keys are enforced."""
    conn = sqlite3.connect(path or DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init(path: Path | str | None = None) -> None:
    """Create any missing tables. Existing data is never touched."""
    with connect(path) as conn:
        conn.executescript(SCHEMA)


def tables(path: Path | str | None = None) -> list[str]:
    with connect(path) as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
    return [r["name"] for r in rows]


if __name__ == "__main__":
    init()
    print(f"database  {DB_PATH}")
    for t in tables():
        with connect() as c:
            n = c.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        print(f"table     {t:<8} rows={n}")