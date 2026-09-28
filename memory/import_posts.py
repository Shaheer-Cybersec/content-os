"""
[L01] import_posts - load your real, already-published LinkedIn posts into memory.

Why: A04 dedups new angles against every post in memory. Posts you wrote by hand
before Content OS existed are not there, so the pipeline could suggest a post you
already published (e.g. the damn-vulnerable-rag refund injection). This puts them in.

Reads every "## Post ..." section of memory/voice/voice_samples.md (text up to the
next "---" line), stores it under repo "manual/linkedin" with no run or angle,
and embeds it for dedup. Idempotent: a post whose exact text is already stored is skipped.

  python -m memory.import_posts            import
  python -m memory.import_posts --dry-run  show what would be imported
"""
from __future__ import annotations

import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from config.settings import VOICE_DIR
from memory import db
from memory import embeddings as E

REPO_ID = "manual/linkedin"
REPO_URL = "https://www.linkedin.com/in/shaheer-hussain-cybersec/"
SECTION_RE = re.compile(r"^## Post[^\n]*\n(.*?)(?=^---\s*$|\Z)", re.M | re.S)


def parse_posts(text: str) -> list[str]:
    return [m.group(1).strip() for m in SECTION_RE.finditer(text) if m.group(1).strip()]


def import_posts(path: Path | None = None, dry_run: bool = False) -> dict:
    path = path or VOICE_DIR / "voice_samples.md"
    if not path.exists():
        raise SystemExit(f"no voice file at {path}")
    posts = parse_posts(path.read_text(encoding="utf-8"))
    db.init()
    added = skipped = embedded = 0
    with db.connect() as conn:
        existing = {r["text"] for r in conn.execute("SELECT text FROM posts")}
        if not dry_run:
            db.upsert_repo(conn, REPO_ID, REPO_URL, None, datetime.now(timezone.utc).isoformat())
        for text in posts:
            if text in existing:
                skipped += 1
                continue
            added += 1
            if dry_run:
                continue
            vec = E.embed(text)
            embedded += vec is not None
            db.insert_post(conn, REPO_ID, None, None, text,
                           E.to_blob(vec) if vec is not None else None)
    return {"found": len(posts), "added": added, "skipped": skipped,
            "embedded": embedded, "dry_run": dry_run}


if __name__ == "__main__":
    r = import_posts(dry_run="--dry-run" in sys.argv)
    print(f"found {r['found']} post(s): added {r['added']}, already there {r['skipped']}, "
          f"embedded {r['embedded']}" + ("  (dry run, nothing written)" if r["dry_run"] else ""))