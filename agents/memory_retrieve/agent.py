"""
[A02] memory_retrieve - static agent (no LLM)

Input : ctx["repo"]     A01 output (needs owner + repo)
Output: ctx["memory"]   {
            "repo_id": "owner/repo",
            "past_posts": [{id, repo_id, text, created_at, angle_title, has_embedding}],
            "past_post_count": int,
            "queued_angles": [{id, title, summary, score, dedup_status}],
            "embedded_post_ids": [int]
        }

Two jobs:
  - past_posts: your recent posts from ALL repos, so A04 can avoid repeating an idea
    you already posted about a different project.
  - queued_angles: unused angles from earlier runs of THIS repo, so a second post
    from the same repo can skip straight to a ready angle.

Embedding bytes never enter the context. A04 loads vectors itself with
db.post_vectors(embedded_post_ids).

A brand-new database is not an error: first run -> empty lists.
"""
from __future__ import annotations

import json
import sys

from config.settings import DATA_DIR
from core.base_agent import BaseAgent
from memory import db

RECENT_LIMIT = 50


class MemoryRetrieve(BaseAgent):
    agent_id = "A02"
    name = "memory_retrieve"
    requires = ("repo",)
    produces = "memory"
    uses_llm = False

    def run(self, ctx: dict) -> dict:
        repo = ctx["repo"]
        repo_id = f"{repo['owner']}/{repo['repo']}"
        db.init()                                   # first run: creates empty tables
        with db.connect() as conn:
            posts = db.recent_posts(conn, RECENT_LIMIT)
            queued = db.queued_angles(conn, repo_id)
        return {
            "repo_id": repo_id,
            "past_posts": posts,
            "past_post_count": len(posts),
            "queued_angles": queued,
            "embedded_post_ids": [p["id"] for p in posts if p["has_embedding"]],
        }


if __name__ == "__main__":
    if "--fixture" not in sys.argv:
        sys.exit("usage: python -m agents.memory_retrieve.agent --fixture")
    db.DB_PATH = DATA_DIR / "fixture.db"            # reads what A11 --fixture wrote
    ctx = {"repo": {"owner": "Shaheer-Cybersec", "repo": "damn-vulnerable-rag"}}
    rec = MemoryRetrieve().execute(ctx)
    print(json.dumps(rec, indent=2))
    if rec["status"] == "success":
        m = ctx["memory"]
        print(f"repo_id           {m['repo_id']}")
        print(f"past posts        {m['past_post_count']}")
        for p in m["past_posts"][:3]:
            print(f"  #{p['id']:<3} {p['angle_title'] or '-'}  (embedded={p['has_embedding']})")
        print(f"queued angles     {len(m['queued_angles'])}")
        for a in m["queued_angles"]:
            print(f"  {a['score']:.2f}  {a['title']}  [{a['dedup_status']}]")
        print(f"embedded ids      {m['embedded_post_ids']}")
        json.dumps(ctx)                             # proves the whole context is JSON-safe
        print("context JSON-safe  yes")
        