"""
[A11] memory_writer - static agent (no LLM). Runs last, only after you approve (G2).

Input : ctx["repo"]           A01 output
        ctx["run_id"]         this run's id
        ctx["approved_post"]  {"text": str, "evidence": [Evidence dicts]}
        ctx["chosen_angle"]   optional {"title", "summary", "score", "dedup_status"}
        ctx["angles"]         optional, every angle A04 proposed (same shape)
Output: ctx["memory_write"]   {"post_id", "angles_used", "angles_queued",
                               "angles_dropped", "embedded", "obsidian_note"}

What it does, in ONE database transaction (all or nothing):
  1. upsert the repo row
  2. mark the run 'done'
  3. store the chosen angle as 'used', the others as 'queued'
     (angles the dedup gate rejected are stored as 'dropped')
  4. store the post text + its embedding (NULL if embeddings are off: fail-closed)
Then, outside the transaction, it writes a read-only markdown note for Obsidian.
"""
from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone

from config.settings import DATA_DIR, OBSIDIAN_DIR
from core.base_agent import AgentError, BaseAgent
from memory import db
from memory import embeddings as E


def _slug(text: str, n: int = 40) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:n] or "post"


def _angle_fields(a: dict) -> tuple:
    return (a.get("title", "").strip(), a.get("summary"), a.get("score"), a.get("dedup_status"))


def obsidian_note(repo: dict, run_id: str, post_id: int, angle: dict | None,
                  post: dict, embedded: bool) -> str:
    title = (angle or {}).get("title") or "Untitled angle"
    lines = [
        "---",
        f"repo: {repo['owner']}/{repo['repo']}",
        f"source: {repo.get('source_type', 'git')}",
        f"commit: {repo['head_sha'][:12]}",
        f"run: {run_id}",
        f"post_id: {post_id}",
        f"embedded: {str(embedded).lower()}",
        f"written: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "---",
        "",
        f"# {title}",
        "",
        post["text"].strip(),
        "",
        "## Evidence",
        "",
    ]
    ev = post.get("evidence") or []
    lines += [f"- {e['claim']} - `{e['source_path']}` @ `{e['source_ref']}`" for e in ev] \
        or ["- (none recorded)"]
    return "\n".join(lines) + "\n"


class MemoryWriter(BaseAgent):
    agent_id = "A11"
    name = "memory_writer"
    requires = ("repo", "run_id", "approved_post")
    produces = "memory_write"
    uses_llm = False

    def run(self, ctx: dict) -> dict:
        repo, run_id, post = ctx["repo"], ctx["run_id"], ctx["approved_post"]
        text = (post.get("text") or "").strip()
        if not text:
            raise AgentError("approved_post has no text")

        repo_id = f"{repo['owner']}/{repo['repo']}"
        chosen = ctx.get("chosen_angle")
        others = [a for a in ctx.get("angles", [])
                  if not chosen or a.get("title") != chosen.get("title")]

        vec = E.embed(text)                        # None when embeddings are off
        blob = E.to_blob(vec) if vec is not None else None

        # Context saved for the run record: never the embedding, never raw bytes
        saved_ctx = {k: v for k, v in ctx.items() if k != "memory_write"}

        db.init()
        used = queued = dropped = 0
        with db.connect() as conn:                 # one transaction: all or nothing
            db.upsert_repo(conn, repo_id, repo["url"], repo["head_sha"], repo["ingested_at"])
            db.finish_run(conn, run_id, repo_id, self.name, json.dumps(saved_ctx))

            angle_id = None
            if chosen:
                angle_id = db.insert_angle(conn, repo_id, run_id, *_angle_fields(chosen), "used")
                used = 1
            for a in others:
                title = a.get("title", "").strip()
                if not title:
                    continue
                status = "dropped" if a.get("dedup_status") == "rejected" else "queued"
                db.insert_angle(conn, repo_id, run_id, *_angle_fields(a), status)
                queued += status == "queued"
                dropped += status == "dropped"

            post_id = db.insert_post(conn, repo_id, run_id, angle_id, text, blob)

        note_name = f"{datetime.now():%Y-%m-%d}-{repo['repo']}-{post_id}-{_slug((chosen or {}).get('title', ''))}.md"
        note_path = OBSIDIAN_DIR / note_name
        note_path.write_text(obsidian_note(repo, run_id, post_id, chosen, post, blob is not None),
                             encoding="utf-8")

        return {
            "post_id": post_id,
            "angles_used": used,
            "angles_queued": queued,
            "angles_dropped": dropped,
            "embedded": blob is not None,
            "obsidian_note": note_path.relative_to(DATA_DIR.parent).as_posix(),
        }


# ---------- fixture: a fake approved run, written to data/fixture.db ----------

FIXTURE = {
    "run_id": "fixture-run-001",
    "repo": {
        "source_type": "git", "owner": "Shaheer-Cybersec", "repo": "damn-vulnerable-rag",
        "url": "https://github.com/Shaheer-Cybersec/damn-vulnerable-rag",
        "head_sha": "38ae3eb0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6",
        "ingested_at": "2026-09-25T00:00:00+00:00",
    },
    "chosen_angle": {"title": "Why top-k size changes indirect injection risk",
                     "summary": "Retrieving more chunks widens the attack surface.",
                     "score": 0.91, "dedup_status": "ok"},
    "angles": [
        {"title": "Why top-k size changes indirect injection risk", "score": 0.91, "dedup_status": "ok"},
        {"title": "Building a RAG lab that is vulnerable on purpose", "score": 0.84, "dedup_status": "ok"},
        {"title": "Three payloads that survived chunking", "score": 0.78, "dedup_status": "unchecked"},
        {"title": "I built a vulnerable RAG app", "score": 0.70, "dedup_status": "rejected"},
    ],
    "approved_post": {
        "text": "I made my RAG lab retrieve 8 chunks instead of 3. Injection success went up.",
        "evidence": [{"claim": "Retriever uses top_k", "source_path": "app/retriever.py",
                      "source_ref": "38ae3eb"}],
    },
}

if __name__ == "__main__":
    if "--fixture" not in sys.argv:
        sys.exit("usage: python -m agents.memory_writer.agent --fixture")
    db.DB_PATH = DATA_DIR / "fixture.db"          # never touch the real database
    ctx = json.loads(json.dumps(FIXTURE))
    rec = MemoryWriter().execute(ctx)
    print(json.dumps(rec, indent=2))
    if rec["status"] == "success":
        print(json.dumps(ctx["memory_write"], indent=2))
        with db.connect() as c:
            for t in ("repos", "runs", "angles", "posts"):
                print(f"{t:<7} rows={c.execute(f'SELECT COUNT(*) FROM {t}').fetchone()[0]}")
            for r in c.execute("SELECT status, COUNT(*) n FROM angles GROUP BY status ORDER BY status"):
                print(f"  angles {r['status']:<8} {r['n']}")