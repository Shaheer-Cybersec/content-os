"""
[A04] angle_extractor - LLM agent (heavy tier).

Input : ctx["analysis"]   A03 output (grounded findings)
        ctx["memory"]     A02 output (past posts, queued angles)
        ctx["repo"], ctx["run_id"]
Output: ctx["angle_set"]  {
            "angles":  [ {title, summary, hook, post_type, audience, score, evidence[],
                          dedup_status: ok|unchecked|rejected, max_similarity, reject_reason} ],
            "counts":  {ok, unchecked, rejected, dropped},
            "dropped": [ {title, reason} ]
        }
        Ranked: ok (by score) -> unchecked -> rejected. G1 shows these to you.

Static steps after the model answers:
  1. Evidence re-check with I05. Angles with no valid evidence are dropped.
  2. Dedup against your past posts (I04 embeddings, threshold from settings).
     Embeddings unavailable -> "unchecked" (fail-closed, never a fake score).
  3. Dedup inside the batch: two angles that say the same thing -> lower score rejected.
  4. Fewer than 3 usable angles (ok + unchecked) -> the stage fails.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from config.settings import DEDUP_THRESHOLD
from core import llm
from core.base_agent import AgentError, BaseAgent
from core.evidence import validate
from core.schemas import Evidence
from memory import db
from memory import embeddings as E

PROMPT_FILE = Path(__file__).with_name("prompt.md")
MIN_USABLE = 3
FINDING_SECTIONS = ("components", "decisions", "security_notes", "limitations")


class Angle(BaseModel):
    title: str = Field(min_length=8, max_length=110)
    summary: str = Field(min_length=20, description="what the post teaches, 1-2 sentences")
    hook: str = Field(min_length=8, max_length=160, description="possible first line")
    post_type: Literal["lesson", "breakdown", "story", "how-to", "opinion"]
    audience: str = Field(min_length=3, description="who this is for")
    score: float = Field(ge=0, le=1)
    evidence: list[Evidence] = Field(min_length=1)


class AnglesOut(BaseModel):
    angles: list[Angle] = Field(min_length=5, max_length=8)


# ---------- prompt ----------

def build_user_prompt(repo: dict, analysis: dict, memory: dict) -> str:
    compact = {k: analysis.get(k) for k in ("summary", "purpose", *FINDING_SECTIONS)}
    past = memory.get("past_posts", [])
    queued = memory.get("queued_angles", [])
    lines = [
        f"REPOSITORY: {repo['owner']}/{repo['repo']}  ({repo['url']})",
        "",
        "ANALYSIS (grounded; reuse its evidence items exactly):",
        json.dumps(compact, indent=1),
        "",
        "PAST POSTS (do not repeat these ideas):",
        *([f"- {p.get('angle_title') or ''}: {p['text'][:200]}" for p in past[:20]] or ["- (none yet)"]),
        "",
        "QUEUED ANGLES for this repo (already saved, do not repeat):",
        *([f"- {a['title']}" for a in queued] or ["- (none)"]),
        "",
        "Return 5-8 distinct angles as JSON matching the schema.",
    ]
    return "\n".join(lines)


def mock_output(analysis: dict) -> dict:
    """Example angles built from the analysis' own findings, so evidence is real."""
    findings = [f for s in FINDING_SECTIONS for f in analysis.get(s, [])]
    types = ["lesson", "breakdown", "how-to", "opinion", "story"]
    angles = []
    for i, f in enumerate((findings * 3)[:6]):
        angles.append({
            "title": f"Mock angle {i + 1}: {f['title']}"[:110],
            "summary": f"Mock: a post built around '{f['title']}'.",
            "hook": f"Mock hook {i + 1} about {f['title']}"[:160],
            "post_type": types[i % len(types)],
            "audience": "security engineers",
            "score": round(0.9 - i * 0.1, 2),
            "evidence": f["evidence"][:1],
        })
    return {"angles": angles}


# ---------- static post-processing ----------

def _angle_text(a: dict) -> str:
    return f"{a['title']}. {a['summary']}"


def ground(angles: list[dict], repo: dict) -> tuple[list[dict], list[dict]]:
    kept, dropped = [], []
    for a in angles:
        check = validate(a["evidence"], repo)
        if check["valid"]:
            kept.append({**a, "evidence": check["valid"]})
        else:
            dropped.append({"title": a["title"],
                            "reason": "no valid evidence: " + check["invalid"][0]["reason"]})
    return kept, dropped


def past_vectors(memory: dict) -> list[list[float]]:
    ids = memory.get("embedded_post_ids") or []
    if not ids:
        return []
    with db.connect() as conn:
        blobs = db.post_vectors(conn, ids)
    return [E.from_blob(b) for b in blobs.values()]


def dedup(angles: list[dict], past: list[list[float]]) -> list[dict]:
    """Mark each angle ok / unchecked / rejected. Never invents a score."""
    vecs = [E.embed(_angle_text(a)) for a in angles]
    embeddings_on = all(v is not None for v in vecs)

    for a, v in zip(angles, vecs):
        a["max_similarity"], a["reject_reason"] = None, None
        if not embeddings_on:
            a["dedup_status"] = "unchecked"
            continue
        sim = max((E.cosine(v, p) for p in past), default=0.0)
        a["max_similarity"] = round(sim, 3)
        if sim > DEDUP_THRESHOLD:
            a["dedup_status"], a["reject_reason"] = "rejected", f"too close to a past post ({sim:.2f})"
        else:
            a["dedup_status"] = "ok"

    if embeddings_on:                       # inside-batch duplicates: keep the higher score
        order = sorted(range(len(angles)), key=lambda i: -angles[i]["score"])
        for pos, i in enumerate(order):
            if angles[i]["dedup_status"] == "rejected":
                continue
            for j in order[:pos]:
                if angles[j]["dedup_status"] == "rejected":
                    continue
                sim = E.cosine(vecs[i], vecs[j])
                if sim > DEDUP_THRESHOLD:
                    angles[i]["dedup_status"] = "rejected"
                    angles[i]["reject_reason"] = f"same idea as '{angles[j]['title']}' ({sim:.2f})"
                    break
    return angles


def rank(angles: list[dict]) -> list[dict]:
    order = {"ok": 0, "unchecked": 1, "rejected": 2}
    return sorted(angles, key=lambda a: (order[a["dedup_status"]], -a["score"]))


# ---------- agent ----------

class AngleExtractor(BaseAgent):
    agent_id = "A04"
    name = "angle_extractor"
    requires = ("analysis", "memory", "repo", "run_id")
    produces = "angle_set"
    uses_llm = True

    def run(self, ctx: dict) -> dict:
        repo, analysis, memory = ctx["repo"], ctx["analysis"], ctx["memory"]
        raw = llm.call(stage=self.name, tier="heavy",
                       system=PROMPT_FILE.read_text(encoding="utf-8"),
                       user=build_user_prompt(repo, analysis, memory),
                       schema=AnglesOut, mock=mock_output(analysis), run_id=ctx["run_id"])
        angles, dropped = ground(raw["angles"], repo)
        angles = rank(dedup(angles, past_vectors(memory)))
        counts = {s: sum(a["dedup_status"] == s for a in angles) for s in ("ok", "unchecked", "rejected")}
        counts["dropped"] = len(dropped)
        usable = counts["ok"] + counts["unchecked"]
        if usable < MIN_USABLE:
            raise AgentError(f"only {usable} usable angle(s) after evidence + dedup checks "
                             f"(need {MIN_USABLE}). counts={counts}")
        return {"angles": angles, "counts": counts, "dropped": dropped}


# ---------- manual checkpoint: A01 -> A02 -> A03 -> A04 ----------
if __name__ == "__main__":
    from agents.analyzer.agent import Analyzer
    from agents.memory_retrieve.agent import MemoryRetrieve
    from agents.repo_ingest.agent import RepoIngest
    source = sys.argv[1] if len(sys.argv) > 1 else "https://github.com/pallets/itsdangerous"
    run_id = sys.argv[2] if len(sys.argv) > 2 else "a03-check"
    ctx = {"source": source, "run_id": run_id}
    for agent in (RepoIngest(), MemoryRetrieve(), Analyzer(), AngleExtractor()):
        rec = agent.execute(ctx)
        print(f"{rec['id']} {rec['agent']:<16} {rec['status']:<8} {rec['error'] or ''}")
        if rec["status"] != "success":
            sys.exit(0)
    s = ctx["angle_set"]
    print(f"\nbackend  {llm.backend_for('angle_extractor')}   counts {s['counts']}")
    for a in s["angles"]:
        sim = "-" if a["max_similarity"] is None else f"{a['max_similarity']:.2f}"
        print(f"  [{a['dedup_status']:<9}] {a['score']:.2f}  sim={sim}  {a['title']}")
        print(f"               hook: {a['hook']}")
        if a["reject_reason"]:
            print(f"               why rejected: {a['reject_reason']}")
    for d in s["dropped"]:
        print(f"  [dropped  ] {d['title']}: {d['reason']}")