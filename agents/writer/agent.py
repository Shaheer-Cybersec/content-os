"""
[A07] writer - LLM agent (heavy tier).

Input : ctx["strategy"]     A05 output (hook, beats, cta, hashtags, evidence, length_target)
        ctx["visual_plan"]  optional, A06 output {"shots": [...]}
        ctx["repo"], ctx["run_id"]
Output: ctx["draft"]        {text, claims[], chars, length_target, voice_used, notes[]}

Static checks after the model answers:
  - claims re-checked with I05; invalid ones dropped, at least 1 valid claim must remain
  - markdown the model slipped in (**bold**, `code` fences, # headings) is stripped
  - the plan's hashtags are appended if the model left them out
  - over 3,000 chars fails the stage (LinkedIn limit)
  - notes[] records soft problems for the critic (A08): length off target, hook changed
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from pydantic import BaseModel, Field

from agents.strategist.agent import voice_samples
from core import llm
from core.base_agent import AgentError, BaseAgent
from core.evidence import validate
from core.schemas import Evidence

PROMPT_FILE = Path(__file__).with_name("prompt.md")
LINKEDIN_MAX_CHARS = 3000
LENGTH_TOLERANCE = 0.25        # beyond +-25% of target -> note for the critic


class Draft(BaseModel):
    text: str = Field(min_length=200, max_length=3500)
    claims: list[Evidence] = Field(min_length=1)


def build_user_prompt(strategy: dict, visual_plan: dict | None, voice: str) -> str:
    shots = (visual_plan or {}).get("shots", []) if isinstance(visual_plan, dict) else (visual_plan or [])
    return "\n".join([
        "POST PLAN:",
        json.dumps({k: strategy.get(k) for k in
                    ("format", "hook", "beats", "length_target", "cta", "hashtags",
                     "evidence", "audience_note")}, indent=1),
        "",
        "VISUALS:",
        *([f"- {s['shot']}" for s in shots] or ["(none)"]),
        "",
        "VOICE SAMPLES (the author's real posts):",
        voice or "(none yet - plain, direct, first-person practitioner voice)",
        "",
        "Return the post as JSON matching the schema.",
    ])


def mock_output(strategy: dict) -> dict:
    filler = ("Mock filler paragraph so the draft clears the 200 character minimum "
              "that the Draft schema enforces on real model output.")
    body = "\n\n".join([strategy["hook"], *strategy["beats"], filler, strategy["cta"],
                        " ".join(strategy.get("hashtags", []))])
    return {"text": body, "claims": strategy["evidence"][:2]}


FENCE_RE = re.compile(r"^```[a-z]*\s*$", re.M)
HEADING_RE = re.compile(r"^#{1,6}\s+", re.M)       # '# Title' (space after #), not '#AppSec'
BOLD_RE = re.compile(r"\*\*(.+?)\*\*|__(.+?)__", re.S)


def strip_markdown(text: str) -> str:
    text = FENCE_RE.sub("", text)
    text = HEADING_RE.sub("", text)
    text = BOLD_RE.sub(lambda m: m.group(1) or m.group(2), text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def ensure_hashtags(text: str, tags: list[str]) -> str:
    missing = [t for t in tags if t.lower() not in text.lower()]
    return f"{text}\n\n{' '.join(missing)}" if missing else text


def review_notes(text: str, strategy: dict) -> list[str]:
    notes = []
    target = strategy.get("length_target") or 0
    if target and abs(len(text) - target) > target * LENGTH_TOLERANCE:
        notes.append(f"length {len(text)} chars vs target {target}")
    first = text.splitlines()[0].strip() if text else ""
    if strategy.get("hook") and first != strategy["hook"].strip():
        notes.append("first line differs from the planned hook")
    return notes


class Writer(BaseAgent):
    agent_id = "A07"
    name = "writer"
    requires = ("strategy", "repo", "run_id")
    produces = "draft"
    uses_llm = True

    def run(self, ctx: dict) -> dict:
        strategy = ctx["strategy"]
        voice = voice_samples()
        d = llm.call(stage=self.name, tier="heavy",
                     system=PROMPT_FILE.read_text(encoding="utf-8"),
                     user=build_user_prompt(strategy, ctx.get("visual_plan"), voice),
                     schema=Draft, mock=mock_output(strategy), run_id=ctx["run_id"])

        check = validate(d["claims"], ctx["repo"])
        if not check["valid"]:
            raise AgentError("draft has no valid claims: " + check["invalid"][0]["reason"])

        text = ensure_hashtags(strip_markdown(d["text"]), strategy.get("hashtags", []))
        if len(text) > LINKEDIN_MAX_CHARS:
            raise AgentError(f"draft is {len(text)} chars; LinkedIn allows {LINKEDIN_MAX_CHARS}")

        notes = review_notes(text, strategy)
        notes += [f"claim dropped: {i['reason']}" for i in check["invalid"]]
        return {"text": text, "claims": check["valid"], "chars": len(text),
                "length_target": strategy.get("length_target"), "voice_used": bool(voice),
                "notes": notes}


# ---------- manual checkpoint: A01 -> A06, then A07, then a preview package (A10) ----------
if __name__ == "__main__":
    from agents.analyzer.agent import Analyzer
    from agents.angle_extractor.agent import AngleExtractor
    from agents.memory_retrieve.agent import MemoryRetrieve
    from agents.packager.agent import Packager
    from agents.repo_ingest.agent import RepoIngest
    from agents.strategist.agent import Strategist
    from agents.visual_planner.agent import VisualPlanner
    source = sys.argv[1] if len(sys.argv) > 1 else "https://github.com/pallets/itsdangerous"
    run_id = sys.argv[2] if len(sys.argv) > 2 else "a03-check"
    pick = int(sys.argv[3]) if len(sys.argv) > 3 else 1

    def step(agent) -> bool:
        rec = agent.execute(ctx)
        print(f"{rec['id']} {rec['agent']:<16} {rec['status']:<8} {rec['error'] or ''}")
        return rec["status"] == "success"

    ctx = {"source": source, "run_id": run_id}
    for agent in (RepoIngest(), MemoryRetrieve(), Analyzer(), AngleExtractor()):
        if not step(agent):
            sys.exit(0)
    usable = [a for a in ctx["angle_set"]["angles"] if a["dedup_status"] != "rejected"]
    ctx["chosen_angle"] = usable[pick - 1]
    print(f"G1  (stand-in)       picked #{pick}: {ctx['chosen_angle']['title']}")
    for agent in (Strategist(), VisualPlanner(), Writer()):
        if not step(agent):
            sys.exit(0)
    d = ctx["draft"]
    print(f"\n----- draft ({d['chars']} chars, target {d['length_target']}, "
          f"voice samples used: {d['voice_used']}) -----")
    print(d["text"])
    print("-----")
    print(f"claims    {len(d['claims'])}: " + ", ".join(c["source_path"] for c in d["claims"]))
    print(f"notes     {d['notes'] or 'none'}")
    print("\npreview package (A08/A09 not built yet, so A10 packages the raw draft):")
    if step(Packager()):
        print(f"file      {ctx['package']['path']}")