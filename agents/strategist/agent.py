"""
[A05] strategist - LLM agent (mid tier).

Input : ctx["chosen_angle"]  the angle you picked at G1
        ctx["analysis"]      A03 output (context for the angle)
        ctx["repo"], ctx["run_id"]
Output: ctx["strategy"]      {format, hook, beats[], length_target, cta, hashtags[],
                              evidence[], audience_note}

Static checks after the model answers:
  - evidence re-checked with I05; at least 1 valid item must remain
  - hashtags normalised to '#Word', max 3, generic tags removed
  - length_target clamped to 700-1800
Voice samples (memory/voice/voice_samples.md) are included when present, so the hook
and CTA already sound like you before the writer starts.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from config.settings import VOICE_DIR
from core import llm
from core.base_agent import AgentError, BaseAgent
from core.evidence import validate
from core.schemas import Evidence

PROMPT_FILE = Path(__file__).with_name("prompt.md")
VOICE_FILE = VOICE_DIR / "voice_samples.md"
VOICE_CHARS = 4_000
MIN_LEN, MAX_LEN = 700, 1800
MAX_TAGS = 3
GENERIC_TAGS = {"#tech", "#technology", "#motivation", "#success", "#linkedin",
                "#innovation", "#growth", "#mindset"}


class Strategy(BaseModel):
    format: Literal["story", "lesson-list", "myth-vs-fact", "how-to", "breakdown", "hot-take"]
    hook: str = Field(min_length=8, max_length=160)
    beats: list[str] = Field(min_length=3, max_length=6)
    length_target: int = Field(ge=300, le=3000)
    cta: str = Field(min_length=8, max_length=200)
    hashtags: list[str] = Field(default_factory=list, max_length=6)
    evidence: list[Evidence] = Field(min_length=1)
    audience_note: str = Field(min_length=8)


def voice_samples() -> str:
    if VOICE_FILE.exists():
        return VOICE_FILE.read_text(encoding="utf-8")[:VOICE_CHARS]
    return ""


def build_user_prompt(angle: dict, analysis: dict, voice: str) -> str:
    return "\n".join([
        "CHOSEN ANGLE:",
        json.dumps({k: angle.get(k) for k in
                    ("title", "summary", "hook", "post_type", "audience", "evidence")}, indent=1),
        "",
        "PROJECT CONTEXT:",
        f"summary: {analysis.get('summary', '')}",
        f"purpose: {analysis.get('purpose', '')}",
        "",
        "VOICE SAMPLES (the author's real posts):",
        voice or "(none yet - write in plain, direct, first-person practitioner voice)",
        "",
        "Return the post plan as JSON matching the schema.",
    ])


def mock_output(angle: dict) -> dict:
    return {
        "format": "lesson-list",
        "hook": angle.get("hook") or "Mock hook for testing.",
        "beats": ["Mock beat: the fact", "Mock beat: why it matters", "Mock beat: what to check"],
        "length_target": 1100,
        "cta": "Mock CTA: what would you check first?",
        "hashtags": ["#AppSec", "#Python"],
        "evidence": angle["evidence"][:2],
        "audience_note": "Mock: developers who ship web apps.",
    }


def clean_hashtags(tags: list[str]) -> list[str]:
    out = []
    for t in tags:
        word = re.sub(r"[^A-Za-z0-9]", "", t)
        tag = f"#{word}"
        if word and tag.lower() not in GENERIC_TAGS and tag.lower() not in {x.lower() for x in out}:
            out.append(tag)
    return out[:MAX_TAGS]


class Strategist(BaseAgent):
    agent_id = "A05"
    name = "strategist"
    requires = ("chosen_angle", "analysis", "repo", "run_id")
    produces = "strategy"
    uses_llm = True

    def run(self, ctx: dict) -> dict:
        angle = ctx["chosen_angle"]
        s = llm.call(stage=self.name, tier="mid",
                     system=PROMPT_FILE.read_text(encoding="utf-8"),
                     user=build_user_prompt(angle, ctx["analysis"], voice_samples()),
                     schema=Strategy, mock=mock_output(angle), run_id=ctx["run_id"])
        check = validate(s["evidence"], ctx["repo"])
        if not check["valid"]:
            raise AgentError("strategy has no valid evidence: " + check["invalid"][0]["reason"])
        s["evidence"] = check["valid"]
        s["hashtags"] = clean_hashtags(s["hashtags"])
        s["length_target"] = max(MIN_LEN, min(MAX_LEN, s["length_target"]))
        s["voice_used"] = bool(voice_samples())
        return s


# ---------- manual checkpoint: A01 -> A04, pick an angle, A05 ----------
if __name__ == "__main__":
    from agents.analyzer.agent import Analyzer
    from agents.angle_extractor.agent import AngleExtractor
    from agents.memory_retrieve.agent import MemoryRetrieve
    from agents.repo_ingest.agent import RepoIngest
    source = sys.argv[1] if len(sys.argv) > 1 else "https://github.com/pallets/itsdangerous"
    run_id = sys.argv[2] if len(sys.argv) > 2 else "a03-check"
    pick = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    ctx = {"source": source, "run_id": run_id}
    for agent in (RepoIngest(), MemoryRetrieve(), Analyzer(), AngleExtractor()):
        rec = agent.execute(ctx)
        print(f"{rec['id']} {rec['agent']:<16} {rec['status']:<8} {rec['error'] or ''}")
        if rec["status"] != "success":
            sys.exit(0)
    usable = [a for a in ctx["angle_set"]["angles"] if a["dedup_status"] != "rejected"]
    ctx["chosen_angle"] = usable[pick - 1]
    print(f"G1  (stand-in)       picked #{pick}: {ctx['chosen_angle']['title']}")
    rec = Strategist().execute(ctx)
    print(f"{rec['id']} {rec['agent']:<16} {rec['status']:<8} {rec['error'] or ''}")
    if rec["status"] == "success":
        s = ctx["strategy"]
        print(f"\nformat    {s['format']}   length {s['length_target']}   voice samples used: {s['voice_used']}")
        print(f"hook      {s['hook']}")
        for i, b in enumerate(s["beats"], 1):
            print(f"beat {i}    {b}")
        print(f"cta       {s['cta']}")
        print(f"hashtags  {' '.join(s['hashtags']) or '(none)'}")
        print(f"evidence  {len(s['evidence'])} item(s): {', '.join(e['source_path'] for e in s['evidence'])}")