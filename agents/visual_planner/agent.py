"""
[A06] visual_planner - LLM agent (mid tier).

Input : ctx["strategy"]   A05 output (hook, beats)
        ctx["repo"]       A01 output (file tree)
        ctx["run_id"]
Output: ctx["visual_plan"] {"shots": [{shot, kind, how, file_path, lines, minutes}],
                            "rationale": str, "dropped": [{shot, reason}]}

Static checks after the model answers:
  - any file_path must exist in the repo tree; code shots must have one
  - minutes clamped to 1-5 per shot, at most 3 shots
Shots that fail are dropped (with the reason), never passed to the packager.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field

from core import llm
from core.base_agent import BaseAgent

PROMPT_FILE = Path(__file__).with_name("prompt.md")
MAX_SHOTS = 3


class Shot(BaseModel):
    shot: str = Field(min_length=4, max_length=100, description="what the image shows")
    kind: Literal["terminal", "code", "diagram", "text-card"]
    how: str = Field(min_length=10, description="concrete steps to produce it")
    file_path: Optional[str] = Field(default=None, description="repo file, required for code")
    lines: Optional[str] = Field(default=None, description="e.g. '24-40' for code crops")
    minutes: int = Field(ge=1, le=15)


class Graphic(BaseModel):
    style: Literal["terminal", "quote", "stat"] = Field(description="card template")
    headline: str = Field(min_length=4, max_length=90, description="the one line the image says")
    subline: Optional[str] = Field(default=None, max_length=140)
    stat: Optional[str] = Field(default=None, max_length=24, description="a big number/term for 'stat' style")


class ImagePrompt(BaseModel):
    tool: str = Field(min_length=2, max_length=40, description="e.g. Canva, ChatGPT image, Gemini, Ideogram")
    prompt: str = Field(min_length=20, max_length=1200)


class VisualPlan(BaseModel):
    shots: list[Shot] = Field(default_factory=list, max_length=5)
    rationale: str = Field(min_length=10)
    graphic: Optional[Graphic] = Field(default=None, description="a simple post card the dashboard renders")
    image_prompts: list[ImagePrompt] = Field(default_factory=list, max_length=3)


def build_user_prompt(strategy: dict, repo: dict) -> str:
    return "\n".join([
        "POST PLAN:",
        json.dumps({k: strategy.get(k) for k in ("format", "hook", "beats", "evidence")}, indent=1),
        "",
        f"FILE TREE of {repo['owner']}/{repo['repo']}:",
        *[f"- {f['path']}" for f in repo.get("tree", [])],
        "",
        "Return the visual plan as JSON matching the schema.",
    ])


def mock_output(strategy: dict) -> dict:
    ev = strategy["evidence"][0]
    return {"shots": [{"shot": "Mock code crop", "kind": "code",
                       "how": "Mock: open the file in VS Code, zoom 150%, crop the lines.",
                       "file_path": ev["source_path"], "lines": "1-20", "minutes": 3}],
            "rationale": "Mock: one code shot proves the main fact.",
            "graphic": {"style": "terminal", "headline": f"Mock headline: {strategy['hook']}"[:90], "subline": "Mock subline"},
            "image_prompts": [{"tool": "Canva", "prompt": "Mock: dark terminal-style LinkedIn post card, green monospace text, one headline line, lots of empty space."}]}


def check_shots(shots: list[dict], repo: dict) -> tuple[list[dict], list[dict]]:
    paths = {f["path"] for f in repo.get("tree", [])}
    kept, dropped = [], []
    for s in shots:
        fp = (s.get("file_path") or "").replace("\\", "/") or None
        s["file_path"] = fp
        if s["kind"] == "code" and not fp:
            dropped.append({"shot": s["shot"], "reason": "code shot without file_path"})
        elif fp and fp not in paths:
            dropped.append({"shot": s["shot"], "reason": f"file not in repo: {fp}"})
        elif len(kept) >= MAX_SHOTS:
            dropped.append({"shot": s["shot"], "reason": f"more than {MAX_SHOTS} shots"})
        else:
            s["minutes"] = max(1, min(5, s["minutes"]))
            kept.append(s)
    return kept, dropped


class VisualPlanner(BaseAgent):
    agent_id = "A06"
    name = "visual_planner"
    requires = ("strategy", "repo", "run_id")
    produces = "visual_plan"
    uses_llm = True

    def run(self, ctx: dict) -> dict:
        plan = llm.call(stage=self.name, tier="mid",
                        system=PROMPT_FILE.read_text(encoding="utf-8"),
                        user=build_user_prompt(ctx["strategy"], ctx["repo"]),
                        schema=VisualPlan, mock=mock_output(ctx["strategy"]), run_id=ctx["run_id"])
        shots, dropped = check_shots(plan["shots"], ctx["repo"])
        return {"shots": shots, "rationale": plan["rationale"], "dropped": dropped,
                "graphic": plan.get("graphic"), "image_prompts": plan.get("image_prompts") or []}


# ---------- manual checkpoint: A01 -> A05, then A06 ----------
if __name__ == "__main__":
    from agents.analyzer.agent import Analyzer
    from agents.angle_extractor.agent import AngleExtractor
    from agents.memory_retrieve.agent import MemoryRetrieve
    from agents.repo_ingest.agent import RepoIngest
    from agents.strategist.agent import Strategist
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
    for agent in (Strategist(), VisualPlanner()):
        rec = agent.execute(ctx)
        print(f"{rec['id']} {rec['agent']:<16} {rec['status']:<8} {rec['error'] or ''}")
        if rec["status"] != "success":
            sys.exit(0)
    v = ctx["visual_plan"]
    print(f"\nrationale  {v['rationale']}")
    for i, s in enumerate(v["shots"], 1):
        where = f"  [{s['file_path']} {s['lines'] or ''}]" if s["file_path"] else ""
        print(f"shot {i}  ({s['kind']}, {s['minutes']} min) {s['shot']}{where}")
        print(f"         how: {s['how']}")
    for d in v["dropped"]:
        print(f"dropped  {d['shot']}: {d['reason']}")
