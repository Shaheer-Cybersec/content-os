"""
[A09] reviser - LLM agent (heavy tier). Conditional: skips when A08 says "pass".

Input : ctx["draft"]        A07 output {text, claims[]}
        ctx["critique"]     A08 output {verdict, flags[]}
        ctx["strategy"]     A05 output (hashtags)
        ctx["repo"], ctx["run_id"]
Output: ctx["final_draft"]  {text, claims[], chars, changes[], unresolved[]}
        A10 packages final_draft when present, otherwise the original draft.

Static checks after the model answers:
  - claims re-checked with I05; at least 1 valid claim must remain
  - markdown stripped, plan hashtags kept, 3,000 char limit (same helpers as A07)
  - a "block" flag whose quote is still in the text, and was not declined with a
    reason, fails the stage (fail-closed: known-wrong text never reaches G2)
  - a "fix" flag still present and not declined goes into unresolved[] for you at G2
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from agents.critic.agent import _norm
from agents.writer.agent import LINKEDIN_MAX_CHARS, ensure_hashtags, strip_markdown
from core import llm
from core.base_agent import AgentError, BaseAgent, SkipStage
from core.evidence import validate
from core.schemas import Evidence

PROMPT_FILE = Path(__file__).with_name("prompt.md")
WHOLE_POST = "(whole post)"


class Change(BaseModel):
    quote: str
    action: Literal["applied", "declined"]
    note: str = Field(min_length=3)


class Revision(BaseModel):
    text: str = Field(min_length=200, max_length=3500)
    claims: list[Evidence] = Field(min_length=1)
    changes: list[Change] = Field(default_factory=list)


def build_user_prompt(draft: dict, flags: list[dict]) -> str:
    return "\n".join([
        "POST:", "<<<", draft["text"], ">>>", "",
        "FLAGS (fix every block and fix):",
        *[f"{i}. [{f['severity']}] \"{f['quote']}\" - {f['issue']} -> {f['suggestion']}"
          for i, f in enumerate(flags, 1)],
        "",
        "CLAIMS (the only files you may cite):",
        json.dumps(draft["claims"], indent=1),
        "",
        "Return the revision as JSON matching the schema.",
    ])


def mock_output(draft: dict, flags: list[dict]) -> dict:
    text = draft["text"]
    for f in flags:
        if f["severity"] != "nit" and f["quote"] != WHOLE_POST:
            text = text.replace(f["quote"], "Mock revised line.")
    return {"text": text, "claims": draft["claims"],
            "changes": [{"quote": f["quote"], "action": "applied", "note": "Mock: applied."}
                        for f in flags]}


def unresolved_flags(text: str, flags: list[dict], changes: list[dict]) -> tuple[list[dict], list[dict]]:
    """Return (blocking, unresolved): must-fix flags whose quote is still in the text."""
    declined = {_norm(c["quote"]) for c in changes if c["action"] == "declined"}
    body = _norm(text)
    blocking, unresolved = [], []
    for f in flags:
        q = _norm(f["quote"])
        if f["severity"] == "nit" or f["quote"] == WHOLE_POST or q in declined or q not in body:
            continue
        (blocking if f["severity"] == "block" else unresolved).append(
            {"severity": f["severity"], "quote": f["quote"], "issue": f["issue"]})
    return blocking, unresolved


class Reviser(BaseAgent):
    agent_id = "A09"
    name = "reviser"
    requires = ("draft", "critique", "strategy", "repo", "run_id")
    produces = "final_draft"
    uses_llm = True

    def run(self, ctx: dict) -> dict:
        critique, draft = ctx["critique"], ctx["draft"]
        if critique["verdict"] == "pass":
            raise SkipStage("critic verdict is pass, nothing to revise")
        flags = critique["flags"]
        r = llm.call(stage=self.name, tier="heavy",
                     system=PROMPT_FILE.read_text(encoding="utf-8"),
                     user=build_user_prompt(draft, flags),
                     schema=Revision, mock=mock_output(draft, flags), run_id=ctx["run_id"])

        check = validate(r["claims"], ctx["repo"])
        if not check["valid"]:
            raise AgentError("revision has no valid claims: " + check["invalid"][0]["reason"])
        text = ensure_hashtags(strip_markdown(r["text"]), ctx["strategy"].get("hashtags", []))
        if len(text) > LINKEDIN_MAX_CHARS:
            raise AgentError(f"revision is {len(text)} chars; LinkedIn allows {LINKEDIN_MAX_CHARS}")

        blocking, unresolved = unresolved_flags(text, flags, r["changes"])
        if blocking:
            raise AgentError(f"{len(blocking)} block flag(s) still in the text: "
                             + "; ".join(f'"{b["quote"][:50]}"' for b in blocking))
        return {"text": text, "claims": check["valid"], "chars": len(text),
                "changes": r["changes"], "unresolved": unresolved,
                "claims_dropped": [i["reason"] for i in check["invalid"]]}


# ---------- manual checkpoint: A01 -> A08, then A09, then the package (A10) ----------
if __name__ == "__main__":
    from agents.analyzer.agent import Analyzer
    from agents.angle_extractor.agent import AngleExtractor
    from agents.critic.agent import Critic
    from agents.memory_retrieve.agent import MemoryRetrieve
    from agents.packager.agent import Packager
    from agents.repo_ingest.agent import RepoIngest
    from agents.strategist.agent import Strategist
    from agents.visual_planner.agent import VisualPlanner
    from agents.writer.agent import Writer
    source = sys.argv[1] if len(sys.argv) > 1 else "https://github.com/pallets/itsdangerous"
    run_id = sys.argv[2] if len(sys.argv) > 2 else "a03-check"
    pick = int(sys.argv[3]) if len(sys.argv) > 3 else 1

    def step(agent) -> str:
        rec = agent.execute(ctx)
        print(f"{rec['id']} {rec['agent']:<16} {rec['status']:<8} {rec['error'] or ''}")
        return rec["status"]

    ctx = {"source": source, "run_id": run_id}
    for agent in (RepoIngest(), MemoryRetrieve(), Analyzer(), AngleExtractor()):
        if step(agent) != "success":
            sys.exit(0)
    usable = [a for a in ctx["angle_set"]["angles"] if a["dedup_status"] != "rejected"]
    ctx["chosen_angle"] = usable[pick - 1]
    print(f"G1  (stand-in)       picked #{pick}: {ctx['chosen_angle']['title']}")
    for agent in (Strategist(), VisualPlanner(), Writer(), Critic()):
        if step(agent) != "success":
            sys.exit(0)
    print(f"          critic verdict: {ctx['critique']['verdict']}, "
          f"{len(ctx['critique']['flags'])} flag(s)")
    status = step(Reviser())
    if status not in ("success", "skipped"):
        sys.exit(0)
    if status == "success":
        f = ctx["final_draft"]
        print(f"\n----- revised post ({f['chars']} chars) -----")
        print(f["text"])
        print("-----")
        for c in f["changes"]:
            print(f"{c['action']:<8}  \"{c['quote'][:55]}\"  {c['note']}")
        print(f"unresolved {f['unresolved'] or 'none'}")
    if step(Packager()) == "success":
        print(f"file      {ctx['package']['path']}")