"""
[A08] critic - static pre-pass + LLM review (mid tier).

Input : ctx["draft"]     A07 output {text, claims[], notes[]}
        ctx["strategy"]  A05 output (the plan the draft must fit)
        ctx["run_id"]
Output: ctx["critique"]  {verdict, flags[], scores{}, dropped[]}
        verdict "revise" if any block/fix flag exists, else "pass" (A09 then skips)

Static pre-pass (code, no LLM), runs first and is passed to the model:
  - AI-tell phrases, emojis, em dashes, leftover markdown, links in the body
  - hook longer than the ~200 char "see more" fold
  - more than 3 hashtags; writer notes[] carried over as "fix" flags
After the model answers:
  - a flag whose quote is not found in the post is dropped (the model cannot
    critique words that are not there)
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from core import llm
from core.base_agent import BaseAgent

PROMPT_FILE = Path(__file__).with_name("prompt.md")
FOLD_CHARS = 200
MAX_HASHTAGS = 3
AI_TELLS = ("delve", "game-changer", "game changer", "in today's", "let's dive", "dive in",
            "unlock", "leverage", "here's the thing", "in the ever", "landscape",
            "it's important to note", "seamless", "robust solution", "elevate", "navigate the")
EMOJI_RE = re.compile("[\U0001F300-\U0001FAFF\u2600-\u27BF]")
LINK_RE = re.compile(r"https?://\S+")
MD_RE = re.compile(r"\*\*|__|```|^#{1,6}\s", re.M)
HASHTAG_RE = re.compile(r"(?<!\w)#\w+")


class Flag(BaseModel):
    severity: Literal["block", "fix", "nit"]
    quote: str = Field(min_length=1, max_length=200)
    issue: str = Field(min_length=5)
    suggestion: str = Field(min_length=3)


class Scores(BaseModel):
    hook: int = Field(ge=1, le=5)
    clarity: int = Field(ge=1, le=5)
    accuracy: int = Field(ge=1, le=5)
    voice: int = Field(ge=1, le=5)


class Critique(BaseModel):
    flags: list[Flag] = Field(default_factory=list, max_length=15)
    scores: Scores


def static_flags(text: str, notes: list[str]) -> list[dict]:
    flags = []

    def add(sev, quote, issue, suggestion):
        flags.append({"severity": sev, "quote": quote, "issue": issue,
                      "suggestion": suggestion, "source": "static"})

    low = text.lower()
    for phrase in AI_TELLS:
        i = low.find(phrase)
        if i >= 0:
            add("fix", text[i:i + len(phrase)], f"AI-tell phrase '{phrase}'", "say it plainly")
    for e in sorted(set(EMOJI_RE.findall(text))):
        add("fix", e, "emoji in the post", "remove it")
    if text.count("\u2014") > 1:
        add("nit", "\u2014", f"{text.count(chr(0x2014))} em dashes read as generated text",
            "use full stops or commas")
    for m in MD_RE.finditer(text):
        add("fix", m.group(0).strip() or m.group(0), "markdown LinkedIn will not render", "remove it")
    for link in LINK_RE.findall(text):
        add("fix", link, "link in the body cuts reach", "move it to the first comment")
    first = text.split("\n\n")[0]
    if len(first) > FOLD_CHARS:
        add("fix", first[:120], f"hook block is {len(first)} chars, past the ~{FOLD_CHARS} char fold",
            "cut the opening to one or two short lines")
    tags = HASHTAG_RE.findall(text)
    if len(tags) > MAX_HASHTAGS:
        add("nit", " ".join(tags), f"{len(tags)} hashtags", f"keep {MAX_HASHTAGS} or fewer")
    for n in notes:
        if n.startswith("absolute wording"):          # heuristic word check: polish, not a revision trigger
            q = re.search(r'"([^"]+)"', n)
            add("nit", q.group(1) if q else "(whole post)", f"writer note: {n}",
                "keep it only if the claims support the absolute")
        else:
            add("fix", "(whole post)", f"writer note: {n}", "address it in the revision")
    return flags


def build_user_prompt(draft: dict, strategy: dict, sflags: list[dict]) -> str:
    return "\n".join([
        "POST:", "<<<", draft["text"], ">>>", "",
        "PLAN:",
        json.dumps({k: strategy.get(k) for k in ("format", "hook", "beats", "cta", "audience_note")},
                   indent=1),
        "",
        "CLAIMS (what the post may state about the code):",
        *[f"- {c['claim']}  [{c['source_path']} @ {c['source_ref']}]" for c in draft["claims"]],
        "",
        "STATIC FLAGS (already found by code, do not repeat):",
        *([f"- {f['severity']}: {f['issue']}" for f in sflags] or ["(none)"]),
        "",
        "Return the critique as JSON matching the schema.",
    ])


def mock_output(draft: dict) -> dict:
    first = draft["text"].splitlines()[0][:60]
    return {"flags": [{"severity": "nit", "quote": first, "issue": "Mock: hook could be sharper.",
                       "suggestion": "Mock: lead with the command."}],
            "scores": {"hook": 4, "clarity": 4, "accuracy": 5, "voice": 4}}


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')).strip()


def check_quotes(flags: list[dict], text: str) -> tuple[list[dict], list[dict]]:
    body = _norm(text)
    kept, dropped = [], []
    for f in flags:
        (kept if _norm(f["quote"]) in body else dropped).append(f)
    return kept, [{"quote": f["quote"], "reason": "quote not found in post"} for f in dropped]


class Critic(BaseAgent):
    agent_id = "A08"
    name = "critic"
    requires = ("draft", "strategy", "run_id")
    produces = "critique"
    uses_llm = True

    def run(self, ctx: dict) -> dict:
        draft = ctx["draft"]
        sflags = static_flags(draft["text"], draft.get("notes", []))
        c = llm.call(stage=self.name, tier="mid",
                     system=PROMPT_FILE.read_text(encoding="utf-8"),
                     user=build_user_prompt(draft, ctx["strategy"], sflags),
                     schema=Critique, mock=mock_output(draft), run_id=ctx["run_id"])
        llm_flags, dropped = check_quotes(c["flags"], draft["text"])
        for f in llm_flags:
            f["source"] = "model"
        order = {"block": 0, "fix": 1, "nit": 2}
        flags = sorted(sflags + llm_flags, key=lambda f: order[f["severity"]])
        verdict = "revise" if any(f["severity"] in ("block", "fix") for f in flags) else "pass"
        return {"verdict": verdict, "flags": flags, "scores": c["scores"], "dropped": dropped}


# ---------- manual checkpoint: A01 -> A07, then A08 ----------
if __name__ == "__main__":
    from agents.analyzer.agent import Analyzer
    from agents.angle_extractor.agent import AngleExtractor
    from agents.memory_retrieve.agent import MemoryRetrieve
    from agents.repo_ingest.agent import RepoIngest
    from agents.strategist.agent import Strategist
    from agents.visual_planner.agent import VisualPlanner
    from agents.writer.agent import Writer
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
    for agent in (Strategist(), VisualPlanner(), Writer(), Critic()):
        if not step(agent):
            sys.exit(0)
    cr = ctx["critique"]
    s = cr["scores"]
    print(f"\nverdict   {cr['verdict']}   scores hook {s['hook']} clarity {s['clarity']} "
          f"accuracy {s['accuracy']} voice {s['voice']}")
    for f in cr["flags"]:
        print(f"[{f['severity']:<5}] ({f['source']}) \"{f['quote'][:70]}\"")
        print(f"          {f['issue']}  ->  {f['suggestion']}")
    for d in cr["dropped"]:
        print(f"dropped   \"{d['quote'][:60]}\": {d['reason']}")