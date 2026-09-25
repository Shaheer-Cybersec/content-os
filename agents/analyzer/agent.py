"""
[A03] analyzer - LLM agent (heavy tier). First stage where Claude does the thinking.

Input : ctx["repo"]      A01 output
        ctx["run_id"]
Output: ctx["analysis"]  {summary, purpose, components[], decisions[],
                          security_notes[], limitations[], dropped[]}
        Each finding = {title, detail, evidence: [Evidence]}

After the model answers, EVERY evidence item is checked with I05 against A01's
real tree and commits:
  - invalid evidence items are removed (and listed in "dropped" with the reason)
  - a finding left with no valid evidence is removed entirely
  - fewer than 2 grounded findings left -> the stage fails (fail-closed)
So nothing ungrounded ever reaches the angle extractor or the writer.

The system prompt lives in agents/analyzer/prompt.md: tune it without touching code.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from pydantic import BaseModel, Field

from core import llm
from core.base_agent import AgentError, BaseAgent
from core.evidence import validate
from core.schemas import Evidence

PROMPT_FILE = Path(__file__).with_name("prompt.md")
MIN_FINDINGS = 2
SECTIONS = ("components", "decisions", "security_notes", "limitations")


class Finding(BaseModel):
    title: str = Field(min_length=3, max_length=120)
    detail: str = Field(min_length=10)
    evidence: list[Evidence] = Field(min_length=1)


class AnalysisOut(BaseModel):
    summary: str = Field(min_length=20, description="2-3 sentences: what this project is")
    purpose: str = Field(min_length=10, description="the problem it exists to solve")
    components: list[Finding] = Field(min_length=1, description="main parts and what each does")
    decisions: list[Finding] = Field(default_factory=list, description="notable technical choices")
    security_notes: list[Finding] = Field(default_factory=list,
                                          description="attack surface, weaknesses, defences")
    limitations: list[Finding] = Field(default_factory=list, description="what it does not do")


# ---------- prompt building ----------

def allowed_refs(repo: dict) -> list[str]:
    refs = [c["sha"][:12] for c in repo.get("commits", [])]
    return refs or [repo["head_sha"][:12]]


def build_user_prompt(repo: dict) -> str:
    parts = [
        f"REPOSITORY: {repo['owner']}/{repo['repo']}  ({repo['url']})",
        f"SOURCE: {repo.get('source_type', 'git')}   HEAD: {repo['head_sha'][:12]}",
        "",
        "ALLOWED COMMIT REFS (use one of these as source_ref):",
        *[f"- {r}" for r in allowed_refs(repo)],
        "",
        "RECENT COMMITS:",
        *[f"- {c['sha'][:7]} {c['date']} {c['subject']}" for c in repo.get("commits", [])[:15]],
        "",
        f"FILE TREE ({repo['file_count']} files{', truncated' if repo.get('tree_truncated') else ''}):",
        *[f"- {f['path']}" for f in repo.get("tree", [])],
        "",
        "README:",
        repo.get("readme") or "(none)",
        "",
        "DEPENDENCY FILES:",
    ]
    for path, text in (repo.get("dependencies") or {}).items():
        parts += [f"--- {path}", text]
    parts += ["", "KEY FILES (entry points first):"]
    for path, text in (repo.get("key_files") or {}).items():
        parts += [f"=== FILE: {path} ===", text]
    parts += ["", "Produce the analysis as JSON matching the schema."]
    return "\n".join(parts)


def mock_output(repo: dict) -> dict:
    """Realistic example output that cites files this repo really has."""
    paths = list((repo.get("key_files") or {}).keys()) or [f["path"] for f in repo["tree"]]
    ref = allowed_refs(repo)[0]
    ev = lambda i, claim: [{"claim": claim, "source_path": paths[i % len(paths)], "source_ref": ref}]
    return {
        "summary": f"{repo['repo']} is a small project analysed in mock mode. "
                   "This text is a placeholder produced without any model.",
        "purpose": "Mock purpose used to test the pipeline wiring.",
        "components": [{"title": "Entry point", "detail": "Mock: the main entry file.",
                        "evidence": ev(0, "This file exists in the repo")}],
        "decisions": [{"title": "Structure", "detail": "Mock: files are split by concern.",
                       "evidence": ev(1, "This file exists in the repo")}],
        "security_notes": [{"title": "Input handling", "detail": "Mock: inputs reach the core logic.",
                            "evidence": ev(0, "This file handles input")}],
        "limitations": [],
    }


# ---------- grounding ----------

def ground(analysis: dict, repo: dict) -> tuple[dict, int]:
    """Drop invalid evidence and unsupported findings. Returns (grounded, kept_count)."""
    dropped, kept = [], 0
    for section in SECTIONS:
        good_findings = []
        for f in analysis.get(section, []):
            check = validate(f["evidence"], repo)
            for bad in check["invalid"]:
                dropped.append({"section": section, "finding": f["title"], "reason": bad["reason"]})
            if check["valid"]:
                good_findings.append({**f, "evidence": check["valid"]})
                kept += 1
            else:
                dropped.append({"section": section, "finding": f["title"],
                                "reason": "no valid evidence left, finding removed"})
        analysis[section] = good_findings
    analysis["dropped"] = dropped
    return analysis, kept


# ---------- agent ----------

class Analyzer(BaseAgent):
    agent_id = "A03"
    name = "analyzer"
    requires = ("repo", "run_id")
    produces = "analysis"
    uses_llm = True

    def run(self, ctx: dict) -> dict:
        repo = ctx["repo"]
        raw = llm.call(stage=self.name, tier="heavy",
                       system=PROMPT_FILE.read_text(encoding="utf-8"),
                       user=build_user_prompt(repo),
                       schema=AnalysisOut, mock=mock_output(repo), run_id=ctx["run_id"])
        analysis, kept = ground(raw, repo)
        if kept < MIN_FINDINGS:
            reasons = "; ".join(d["reason"] for d in analysis["dropped"][:5])
            raise AgentError(f"only {kept} grounded finding(s) after evidence check "
                             f"(need {MIN_FINDINGS}). Dropped: {reasons}")
        return analysis


# ---------- manual checkpoint: A01 then A03 on a real source ----------
if __name__ == "__main__":
    from agents.repo_ingest.agent import RepoIngest
    source = sys.argv[1] if len(sys.argv) > 1 else "https://github.com/pallets/itsdangerous"
    run_id = sys.argv[2] if len(sys.argv) > 2 else "a03-check"
    ctx = {"source": source, "run_id": run_id}
    for agent in (RepoIngest(), Analyzer()):
        rec = agent.execute(ctx)
        print(f"{rec['id']} {rec['agent']:<12} {rec['status']:<8} {rec['error'] or ''}")
        if rec["status"] != "success":
            sys.exit(0)
    a = ctx["analysis"]
    print(f"\nbackend   {llm.backend_for('analyzer')}")
    print(f"summary   {a['summary']}")
    for s in SECTIONS:
        print(f"{s:<15} {len(a[s])} finding(s)")
        for f in a[s]:
            print(f"   - {f['title']}  [{', '.join(e['source_path'] for e in f['evidence'])}]")
    print(f"dropped   {len(a['dropped'])}")
    for d in a["dropped"]:
        print(f"   x {d['section']}/{d['finding']}: {d['reason']}")