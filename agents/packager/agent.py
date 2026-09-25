"""
[A10] packager - static agent (no LLM). Last stage before your approval (G2).

Input : ctx["repo"]         A01 output (tree + commits, for the final evidence check)
        ctx["run_id"]
        ctx["final_draft"]  A09 output {"text", "claims": [Evidence]}
                            (falls back to ctx["draft"] when A09 was skipped)
        ctx["visual_plan"]  optional, A06 output [{"shot", "how"}]
        ctx["chosen_angle"] optional {"title"}
Output: ctx["package"]      {"path", "chars", "words", "hashtags", "links",
                             "warnings", "evidence_checked", "post"}
        and one markdown file in outputs/ that you read before approving.

Hard stops (the run fails, no file is written):
  - over LinkedIn's 3,000 character limit
  - ANY evidence item that fails I05 (fail-closed: a post with an invented
    file or commit is never packaged, even if the critic missed it)
Soft warnings (packaged, but listed at the top of the file):
  - links in the body (LinkedIn reach drops; put links in the first comment)
  - more than 5 hashtags
"""
from __future__ import annotations

import json
import re
import sys
from datetime import datetime

from config.settings import OUTPUTS_DIR
from core.base_agent import AgentError, BaseAgent
from core.evidence import validate

LINKEDIN_MAX_CHARS = 3000
MAX_HASHTAGS = 5
URL_RE = re.compile(r"https?://\S+")
HASHTAG_RE = re.compile(r"(?<!\w)#\w+")


def stats(text: str) -> dict:
    return {
        "chars": len(text),
        "words": len(text.split()),
        "hashtags": len(HASHTAG_RE.findall(text)),
        "links": len(URL_RE.findall(text)),
    }


def render(repo: dict, run_id: str, angle: dict | None, text: str, s: dict,
           warnings: list[str], claims: list[dict], shots: list[dict]) -> str:
    title = (angle or {}).get("title") or "Untitled angle"
    out = [
        f"# {title}",
        "",
        f"- Repo: {repo['url']} @ `{repo['head_sha'][:12]}`",
        f"- Run: `{run_id}`",
        f"- Length: {s['chars']} / {LINKEDIN_MAX_CHARS} chars, {s['words']} words, "
        f"{s['hashtags']} hashtags, {s['links']} links",
        "",
    ]
    if warnings:
        out += ["## Warnings", ""] + [f"- {w}" for w in warnings] + [""]
    out += ["## Post (copy from here)", "", "```text", text, "```", ""]
    out += ["## Screenshots to take", ""]
    out += [f"- [ ] {v.get('shot', '?')}: {v.get('how', '')}".rstrip(": ") for v in shots] \
        or ["- (none planned)"]
    out += ["", "## Evidence (all verified against the repo)", "",
            "| Claim | File | Commit |", "|---|---|---|"]
    out += [f"| {c['claim']} | `{c['source_path']}` | `{c['source_ref']}` |" for c in claims] \
        or ["| (no factual claims) | | |"]
    return "\n".join(out) + "\n"


class Packager(BaseAgent):
    agent_id = "A10"
    name = "packager"
    requires = ("repo", "run_id")
    produces = "package"
    uses_llm = False

    def run(self, ctx: dict) -> dict:
        draft = ctx.get("final_draft") or ctx.get("draft")
        if not draft:
            raise AgentError("no final_draft or draft to package")
        text = (draft.get("text") or "").strip()
        if not text:
            raise AgentError("draft has no text")

        s = stats(text)
        if s["chars"] > LINKEDIN_MAX_CHARS:
            raise AgentError(f"post is {s['chars']} chars; LinkedIn allows {LINKEDIN_MAX_CHARS}")

        claims = draft.get("claims") or []
        check = validate(claims, ctx["repo"])
        if not check["ok"]:
            reasons = "; ".join(i["reason"] for i in check["invalid"])
            raise AgentError(f"{len(check['invalid'])} evidence item(s) failed: {reasons}")

        warnings = []
        if s["links"]:
            warnings.append(f"{s['links']} link(s) in the body. Move them to the first comment.")
        if s["hashtags"] > MAX_HASHTAGS:
            warnings.append(f"{s['hashtags']} hashtags. Keep it to {MAX_HASHTAGS} or fewer.")

        repo, run_id = ctx["repo"], ctx["run_id"]
        name = f"{datetime.now():%Y-%m-%d}-{repo['repo']}-{run_id}.md"
        path = OUTPUTS_DIR / name
        path.write_text(render(repo, run_id, ctx.get("chosen_angle"), text, s, warnings,
                               claims, ctx.get("visual_plan") or []), encoding="utf-8")

        return {
            "path": path.as_posix(),
            **s,
            "warnings": warnings,
            "evidence_checked": len(claims),
            "post": {"text": text, "evidence": claims},   # G2 hands this to A11 on approval
        }


# ---------- fixture ----------

FIXTURE = {
    "run_id": "fixture-run-001",
    "repo": {
        "owner": "Shaheer-Cybersec", "repo": "damn-vulnerable-rag",
        "url": "https://github.com/Shaheer-Cybersec/damn-vulnerable-rag",
        "head_sha": "38ae3eb0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6",
        "tree_truncated": False,
        "tree": [{"path": "app/retriever.py", "size": 900}, {"path": "payloads/indirect.md", "size": 400}],
        "commits": [{"sha": "38ae3eb0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6", "subject": "tune top_k"}],
    },
    "chosen_angle": {"title": "Why top-k size changes indirect injection risk"},
    "draft": {
        "text": ("I made my RAG lab retrieve 8 chunks instead of 3.\n\n"
                 "Indirect prompt injection got easier. More chunks means more chances "
                 "for a poisoned document to reach the model.\n\n"
                 "#AISecurity #RAG #PromptInjection"),
        "claims": [
            {"claim": "The retriever's top_k is configurable", "source_path": "app/retriever.py",
             "source_ref": "38ae3eb"},
            {"claim": "Indirect payloads live in a separate folder", "source_path": "payloads/indirect.md",
             "source_ref": "38ae3eb"},
        ],
    },
    "visual_plan": [
        {"shot": "retriever.py top_k line", "how": "VS Code, zoom 150%, crop to 10 lines"},
        {"shot": "Injection success at k=3 vs k=8", "how": "terminal output side by side"},
    ],
}

if __name__ == "__main__":
    if "--fixture" not in sys.argv:
        sys.exit("usage: python -m agents.packager.agent --fixture")
    ctx = json.loads(json.dumps(FIXTURE))
    rec = Packager().execute(ctx)
    print(json.dumps(rec, indent=2))
    if rec["status"] == "success":
        p = ctx["package"]
        print(f"file       {p['path']}")
        print(f"length     {p['chars']} chars, {p['words']} words, {p['hashtags']} hashtags, {p['links']} links")
        print(f"evidence   {p['evidence_checked']} checked, all valid")
        print(f"warnings   {p['warnings'] or 'none'}")