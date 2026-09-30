"""
[A10] packager - static agent (no LLM). Last stage before your approval (G2).

Input : ctx["repo"]         A01 output (tree + commits, for the final evidence check)
        ctx["run_id"]
        ctx["final_draft"]  A09 output {"text", "claims": [Evidence]}
                            (falls back to ctx["draft"] when A09 was skipped)
        ctx["visual_plan"]  optional, A06 output {"shots": [{"shot", "how"}]} (or a plain list)
        ctx["chosen_angle"] optional {"title"}
Output: ctx["package"]      {"path", "folder", "pdf", "chars", "words", "hashtags", "links",
                             "warnings", "evidence_checked", "post", "kit"}
        and in outputs/<repo>/<date>_<run>/  (built by core/report.py):
              00_REPORT.pdf        ONE visual PDF with everything
              00_START_HERE.md     file index + the 5 posting steps
              01_caption.md  02_first_comment.md  03_graphic.png  04_alt_text.md
              05_visuals.md  06_evidence.md  07_run_summary.md
              copy-paste.html      one page with a Copy button per block

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

from config.settings import OUTPUTS_DIR
from core.base_agent import AgentError, BaseAgent
from core.evidence import validate
from core.report import HOW_TO_POST, build_folder, linkedin_kit, run_folder  # noqa: F401

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
        post = {"text": text, "evidence": claims}
        folder = run_folder(OUTPUTS_DIR, repo, run_id)
        built = build_folder(dict(ctx, package={"post": post, "warnings": warnings}), folder,
                             records=ctx.get("_records"))
        warnings += built["skipped"]

        return {
            "path": (folder / "00_START_HERE.md").as_posix(),
            "folder": folder.as_posix(),
            "pdf": (folder / "00_REPORT.pdf").as_posix(),
            **s,
            "warnings": warnings,
            "evidence_checked": len(claims),
            "post": post,                                  # G2 hands this to A11 on approval
            "kit": {**built["kit"], "dir": folder.as_posix()},
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
