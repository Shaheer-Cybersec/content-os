
"""
[I07] orchestrator - runs the pipeline as a resumable state machine.

    A01 A02 A03 A04 | G1 | A05 A06 A07 A08 A09 A10 | G2 | A11

Every step's output lands in ctx, and ctx is saved to data/runs/<run_id>/state.json
after every step. So a run can stop at any point and pick up exactly where it was:
  - "waiting"  an LLM stage wrote its prompt for Claude; resume() re-runs only that
               stage, which now finds the response file
  - "gate"     G1 (you pick an angle) or G2 (you approve the package)
  - "failed"   a stage failed; fix the cause, then resume() retries that stage
  - "done"     A11 saved the post to memory
  - "rejected" you said no at G2; nothing is written to memory

Public API (the I09 CLI and the R01 runner skill call only these):
  start(source, run_id=None)  pick(run_id, n)  approve(run_id)  reject(run_id, reason)
  resume(run_id)  load(run_id)  summary(state)
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

from config import settings
from core.base_agent import AgentError

PLAN = ("A01", "A02", "A03", "A04", "G1", "A05", "A06", "A07", "A08", "A09", "A10", "G2", "A11")
FINAL = ("done", "rejected")


def _agents() -> dict:
    """Imported lazily so importing the orchestrator stays cheap."""
    from agents.analyzer.agent import Analyzer
    from agents.angle_extractor.agent import AngleExtractor
    from agents.critic.agent import Critic
    from agents.memory_retrieve.agent import MemoryRetrieve
    from agents.memory_writer.agent import MemoryWriter
    from agents.packager.agent import Packager
    from agents.repo_ingest.agent import RepoIngest
    from agents.reviser.agent import Reviser
    from agents.strategist.agent import Strategist
    from agents.visual_planner.agent import VisualPlanner
    from agents.writer.agent import Writer
    return {"A01": RepoIngest, "A02": MemoryRetrieve, "A03": Analyzer, "A04": AngleExtractor,
            "A05": Strategist, "A06": VisualPlanner, "A07": Writer, "A08": Critic,
            "A09": Reviser, "A10": Packager, "A11": MemoryWriter}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _path(run_id: str):
    return settings.RUNS_DIR / run_id / "state.json"


def load(run_id: str) -> dict:
    p = _path(run_id)
    if not p.exists():
        raise AgentError(f"no run '{run_id}' (expected {p.as_posix()})")
    return json.loads(p.read_text(encoding="utf-8"))


def _save(st: dict) -> dict:
    st["updated"] = _now()
    p = _path(st["run_id"])
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, indent=1), encoding="utf-8")
    tmp.replace(p)                       # atomic: a crash never leaves half a state file
    return st


def usable_angles(ctx: dict) -> list[dict]:
    return [a for a in ctx.get("angle_set", {}).get("angles", []) if a.get("dedup_status") != "rejected"]


def _advance(st: dict) -> dict:
    ctx, agents = st["ctx"], _agents()
    st["status"], st["gate"] = "running", None
    while st["pos"] < len(PLAN):
        step = PLAN[st["pos"]]
        if step == "G1" and "chosen_angle" not in ctx:
            st["status"], st["gate"] = "gate", "G1"
            return _save(st)
        if step == "G2" and "approved_post" not in ctx:
            st["status"], st["gate"] = "gate", "G2"
            return _save(st)
        if step.startswith("A"):
            rec = agents[step]().execute(ctx)
            rec["at"] = _now()
            st["records"][step] = rec
            if rec["status"] in ("waiting", "failed"):
                st["status"] = rec["status"]
                return _save(st)
        st["pos"] += 1
        _save(st)                        # checkpoint after every step
    st["status"] = "done"
    return _save(st)


def start(source: str, run_id: str | None = None) -> dict:
    run_id = run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
    if _path(run_id).exists():
        raise AgentError(f"run '{run_id}' already exists; use resume")
    st = {"run_id": run_id, "source": source, "status": "running", "gate": None, "pos": 0,
          "created": _now(), "updated": _now(), "note": None, "records": {},
          "ctx": {"source": source, "run_id": run_id}}
    return _advance(_save(st))


def resume(run_id: str) -> dict:
    st = load(run_id)
    if st["status"] in FINAL:
        return st
    if st["status"] == "gate":
        return st                        # a gate needs pick/approve/reject, not resume
    return _advance(st)


def pick(run_id: str, n: int) -> dict:
    st = load(run_id)
    if st["gate"] != "G1":
        raise AgentError(f"run '{run_id}' is not at G1 (status {st['status']}, gate {st['gate']})")
    angles = usable_angles(st["ctx"])
    if not 1 <= n <= len(angles):
        raise AgentError(f"pick 1-{len(angles)}, got {n}")
    st["ctx"]["chosen_angle"] = angles[n - 1]
    return _advance(st)


def approve(run_id: str) -> dict:
    st = load(run_id)
    if st["gate"] != "G2":
        raise AgentError(f"run '{run_id}' is not at G2 (status {st['status']}, gate {st['gate']})")
    ctx = st["ctx"]
    ctx["approved_post"] = ctx["package"]["post"]
    ctx["angles"] = ctx["angle_set"]["angles"]
    return _advance(st)


def reject(run_id: str, reason: str = "") -> dict:
    st = load(run_id)
    if st["status"] in FINAL:
        raise AgentError(f"run '{run_id}' is already {st['status']}")
    st["status"], st["gate"], st["note"] = "rejected", None, reason or "rejected at review"
    return _save(st)


def summary(st: dict) -> dict:
    """What a human (or the runner skill) needs to decide the next move."""
    ctx = st["ctx"]
    out = {"run_id": st["run_id"], "status": st["status"], "gate": st["gate"],
           "next": PLAN[st["pos"]] if st["pos"] < len(PLAN) else None,
           "stages": [{"id": k, "status": r["status"], "ms": r["duration_ms"], "error": r["error"]}
                      for k, r in st["records"].items()]}
    if st["status"] in ("waiting", "failed"):
        out["detail"] = st["records"][PLAN[st["pos"]]]["error"]
    if st["gate"] == "G1":
        out["angles"] = [{"n": i, "title": a["title"], "score": a.get("score"),
                          "post_type": a.get("post_type"), "hook": a.get("hook")}
                         for i, a in enumerate(usable_angles(ctx), 1)]
    if st["gate"] == "G2":
        pkg = ctx["package"]
        out["package"] = {"path": pkg["path"], "chars": pkg["chars"], "warnings": pkg["warnings"],
                          "critic_verdict": ctx.get("critique", {}).get("verdict"),
                          "unresolved": ctx.get("final_draft", {}).get("unresolved", []),
                          "text": pkg["post"]["text"]}
    if st["status"] == "done":
        out["memory_write"] = ctx.get("memory_write")
    if st["status"] == "rejected":
        out["note"] = st["note"]
    return out