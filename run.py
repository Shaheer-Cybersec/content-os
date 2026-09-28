"""
[I09] Content OS command line.

  python run.py start <github-url | zip-url | zip-file> [--id RUN_ID]
  python run.py status  <run_id>
  python run.py resume  <run_id>          after Claude saved a *.response.json, or after a fix
  python run.py pick    <run_id> <n>      G1: choose angle n
  python run.py approve <run_id>          G2: approve the package, save to memory (A11)
  python run.py reject  <run_id> [reason] G2: discard the run, nothing saved
  python run.py list                      all runs, newest first

Add --json to any command for machine-readable output (used by the R01 runner skill).
"""
from __future__ import annotations

import argparse
import json
import sys

from config import settings
from core import orchestrator as O
from core.base_agent import AgentError

NEXT = {
    "waiting": "Claude answers the prompt above, then: python run.py resume {id}",
    "failed": "fix the cause, then: python run.py resume {id}",
    "G1": "python run.py pick {id} <n>",
    "G2": "open the file, then: python run.py approve {id}   (or: reject {id} \"reason\")",
    "done": "post saved to memory. Publish it on LinkedIn.",
    "rejected": "run closed, nothing saved.",
}


def show(s: dict) -> None:
    print(f"run      {s['run_id']}   status {s['status']}" + (f"   gate {s['gate']}" if s["gate"] else ""))
    for r in s["stages"]:
        err = f"  {r['error']}" if r["error"] and r["status"] != "waiting" else ""
        print(f"  {r['id']}  {r['status']:<8} {r['ms']:>6} ms{err}")
    if s["status"] in ("waiting", "failed"):
        print(f"\n{s['next']}: {s['detail']}")
    if s["gate"] == "G1":
        print("\nG1  pick an angle:")
        for a in s["angles"]:
            print(f"  {a['n']}. [{a['score']:.2f}] ({a['post_type']}) {a['title']}")
            print(f"       hook: {a['hook']}")
    if s["gate"] == "G2":
        p = s["package"]
        print(f"\nG2  review {p['path']}")
        print(f"    {p['chars']} chars, critic {p['critic_verdict']}, "
              f"unresolved {len(p['unresolved'])}, warnings {len(p['warnings'])}")
        for w in p["warnings"]:
            print(f"    warning: {w}")
        for u in p["unresolved"]:
            print(f"    unresolved [{u['severity']}]: \"{u['quote'][:60]}\" {u['issue']}")
    if s["status"] == "done":
        m = s["memory_write"]
        print(f"\nsaved    post {m['post_id']}, {m['angles_queued']} angle(s) queued, "
              f"note {m['obsidian_note']}")
    if s["status"] == "rejected":
        print(f"\nnote     {s['note']}")
    key = s["gate"] or s["status"]
    if key in NEXT:
        print(f"\nnext     {NEXT[key].format(id=s['run_id'])}")


def list_runs() -> list[dict]:
    rows = []
    for p in settings.RUNS_DIR.glob("*/state.json"):
        st = json.loads(p.read_text(encoding="utf-8"))
        rows.append({"run_id": st["run_id"], "status": st["status"], "gate": st["gate"],
                     "source": st["source"], "updated": st["updated"]})
    return sorted(rows, key=lambda r: r["updated"], reverse=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="run.py", description="Content OS")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("start"); p.add_argument("source"); p.add_argument("--id")
    for name in ("status", "resume", "approve"):
        sub.add_parser(name).add_argument("run_id")
    p = sub.add_parser("pick"); p.add_argument("run_id"); p.add_argument("n", type=int)
    p = sub.add_parser("reject"); p.add_argument("run_id"); p.add_argument("reason", nargs="?", default="")
    sub.add_parser("list")
    a = ap.parse_args(argv)

    try:
        if a.cmd == "list":
            rows = list_runs()
            if a.json:
                print(json.dumps(rows, indent=1))
            for r in rows if not a.json else []:
                print(f"{r['run_id']:<22} {r['status']:<9} {r['gate'] or '':<3} {r['source']}")
            return 0
        st = {"start": lambda: O.start(a.source, a.id),
              "status": lambda: O.load(a.run_id),
              "resume": lambda: O.resume(a.run_id),
              "pick": lambda: O.pick(a.run_id, a.n),
              "approve": lambda: O.approve(a.run_id),
              "reject": lambda: O.reject(a.run_id, a.reason)}[a.cmd]()
    except AgentError as e:
        print(json.dumps({"error": str(e)}) if a.json else f"error    {e}")
        return 1
    s = O.summary(st)
    print(json.dumps(s, indent=1)) if a.json else show(s)
    return 0 if st["status"] != "failed" else 1


if __name__ == "__main__":
    sys.exit(main())