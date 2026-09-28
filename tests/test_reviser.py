"""[A09] checkpoint tests. Mock LLM via conftest."""
import copy

from agents.reviser import agent as RV
from agents.reviser.agent import Reviser, mock_output, unresolved_flags
from core import llm

SHA = "672971d66a2ef9f85151e53283113f33d642dabd"
REPO = {"head_sha": SHA, "tree_truncated": False, "tree": [{"path": "a.py", "size": 1}],
        "commits": [{"sha": SHA}]}
EV = {"claim": "c", "source_path": "a.py", "source_ref": SHA[:7]}
TEXT = ("Signed is not encrypted.\n\nThe decode is two lines of Python.\n\n"
        "Anyone can read the payload without the key, and that surprises people who "
        "assume a signature means the contents are hidden from whoever holds the token.\n\n"
        "Decode one token today. What is in it?\n\n#AppSec #Python")
FLAGS = [{"severity": "fix", "quote": "The decode is two lines of Python.", "issue": "not code",
          "suggestion": "The decode takes two steps."},
         {"severity": "nit", "quote": "What is in it?", "issue": "weak", "suggestion": "s"}]

def ctx(verdict="revise", flags=None):
    return {"draft": {"text": TEXT, "claims": [dict(EV)]},
            "critique": {"verdict": verdict, "flags": copy.deepcopy(flags or FLAGS)},
            "strategy": {"hashtags": ["#AppSec", "#Python"]},
            "repo": copy.deepcopy(REPO), "run_id": "t-a09"}

def fake_llm(monkeypatch, out):
    monkeypatch.setattr(RV.llm, "call", lambda **kw: copy.deepcopy(out))


def test_pass_verdict_skips():
    rec = Reviser().execute(ctx(verdict="pass"))
    assert rec["status"] == "skipped" and "pass" in rec["error"]

def test_mock_revision_succeeds():
    c = ctx()
    assert Reviser().execute(c)["status"] == "success"
    f = c["final_draft"]
    assert "two lines of Python" not in f["text"] and f["unresolved"] == []
    assert "What is in it?" in f["text"]          # nit left alone

def test_block_still_present_fails(monkeypatch):
    flags = [dict(FLAGS[0], severity="block")]
    out = mock_output({"text": TEXT, "claims": [EV]}, [])        # model changed nothing
    out["changes"] = [{"quote": flags[0]["quote"], "action": "applied", "note": "done"}]
    fake_llm(monkeypatch, out)
    rec = Reviser().execute(ctx(flags=flags))
    assert rec["status"] == "failed" and "block flag" in rec["error"]

def test_declined_block_is_allowed(monkeypatch):
    flags = [dict(FLAGS[0], severity="block")]
    out = mock_output({"text": TEXT, "claims": [EV]}, [])
    out["changes"] = [{"quote": flags[0]["quote"], "action": "declined",
                       "note": "the flag is wrong: it is two lines"}]
    fake_llm(monkeypatch, out)
    assert Reviser().execute(ctx(flags=flags))["status"] == "success"

def test_unfixed_fix_goes_to_unresolved():
    blocking, unresolved = unresolved_flags(TEXT, FLAGS, [])
    assert blocking == [] and len(unresolved) == 1 and unresolved[0]["severity"] == "fix"

def test_whole_post_flags_ignored():
    fl = [{"severity": "fix", "quote": "(whole post)", "issue": "writer note", "suggestion": "s"}]
    assert unresolved_flags(TEXT, fl, []) == ([], [])

def test_invented_claim_fails(monkeypatch):
    out = mock_output({"text": TEXT, "claims": [EV]}, FLAGS)
    out["claims"] = [dict(EV, source_path="fake.py")]
    fake_llm(monkeypatch, out)
    rec = Reviser().execute(ctx())
    assert rec["status"] == "failed" and "no valid claims" in rec["error"]

def test_claude_backend_waits(monkeypatch, tmp_path):
    monkeypatch.setattr(llm, "BACKEND", "claude")
    monkeypatch.setattr(llm, "RUNS_DIR", tmp_path)
    assert Reviser().execute(ctx())["status"] == "waiting"
    prompt = (tmp_path / "t-a09" / "reviser.prompt.md").read_text(encoding="utf-8")
    assert "FLAGS" in prompt and "two lines of Python" in prompt