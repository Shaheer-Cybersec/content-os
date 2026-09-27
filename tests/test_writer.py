"""[A07] checkpoint tests. Mock LLM via conftest."""
import copy

import pytest

from agents.strategist import agent as ST
from agents.writer import agent as WR
from agents.writer.agent import Writer, ensure_hashtags, mock_output, review_notes, strip_markdown
from core import llm

SHA = "672971d66a2ef9f85151e53283113f33d642dabd"
REPO = {"head_sha": SHA, "tree_truncated": False, "tree": [{"path": "a.py", "size": 1}],
        "commits": [{"sha": SHA}]}
EV = {"claim": "load_payload only base64-decodes", "source_path": "a.py", "source_ref": SHA[:7]}
STRATEGY = {"format": "myth-vs-fact", "hook": "Signed is not encrypted.",
            "beats": ["Myth vs fact.", "The decode.", "The rule."], "length_target": 1000,
            "cta": "What is in your cookies?", "hashtags": ["#AppSec", "#Python"],
            "evidence": [EV], "audience_note": "devs"}

@pytest.fixture(autouse=True)
def no_voice(tmp_path, monkeypatch):
    monkeypatch.setattr(ST, "VOICE_FILE", tmp_path / "voice_samples.md")

def ctx():
    return {"strategy": copy.deepcopy(STRATEGY), "repo": copy.deepcopy(REPO), "run_id": "t-a07",
            "visual_plan": {"shots": [{"shot": "decode in terminal"}]}}

def fake_llm(monkeypatch, out):
    monkeypatch.setattr(WR.llm, "call", lambda **kw: copy.deepcopy(out))


def test_mock_run_succeeds():
    c = ctx()
    assert Writer().execute(c)["status"] == "success"
    d = c["draft"]
    assert d["text"].startswith("Signed is not encrypted.")
    assert d["claims"][0]["source_path"] == "a.py" and d["chars"] == len(d["text"])

def test_invented_claims_fail(monkeypatch):
    out = mock_output(STRATEGY)
    out["claims"] = [dict(EV, source_path="fake.py")]
    fake_llm(monkeypatch, out)
    rec = Writer().execute(ctx())
    assert rec["status"] == "failed" and "no valid claims" in rec["error"]

def test_one_bad_claim_dropped_and_noted(monkeypatch):
    out = mock_output(STRATEGY)
    out["claims"] = [EV, dict(EV, source_path="fake.py")]
    fake_llm(monkeypatch, out)
    c = ctx()
    Writer().execute(c)
    assert len(c["draft"]["claims"]) == 1
    assert any("claim dropped" in n for n in c["draft"]["notes"])

def test_markdown_stripped_hashtags_kept():
    raw = "# Title\n\n**Bold** and __this__\n```python\nx = 1\n```\n\n#AppSec #Python"
    out = strip_markdown(raw)
    assert "**" not in out and "```" not in out and not out.startswith("# ")
    assert out.startswith("Title") and "Bold and this" in out and out.endswith("#AppSec #Python")

def test_missing_hashtags_appended():
    assert ensure_hashtags("Body text", ["#AppSec", "#Python"]).endswith("\n\n#AppSec #Python")
    assert ensure_hashtags("Body #appsec", ["#AppSec"]) == "Body #appsec"

def test_over_limit_fails(monkeypatch):
    out = mock_output(STRATEGY)
    out["text"] = "Signed is not encrypted.\n" + "x" * 3100
    fake_llm(monkeypatch, out)
    rec = Writer().execute(ctx())
    assert rec["status"] == "failed" and "3000" in rec["error"]

def test_notes_for_length_and_hook():
    notes = review_notes("A different first line\n" + "x" * 2000, STRATEGY)
    assert any("vs target 1000" in n for n in notes)
    assert any("hook" in n for n in notes)
    assert review_notes("Signed is not encrypted.\n" + "x" * 980, STRATEGY) == []

def test_claude_backend_waits(monkeypatch, tmp_path):
    monkeypatch.setattr(llm, "BACKEND", "claude")
    monkeypatch.setattr(llm, "RUNS_DIR", tmp_path)
    assert Writer().execute(ctx())["status"] == "waiting"
    prompt = (tmp_path / "t-a07" / "writer.prompt.md").read_text(encoding="utf-8")
    assert "POST PLAN" in prompt and "decode in terminal" in prompt