"""[A05] checkpoint tests. Mock LLM via conftest."""
import copy

import pytest

from agents.strategist import agent as ST
from agents.strategist.agent import Strategist, build_user_prompt, clean_hashtags, mock_output
from core import llm

SHA = "672971d66a2ef9f85151e53283113f33d642dabd"
REPO = {"head_sha": SHA, "tree_truncated": False, "tree": [{"path": "a.py", "size": 1}],
        "commits": [{"sha": SHA}]}
ANGLE = {"title": "Signed is not encrypted", "summary": "Tokens are readable.", "hook": "Decode any cookie.",
         "post_type": "lesson", "audience": "devs",
         "evidence": [{"claim": "c", "source_path": "a.py", "source_ref": SHA[:7]}]}

@pytest.fixture(autouse=True)
def no_voice(tmp_path, monkeypatch):
    monkeypatch.setattr(ST, "VOICE_FILE", tmp_path / "voice_samples.md")
    return tmp_path / "voice_samples.md"

def ctx():
    return {"chosen_angle": copy.deepcopy(ANGLE), "analysis": {"summary": "s", "purpose": "p"},
            "repo": copy.deepcopy(REPO), "run_id": "t-a05"}

def fake_llm(monkeypatch, out):
    monkeypatch.setattr(ST.llm, "call", lambda **kw: copy.deepcopy(out))


def test_mock_run_succeeds():
    c = ctx()
    assert Strategist().execute(c)["status"] == "success"
    assert c["strategy"]["hook"] == "Decode any cookie."
    assert c["strategy"]["voice_used"] is False

def test_voice_samples_used_when_present(no_voice):
    no_voice.write_text("I broke my own RAG app last week.", encoding="utf-8")
    c = ctx()
    Strategist().execute(c)
    assert c["strategy"]["voice_used"] is True
    assert "I broke my own RAG app" in build_user_prompt(ANGLE, {}, ST.voice_samples())

def test_hashtags_cleaned():
    assert clean_hashtags(["AppSec", "#Tech", "#python", "#Python", "#Cyber Security!", "#A", "#B"]) == \
        ["#AppSec", "#python", "#CyberSecurity"]

def test_length_clamped(monkeypatch):
    out = mock_output(ANGLE)
    out["length_target"] = 2900
    fake_llm(monkeypatch, out)
    c = ctx()
    Strategist().execute(c)
    assert c["strategy"]["length_target"] == 1800

def test_invented_evidence_fails(monkeypatch):
    out = mock_output(ANGLE)
    out["evidence"] = [{"claim": "c", "source_path": "fake.py", "source_ref": SHA[:7]}]
    fake_llm(monkeypatch, out)
    rec = Strategist().execute(ctx())
    assert rec["status"] == "failed" and "no valid evidence" in rec["error"]

def test_needs_a_chosen_angle():
    c = ctx()
    del c["chosen_angle"]
    rec = Strategist().execute(c)
    assert rec["status"] == "failed" and "chosen_angle" in rec["error"]

def test_claude_backend_waits(monkeypatch, tmp_path):
    monkeypatch.setattr(llm, "BACKEND", "claude")
    monkeypatch.setattr(llm, "RUNS_DIR", tmp_path)
    assert Strategist().execute(ctx())["status"] == "waiting"
    assert "CHOSEN ANGLE" in (tmp_path / "t-a05" / "strategist.prompt.md").read_text(encoding="utf-8")