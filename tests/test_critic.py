"""[A08] checkpoint tests. Mock LLM via conftest."""
import copy

from agents.critic import agent as CR
from agents.critic.agent import Critic, check_quotes, static_flags
from core import llm

TEXT = ("Signed is not encrypted.\n\nAnyone can decode the payload with base64.\n\n"
        "Decode one token today. What is in it?\n\n#AppSec #Python")
DRAFT = {"text": TEXT, "notes": [],
         "claims": [{"claim": "c", "source_path": "a.py", "source_ref": "672971d"}]}
STRATEGY = {"format": "myth-vs-fact", "hook": "Signed is not encrypted.", "beats": ["a", "b", "c"],
            "cta": "Decode one token today.", "audience_note": "devs"}

def ctx(draft=None):
    return {"draft": copy.deepcopy(draft or DRAFT), "strategy": copy.deepcopy(STRATEGY),
            "run_id": "t-a08"}

def fake_llm(monkeypatch, flags):
    out = {"flags": flags, "scores": {"hook": 4, "clarity": 4, "accuracy": 5, "voice": 4}}
    monkeypatch.setattr(CR.llm, "call", lambda **kw: copy.deepcopy(out))


def test_mock_run_nit_only_passes():
    c = ctx()
    assert Critic().execute(c)["status"] == "success"
    assert c["critique"]["verdict"] == "pass" and c["critique"]["flags"][0]["severity"] == "nit"

def test_clean_text_has_no_static_flags():
    assert static_flags(TEXT, []) == []

def test_static_catches_tells_emoji_markdown_links():
    bad = "Let's dive in \U0001F680\n\n**Bold** see https://x.io\n\n" + " ".join(f"#T{i}" for i in range(5))
    issues = " | ".join(f["issue"] for f in static_flags(bad, []))
    for word in ("let's dive", "emoji", "markdown", "link", "5 hashtags"):
        assert word in issues

def test_long_hook_and_writer_notes_flagged():
    fl = static_flags("x" * 250 + "\n\nbody", ["length 500 chars vs target 1300"])
    assert any("fold" in f["issue"] for f in fl)
    assert any("writer note" in f["issue"] and f["severity"] == "fix" for f in fl)

def test_invented_quote_dropped():
    kept, dropped = check_quotes([{"severity": "block", "quote": "uses AES-256", "issue": "x",
                                   "suggestion": "y"}], TEXT)
    assert kept == [] and dropped[0]["reason"] == "quote not found in post"

def test_quote_match_ignores_whitespace_and_smart_quotes():
    kept, _ = check_quotes([{"severity": "fix", "quote": "Anyone  can decode\nthe payload",
                             "issue": "x", "suggestion": "y"}], TEXT)
    assert len(kept) == 1

def test_block_flag_means_revise_and_sorted_first(monkeypatch):
    fake_llm(monkeypatch, [{"severity": "nit", "quote": "What is in it?", "issue": "weak cta",
                            "suggestion": "s"},
                           {"severity": "block", "quote": "Anyone can decode the payload",
                            "issue": "unsupported", "suggestion": "s"}])
    c = ctx()
    Critic().execute(c)
    assert c["critique"]["verdict"] == "revise"
    assert c["critique"]["flags"][0]["severity"] == "block"

def test_all_quotes_invented_means_pass(monkeypatch):
    fake_llm(monkeypatch, [{"severity": "block", "quote": "not in the post", "issue": "x",
                            "suggestion": "y"}])
    c = ctx()
    Critic().execute(c)
    assert c["critique"]["verdict"] == "pass" and len(c["critique"]["dropped"]) == 1

def test_claude_backend_waits(monkeypatch, tmp_path):
    monkeypatch.setattr(llm, "BACKEND", "claude")
    monkeypatch.setattr(llm, "RUNS_DIR", tmp_path)
    assert Critic().execute(ctx())["status"] == "waiting"
    prompt = (tmp_path / "t-a08" / "critic.prompt.md").read_text(encoding="utf-8")
    assert "STATIC FLAGS" in prompt and "CLAIMS" in prompt


def test_absolute_wording_note_is_only_a_nit_but_result_claims_still_fix():
    from agents.critic.agent import static_flags
    text = "The check never looks at context. " + "x" * 200
    fl = static_flags(text, ['absolute wording, check the evidence supports it: "never"',
                             'claims a result or run the evidence does not show: "I tested"'])
    sev = {f["quote"]: f["severity"] for f in fl if f["issue"].startswith("writer note")}
    assert sev["never"] == "nit" and sev["(whole post)"] == "fix"
