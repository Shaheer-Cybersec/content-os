"""[A03] checkpoint tests. Mock backend by default (tests/conftest.py)."""
import copy

import pytest

from agents.analyzer import agent as AZ
from agents.analyzer.agent import Analyzer, build_user_prompt, mock_output
from core import llm

SHA = "672971d66a2ef9f85151e53283113f33d642dabd"
REPO = {
    "owner": "o", "repo": "demo", "url": "https://github.com/o/demo", "source_type": "git",
    "head_sha": SHA, "file_count": 3, "tree_truncated": False,
    "tree": [{"path": "app/main.py", "size": 10}, {"path": "app/retriever.py", "size": 10},
             {"path": "README.md", "size": 5}],
    "commits": [{"sha": SHA, "date": "2026-09-01", "subject": "add retriever"}],
    "readme": "# demo", "dependencies": {"requirements.txt": "fastapi"},
    "key_files": {"app/main.py": "TOP_K = 8", "app/retriever.py": "def retrieve(): ..."},
}

def ctx():
    return {"repo": copy.deepcopy(REPO), "run_id": "t-a03"}

def finding(path, ref=SHA[:7], title="A finding"):
    return {"title": title, "detail": "Some specific detail here.",
            "evidence": [{"claim": "c", "source_path": path, "source_ref": ref}]}

def fake_llm(monkeypatch, output):
    monkeypatch.setattr(AZ.llm, "call", lambda **kw: copy.deepcopy(output))


def test_mock_run_succeeds_and_is_grounded():
    c = ctx()
    rec = Analyzer().execute(c)
    assert rec["status"] == "success"
    assert c["analysis"]["dropped"] == []
    assert c["analysis"]["components"][0]["evidence"][0]["source_path"] in {"app/main.py", "app/retriever.py"}

def test_mock_output_matches_schema():
    AZ.AnalysisOut.model_validate(mock_output(REPO))

def test_prompt_contains_tree_commits_and_code():
    p = build_user_prompt(REPO)
    assert "- app/retriever.py" in p and "672971d66a2e" in p and "TOP_K = 8" in p

def test_invented_evidence_is_dropped(monkeypatch):
    out = mock_output(REPO)
    out["components"].append(finding("app/guardrails.py", title="Guardrails"))
    fake_llm(monkeypatch, out)
    c = ctx()
    assert Analyzer().execute(c)["status"] == "success"
    titles = [f["title"] for f in c["analysis"]["components"]]
    assert "Guardrails" not in titles
    assert any("path not in repo" in d["reason"] for d in c["analysis"]["dropped"])

def test_partially_bad_finding_keeps_good_evidence(monkeypatch):
    out = mock_output(REPO)
    out["decisions"][0]["evidence"].append({"claim": "x", "source_path": "fake.py", "source_ref": SHA[:7]})
    fake_llm(monkeypatch, out)
    c = ctx()
    Analyzer().execute(c)
    assert len(c["analysis"]["decisions"]) == 1
    assert len(c["analysis"]["decisions"][0]["evidence"]) == 1

def test_too_few_grounded_findings_fails(monkeypatch):
    out = mock_output(REPO)
    for s in ("components", "decisions", "security_notes"):
        out[s] = [finding("nope.py")]
    fake_llm(monkeypatch, out)
    rec = Analyzer().execute(ctx())
    assert rec["status"] == "failed" and "grounded finding" in rec["error"]

def test_claude_backend_waits_with_prompt_file(monkeypatch, tmp_path):
    monkeypatch.setattr(llm, "BACKEND", "claude")
    monkeypatch.setattr(llm, "RUNS_DIR", tmp_path)
    rec = Analyzer().execute(ctx())
    assert rec["status"] == "waiting"
    prompt = (tmp_path / "t-a03" / "analyzer.prompt.md").read_text(encoding="utf-8")
    assert "senior security engineer" in prompt and "=== FILE: app/main.py ===" in prompt

def test_missing_repo_fails_cleanly():
    rec = Analyzer().execute({"run_id": "x"})
    assert rec["status"] == "failed" and "repo" in rec["error"]