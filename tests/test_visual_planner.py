"""[A06] checkpoint tests. Mock LLM via conftest."""
import copy

from agents.visual_planner import agent as VP
from agents.visual_planner.agent import VisualPlanner, check_shots, mock_output
from core import llm

SHA = "672971d66a2ef9f85151e53283113f33d642dabd"
REPO = {"owner": "o", "repo": "demo", "tree": [{"path": "src/a.py", "size": 1}]}
STRATEGY = {"format": "how-to", "hook": "h", "beats": ["a", "b", "c"],
            "evidence": [{"claim": "c", "source_path": "src/a.py", "source_ref": SHA[:7]}]}

def ctx():
    return {"strategy": copy.deepcopy(STRATEGY), "repo": copy.deepcopy(REPO), "run_id": "t-a06"}

def shot(kind="code", fp="src/a.py", minutes=3, name="A shot"):
    return {"shot": name, "kind": kind, "how": "Open it and crop the lines.",
            "file_path": fp, "lines": "1-10", "minutes": minutes}


def test_mock_run_succeeds():
    c = ctx()
    assert VisualPlanner().execute(c)["status"] == "success"
    assert c["visual_plan"]["shots"][0]["file_path"] == "src/a.py"

def test_invented_file_dropped():
    kept, dropped = check_shots([shot(fp="src/fake.py")], REPO)
    assert kept == [] and "not in repo" in dropped[0]["reason"]

def test_code_shot_needs_file():
    kept, dropped = check_shots([shot(fp=None)], REPO)
    assert kept == [] and "without file_path" in dropped[0]["reason"]

def test_terminal_shot_without_file_is_fine():
    kept, _ = check_shots([shot(kind="terminal", fp=None)], REPO)
    assert len(kept) == 1

def test_max_three_and_minutes_clamped():
    kept, dropped = check_shots([shot(minutes=12, name=f"s{i}") for i in range(4)], REPO)
    assert len(kept) == 3 and len(dropped) == 1
    assert all(s["minutes"] == 5 for s in kept)

def test_windows_path_normalised():
    kept, _ = check_shots([shot(fp="src\\a.py")], REPO)
    assert kept[0]["file_path"] == "src/a.py"

def test_claude_backend_waits(monkeypatch, tmp_path):
    monkeypatch.setattr(llm, "BACKEND", "claude")
    monkeypatch.setattr(llm, "RUNS_DIR", tmp_path)
    assert VisualPlanner().execute(ctx())["status"] == "waiting"
    assert "FILE TREE" in (tmp_path / "t-a06" / "visual_planner.prompt.md").read_text(encoding="utf-8")