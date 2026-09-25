"""[A04] checkpoint tests. Mock LLM (conftest) and fake embeddings; no model downloads."""
import copy

import pytest

from agents.angle_extractor import agent as AX
from agents.angle_extractor.agent import AngleExtractor, build_user_prompt, mock_output
from core import llm

SHA = "672971d66a2ef9f85151e53283113f33d642dabd"
REPO = {"owner": "o", "repo": "demo", "url": "https://github.com/o/demo", "head_sha": SHA,
        "tree_truncated": False, "tree": [{"path": "a.py", "size": 1}, {"path": "b.py", "size": 1}],
        "commits": [{"sha": SHA}]}

def ev(path="a.py"):
    return [{"claim": "c", "source_path": path, "source_ref": SHA[:7]}]

def finding(title, path="a.py"):
    return {"title": title, "detail": "Specific detail text.", "evidence": ev(path)}

ANALYSIS = {"summary": "A demo project for tests.", "purpose": "Testing.",
            "components": [finding("Signer"), finding("Timestamps", "b.py")],
            "decisions": [finding("SHA-1 default")], "security_notes": [finding("NoneAlgorithm")],
            "limitations": [], "dropped": []}
MEMORY = {"past_posts": [], "queued_angles": [], "embedded_post_ids": []}

# Fake embeddings: each angle gets a distinct direction unless its title says "DUP".
_seen = {}
def fake_embed(text):
    key = "DUP" if "DUP" in text else text
    if key not in _seen:
        v = [0.0] * 32
        v[len(_seen) % 32] = 1.0
        _seen[key] = v
    return _seen[key]

@pytest.fixture(autouse=True)
def fakes(monkeypatch):
    _seen.clear()
    monkeypatch.setattr(AX.E, "embed", fake_embed)
    monkeypatch.setattr(AX, "past_vectors", lambda memory: [])

def ctx():
    return {"repo": copy.deepcopy(REPO), "analysis": copy.deepcopy(ANALYSIS),
            "memory": copy.deepcopy(MEMORY), "run_id": "t-a04"}

def angle(title, score=0.8, path="a.py"):
    return {"title": title, "summary": "What this post teaches, clearly.", "hook": "A concrete first line.",
            "post_type": "lesson", "audience": "devs", "score": score, "evidence": ev(path)}

def fake_llm(monkeypatch, angles):
    monkeypatch.setattr(AX.llm, "call", lambda **kw: {"angles": copy.deepcopy(angles)})


def test_mock_run_all_ok_and_ranked():
    c = ctx()
    assert AngleExtractor().execute(c)["status"] == "success"
    s = c["angle_set"]
    assert s["counts"]["ok"] == 6
    scores = [a["score"] for a in s["angles"]]
    assert scores == sorted(scores, reverse=True)

def test_mock_output_is_schema_valid():
    AX.AnglesOut.model_validate(mock_output(ANALYSIS))

def test_prompt_lists_past_and_queued():
    m = {"past_posts": [{"angle_title": "Old idea", "text": "old post text"}],
         "queued_angles": [{"title": "Queued idea"}]}
    p = build_user_prompt(REPO, ANALYSIS, m)
    assert "Old idea" in p and "Queued idea" in p and "NoneAlgorithm" in p

def test_invented_evidence_drops_angle(monkeypatch):
    angles = [angle(f"Real angle number {i}") for i in range(5)] + [angle("Fake file angle", path="x.py")]
    fake_llm(monkeypatch, angles)
    c = ctx()
    AngleExtractor().execute(c)
    assert c["angle_set"]["counts"]["dropped"] == 1
    assert "Fake file angle" not in [a["title"] for a in c["angle_set"]["angles"]]

def test_near_duplicate_inside_batch_rejected(monkeypatch):
    angles = [angle(f"Distinct angle {i}", 0.5) for i in range(4)] + \
             [angle("DUP first version", 0.9), angle("DUP second version", 0.6)]
    fake_llm(monkeypatch, angles)
    c = ctx()
    AngleExtractor().execute(c)
    by = {a["title"]: a for a in c["angle_set"]["angles"]}
    assert by["DUP first version"]["dedup_status"] == "ok"            # higher score kept
    assert by["DUP second version"]["dedup_status"] == "rejected"
    assert c["angle_set"]["angles"][-1]["title"] == "DUP second version"  # rejected ranked last

def test_duplicate_of_past_post_rejected(monkeypatch):
    fake_llm(monkeypatch, [angle(f"Angle {i}") for i in range(5)])
    monkeypatch.setattr(AX, "past_vectors", lambda m: [fake_embed("Angle 0. What this post teaches, clearly.")])
    c = ctx()
    AngleExtractor().execute(c)
    rejected = [a for a in c["angle_set"]["angles"] if a["dedup_status"] == "rejected"]
    assert len(rejected) == 1 and "past post" in rejected[0]["reject_reason"]

def test_embeddings_off_marks_unchecked_not_fake(monkeypatch):
    monkeypatch.setattr(AX.E, "embed", lambda t: None)
    c = ctx()
    AngleExtractor().execute(c)
    assert all(a["dedup_status"] == "unchecked" and a["max_similarity"] is None
               for a in c["angle_set"]["angles"])

def test_too_few_usable_angles_fails(monkeypatch):
    angles = [angle("Good one", 0.9), angle("Good two", 0.8)] + \
             [angle(f"Bad {i}", path="nope.py") for i in range(3)]
    fake_llm(monkeypatch, angles)
    rec = AngleExtractor().execute(ctx())
    assert rec["status"] == "failed" and "usable angle" in rec["error"]

def test_claude_backend_waits(monkeypatch, tmp_path):
    monkeypatch.setattr(llm, "BACKEND", "claude")
    monkeypatch.setattr(llm, "RUNS_DIR", tmp_path)
    rec = AngleExtractor().execute(ctx())
    assert rec["status"] == "waiting"
    assert "PAST POSTS" in (tmp_path / "t-a04" / "angle_extractor.prompt.md").read_text(encoding="utf-8")