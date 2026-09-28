"""[I07] orchestrator tests: full mock pipeline, offline, every gate and resume path."""
import json
import zipfile

import pytest

from agents.memory_writer import agent as MW
from agents.packager import agent as PK
from agents.repo_ingest import agent as RI
from agents.strategist import agent as ST
from config import settings
from core import llm
from core import orchestrator as O
from core.base_agent import AgentError
from memory import db
from memory import embeddings as E

SHA = "672971d66a2ef9f85151e53283113f33d642dabd"
FILES = {"README.md": "# demo\nA tiny signing lib.", "requirements.txt": "itsdangerous\n",
         "src/demo/signer.py": "def sign(data, key):\n    return data + '.' + key\n",
         "src/demo/url_safe.py": "def load_payload(p):\n    return p\n",
         "tests/test_signer.py": "def test_sign(): ...\n"}


@pytest.fixture
def sb(tmp_path, monkeypatch):
    for mod, attr, sub in [(RI, "CACHE_DIR", "cache"), (MW, "OBSIDIAN_DIR", "obsidian"),
                           (PK, "OUTPUTS_DIR", "outputs"), (settings, "RUNS_DIR", "runs"),
                           (llm, "RUNS_DIR", "runs")]:
        (tmp_path / sub).mkdir(exist_ok=True)
        monkeypatch.setattr(mod, attr, tmp_path / sub)
    monkeypatch.setattr(MW, "DATA_DIR", tmp_path)
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "o.db")
    monkeypatch.setattr(E, "embed", lambda text: None)          # embeddings off: no dedup, no model
    monkeypatch.setattr(ST, "VOICE_FILE", tmp_path / "none.md")
    zp = tmp_path / "demo-main.zip"
    with zipfile.ZipFile(zp, "w") as z:
        for rel, text in FILES.items():
            z.writestr("demo-main/" + rel, text)
        z.comment = SHA.encode()
    return {"zip": str(zp), "root": tmp_path}


def test_runs_to_g1_then_g2_then_done(sb):
    st = O.start(sb["zip"], "r1")
    assert (st["status"], st["gate"]) == ("gate", "G1")
    assert list(st["records"]) == ["A01", "A02", "A03", "A04"]
    assert O.summary(st)["angles"][0]["n"] == 1

    st = O.pick("r1", 1)
    assert (st["status"], st["gate"]) == ("gate", "G2")
    s = O.summary(st)
    assert s["package"]["path"].endswith(".md") and s["package"]["critic_verdict"] in ("pass", "revise")

    st = O.approve("r1")
    assert st["status"] == "done" and O.summary(st)["memory_write"]["post_id"] == 1
    assert list(st["records"])[-1] == "A11"


def test_state_file_is_valid_json_after_each_stop(sb):
    O.start(sb["zip"], "r2")
    state = json.loads((sb["root"] / "runs" / "r2" / "state.json").read_text(encoding="utf-8"))
    assert state["pos"] == O.PLAN.index("G1") and state["ctx"]["repo"]["file_count"] == 5

def test_reject_at_g2_writes_nothing(sb):
    O.start(sb["zip"], "r3")
    O.pick("r3", 1)
    st = O.reject("r3", "hook too weak")
    assert st["status"] == "rejected" and O.summary(st)["note"] == "hook too weak"
    assert "A11" not in st["records"]
    assert not (sb["root"] / "o.db").exists() or db.recent_posts(db.connect()) == []
    assert O.resume("r3")["status"] == "rejected"

def test_wrong_gate_calls_refused(sb):
    O.start(sb["zip"], "r4")
    with pytest.raises(AgentError, match="not at G2"):
        O.approve("r4")
    with pytest.raises(AgentError, match="pick 1-"):
        O.pick("r4", 99)

def test_duplicate_run_id_refused(sb):
    O.start(sb["zip"], "r5")
    with pytest.raises(AgentError, match="already exists"):
        O.start(sb["zip"], "r5")

def test_claude_waiting_then_resume(sb, monkeypatch):
    monkeypatch.setattr(llm, "BACKEND", "claude")
    st = O.start(sb["zip"], "r6")
    assert st["status"] == "waiting" and O.summary(st)["next"] == "A03"
    assert "analyzer.prompt.md" in O.summary(st)["detail"]
    assert O.resume("r6")["status"] == "waiting"                 # still no answer: still waiting

    monkeypatch.setattr(llm, "BACKEND", "mock")                   # stand-in for "Claude answered"
    st = O.resume("r6")
    assert (st["status"], st["gate"]) == ("gate", "G1")
    assert st["records"]["A01"]["status"] == "success"            # A01 not re-run: one record

def test_failed_stage_resumes_from_that_stage(sb, monkeypatch):
    from agents.critic import agent as CR
    real = CR.Critic.run
    monkeypatch.setattr(CR.Critic, "run", lambda self, ctx: (_ for _ in ()).throw(AgentError("boom")))
    O.start(sb["zip"], "r7")
    st = O.pick("r7", 1)
    assert st["status"] == "failed" and O.summary(st)["detail"] == "boom"
    monkeypatch.setattr(CR.Critic, "run", real)
    assert O.resume("r7")["gate"] == "G2"

def test_unknown_run(sb):
    with pytest.raises(AgentError, match="no run"):
        O.load("nope")