"""[I02] checkpoint tests for the agent contract and the Evidence schema."""
import pytest
from pydantic import ValidationError

from core.base_agent import AgentError, BaseAgent, SkipStage
from core.schemas import Evidence


# ---------- tiny fake agents used only in these tests ----------

class Echo(BaseAgent):
    agent_id, name, requires, produces = "T01", "echo", ("text",), "echoed"
    def run(self, ctx):
        return {"text": ctx["text"].upper()}

class ReturnsBytes(Echo):
    def run(self, ctx):
        return {"blob": b"\x00\x01"}

class Refuses(Echo):
    def run(self, ctx):
        raise AgentError("repo is empty")

class Skips(Echo):
    def run(self, ctx):
        raise SkipStage("no critic flags")

class Buggy(Echo):
    def run(self, ctx):
        return 1 / 0


# ---------- agent contract ----------

def test_success_writes_output_and_record():
    ctx = {"text": "hi"}
    rec = Echo().execute(ctx)
    assert rec["status"] == "success" and rec["id"] == "T01"
    assert ctx["echoed"] == {"text": "HI"}

def test_missing_input_fails_cleanly():
    ctx = {}
    rec = Echo().execute(ctx)
    assert rec["status"] == "failed"
    assert "text" in rec["error"]
    assert "echoed" not in ctx          # nothing half-written

def test_bytes_output_rejected():
    ctx = {"text": "hi"}
    rec = ReturnsBytes().execute(ctx)
    assert rec["status"] == "failed"
    assert "JSON" in rec["error"]
    assert "echoed" not in ctx

def test_agent_error_message_kept():
    rec = Refuses().execute({"text": "hi"})
    assert rec["status"] == "failed" and rec["error"] == "repo is empty"

def test_skip_is_not_a_failure():
    rec = Skips().execute({"text": "hi"})
    assert rec["status"] == "skipped"

def test_unexpected_bug_is_caught_with_type():
    rec = Buggy().execute({"text": "hi"})
    assert rec["status"] == "failed"
    assert rec["error"].startswith("ZeroDivisionError")


# ---------- Evidence schema ----------

def test_evidence_valid():
    e = Evidence(claim="Top-k is 5", source_path="app/retriever.py", source_ref="38ae3eb")
    assert e.source_path == "app/retriever.py"

def test_evidence_windows_path_normalised():
    e = Evidence(claim="x", source_path=r"app\retriever.py", source_ref="38ae3eb")
    assert e.source_path == "app/retriever.py"

@pytest.mark.parametrize("bad", ["", "   ", "/etc/passwd", "../secrets.txt", "C:/x.py"])
def test_evidence_bad_path_rejected(bad):
    with pytest.raises(ValidationError):
        Evidence(claim="x", source_path=bad, source_ref="38ae3eb")

def test_evidence_extra_field_rejected():
    with pytest.raises(ValidationError):
        Evidence(claim="x", source_path="a.py", source_ref="1", confidence=0.9)