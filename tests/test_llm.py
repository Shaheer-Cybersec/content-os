"""[I06] checkpoint tests. Ollama is faked with a tiny local HTTP server."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from pydantic import BaseModel

from core import llm
from core.base_agent import HandoffPending


class Out(BaseModel):
    summary: str
    keywords: list[str]

GOOD = {"summary": "A RAG lab.", "keywords": ["rag"]}


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "RUNS_DIR", tmp_path)
    cfg = {"tiers": {"heavy": {"ollama_model": "m"}, "mid": {"ollama_model": "m"}},
           "stages": {}, "ollama": {"retries": 2, "timeout_s": 5}}
    monkeypatch.setattr(llm, "load_config", lambda: cfg)
    return tmp_path

def use(monkeypatch, backend):
    monkeypatch.setattr(llm, "BACKEND", backend)

def call(**kw):
    args = dict(stage="demo", tier="mid", system="s", user="u", schema=Out, run_id="r1")
    args.update(kw)
    return llm.call(**args)


# ---------- routing ----------

def test_mock_env_overrides_stage_routing(monkeypatch):
    use(monkeypatch, "mock")
    assert llm.backend_for("demo", {"stages": {"demo": "ollama"}}) == "mock"

def test_stage_override_used_when_not_mock(monkeypatch):
    use(monkeypatch, "ollama")
    assert llm.backend_for("demo", {"stages": {"demo": "claude"}}) == "claude"
    assert llm.backend_for("other", {"stages": {"demo": "claude"}}) == "ollama"

def test_unknown_tier_rejected(monkeypatch):
    use(monkeypatch, "mock")
    with pytest.raises(llm.LLMError, match="unknown tier"):
        call(tier="huge", mock=GOOD)


# ---------- mock ----------

def test_mock_returns_validated_output(monkeypatch, isolated):
    use(monkeypatch, "mock")
    assert call(mock=GOOD) == GOOD
    log = (isolated / "r1" / "llm_log.jsonl").read_text().splitlines()
    assert json.loads(log[-1])["ok"] is True

def test_bad_mock_is_rejected(monkeypatch):
    use(monkeypatch, "mock")
    with pytest.raises(llm.LLMError, match="keywords"):
        call(mock={"summary": "no keywords"})


# ---------- claude (file handoff) ----------

def test_claude_writes_prompt_then_waits(monkeypatch, isolated):
    use(monkeypatch, "claude")
    with pytest.raises(HandoffPending):
        call()
    prompt = (isolated / "r1" / "demo.prompt.md").read_text()
    assert "demo.response.json" in prompt and '"keywords"' in prompt

def test_claude_reads_response_file(monkeypatch, isolated):
    use(monkeypatch, "claude")
    (isolated / "r1").mkdir()
    (isolated / "r1" / "demo.response.json").write_text(json.dumps(GOOD))
    assert call() == GOOD

def test_claude_bad_response_says_how_to_fix(monkeypatch, isolated):
    use(monkeypatch, "claude")
    (isolated / "r1").mkdir()
    (isolated / "r1" / "demo.response.json").write_text('{"summary": 1}')
    with pytest.raises(llm.LLMError, match="Fix the file"):
        call()


# ---------- ollama (fake server) ----------

class FakeOllama(BaseHTTPRequestHandler):
    replies: list = []
    requests: list = []
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeOllama.requests.append(body)
        content = FakeOllama.replies.pop(0)
        out = json.dumps({"message": {"content": content},
                          "prompt_eval_count": 10, "eval_count": 5}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(out)
    def log_message(self, *a):
        pass

@pytest.fixture
def ollama(monkeypatch):
    srv = HTTPServer(("127.0.0.1", 0), FakeOllama)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setattr(llm, "OLLAMA_HOST", f"http://127.0.0.1:{srv.server_port}")
    use(monkeypatch, "ollama")
    FakeOllama.requests = []
    yield FakeOllama
    srv.shutdown()

def test_ollama_valid_first_try(ollama, isolated):
    ollama.replies = [json.dumps(GOOD)]
    assert call() == GOOD
    sent = ollama.requests[0]
    assert sent["format"]["properties"]["keywords"]            # schema sent as format
    entry = json.loads((isolated / "r1" / "llm_log.jsonl").read_text().splitlines()[-1])
    assert entry["tokens_in"] == 10 and entry["repaired"] is False

def test_ollama_one_repair_turn(ollama, isolated):
    ollama.replies = ['{"summary": "x"}', json.dumps(GOOD)]
    assert call() == GOOD
    repair_msgs = ollama.requests[1]["messages"]
    assert repair_msgs[-1]["role"] == "user" and "rejected" in repair_msgs[-1]["content"]
    entry = json.loads((isolated / "r1" / "llm_log.jsonl").read_text().splitlines()[-1])
    assert entry["repaired"] is True

def test_ollama_gives_up_after_one_repair(ollama):
    ollama.replies = ["not json", "still not json"]
    with pytest.raises(llm.LLMError, match="after 1 repair turn"):
        call()

def test_ollama_down_fails_cleanly(monkeypatch):
    use(monkeypatch, "ollama")
    monkeypatch.setattr(llm, "OLLAMA_HOST", "http://127.0.0.1:9")    # nothing listens
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    with pytest.raises(llm.LLMError, match="not reachable"):
        call()