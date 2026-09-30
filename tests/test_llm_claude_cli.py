"""[I06] claude_cli backend, tested against a fake `claude` program (never the real one)."""
import json
import os
import stat
import sys

import pytest
from pydantic import BaseModel

from core import llm

FAKE = r'''#!{py}
import json, os, sys
prompt = sys.stdin.read()
mode = os.environ.get("FAKE_MODE", "good")
count_file = os.environ["FAKE_COUNT"]
n = int(open(count_file).read()) + 1 if os.path.exists(count_file) else 1
open(count_file, "w").write(str(n))
open(os.environ["FAKE_ARGS"], "w").write(json.dumps(sys.argv[1:]))
def ev(o): print(json.dumps(o), flush=True)
if mode == "noresult":
    sys.exit(3)
if mode == "error":
    ev({{"type": "result", "is_error": True, "result": "Not logged in"}}); sys.exit(1)
if mode == "fix" and n == 1:
    answer = "Here you go: {{\"summary\": 5}}"
else:
    answer = "```json\n{{\"summary\": \"ok\", \"keywords\": [\"a\"]}}\n```"
for chunk in (answer[:10], answer[10:]):
    ev({{"type": "stream_event", "event": {{"type": "content_block_delta", "delta": {{"type": "text_delta", "text": chunk}}}}}})
ev({{"type": "result", "is_error": False, "result": answer}})
'''


class Demo(BaseModel):
    summary: str
    keywords: list[str]


@pytest.fixture
def fake(tmp_path, monkeypatch):
    if sys.platform == "win32":
        pytest.skip("fake CLI script is POSIX-only; the real CLI is covered by the manual check")
    exe = tmp_path / "claude"
    exe.write_text(FAKE.format(py=sys.executable), encoding="utf-8")
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setattr(llm, "CLAUDE_BIN", str(exe))
    monkeypatch.setattr(llm, "BACKEND", "claude_cli")
    monkeypatch.setattr(llm, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setenv("FAKE_COUNT", str(tmp_path / "count"))
    monkeypatch.setenv("FAKE_ARGS", str(tmp_path / "args.json"))
    return tmp_path


def ask():
    return llm.call(stage="demo", tier="mid", system="s", user="u", schema=Demo, run_id="r")


def test_answer_streamed_parsed_and_saved(fake):
    assert ask() == {"summary": "ok", "keywords": ["a"]}
    d = fake / "runs" / "r"
    assert "keywords" in (d / "demo.live.txt").read_text(encoding="utf-8")
    assert json.loads((d / "demo.response.json").read_text(encoding="utf-8"))["summary"] == "ok"
    args = json.loads((fake / "args.json").read_text())
    assert args[args.index("--tools") + 1] == "" and "-p" in args          # tools off, print mode
    assert args[args.index("--model") + 1] == "opus"

def test_saved_answer_reused_without_calling_again(fake):
    ask(); ask()
    assert (fake / "count").read_text() == "1"

def test_bad_json_gets_one_repair_turn(fake, monkeypatch):
    monkeypatch.setenv("FAKE_MODE", "fix")
    assert ask()["summary"] == "ok"
    assert (fake / "count").read_text() == "2"
    log = (fake / "runs" / "r" / "llm_log.jsonl").read_text().splitlines()[-1]
    assert json.loads(log)["repaired"] is True

def test_cli_error_and_missing_result_fail_clearly(fake, monkeypatch):
    monkeypatch.setenv("FAKE_MODE", "error")
    with pytest.raises(llm.LLMError, match="Not logged in"):
        ask()
    monkeypatch.setenv("FAKE_MODE", "noresult")
    with pytest.raises(llm.LLMError, match="without an answer"):
        ask()

def test_missing_cli_explains_how_to_install(fake, monkeypatch):
    monkeypatch.setattr(llm, "CLAUDE_BIN", str(fake / "nope"))
    with pytest.raises(llm.LLMError, match="npm install -g @anthropic-ai/claude-code"):
        ask()
