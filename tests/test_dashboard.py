"""[I08] dashboard API tests: offline, mock backend, same sandbox as the orchestrator tests."""
import json
import threading
import time
import urllib.request

import pytest

from core import llm
from core.base_agent import AgentError
from dashboard import server as D
from tests.test_orchestrator import sb  # noqa: F401  (sandbox fixture)


def wait_idle(run_id, timeout=30):
    t = time.time()
    while run_id in D._busy:
        assert time.time() - t < timeout, "background work did not finish"
        time.sleep(0.05)
    assert run_id not in D._errors, D._errors.get(run_id)


def test_full_run_through_dashboard(sb):
    rid = D.start(sb["zip"], "d1")
    wait_idle(rid)
    d = D.run_detail(rid)
    assert d["summary"]["gate"] == "G1" and len(d["stages"]) == 13
    assert [s["status"] for s in d["stages"][:5]] == ["success"] * 4 + ["gate"]
    assert d["outputs"]["A01"]["file_count"] == 5 and "key_files" in d["outputs"]["A01"]

    D.pick(rid, 1); wait_idle(rid)
    d = D.run_detail(rid)
    assert d["summary"]["gate"] == "G2" and d["package_md"] and "A10" in d["outputs"]

    D.approve(rid); wait_idle(rid)
    d = D.run_detail(rid)
    assert d["summary"]["status"] == "done" and d["stages"][-1]["status"] == "success"
    assert D.memory_view()["counts"]["posts"] == 1

def test_claude_handoff_paste_answer(sb, monkeypatch):
    monkeypatch.setattr(llm, "BACKEND", "claude")
    rid = D.start(sb["zip"], "d2"); wait_idle(rid)
    d = D.run_detail(rid)
    assert d["summary"]["status"] == "waiting" and d["pending"] == ("A03", "analyzer")
    assert d["stages"][2]["status"] == "waiting"
    assert "Claude stage: analyzer" in D.get_prompt(rid)["text"]
    with pytest.raises(AgentError, match="not valid JSON"):
        D.submit_response(rid, "not json")
    from agents.analyzer.agent import mock_output
    from core import orchestrator as O
    answer = json.dumps(mock_output(O.load(rid)["ctx"]["repo"]))
    D.submit_response(rid, "```json\n" + answer + "\n```"); wait_idle(rid)   # fence tolerated
    assert D.run_detail(rid)["stages"][2]["status"] == "success"

def test_upload_zip_starts_run(sb):
    rid = D.upload_and_start("demo-main.zip", open(sb["zip"], "rb").read())
    wait_idle(rid)
    assert D.run_detail(rid)["summary"]["gate"] == "G1"
    with pytest.raises(AgentError, match="only .zip"):
        D.upload_and_start("evil.exe", b"x")

def test_reject_and_bad_input(sb):
    rid = D.start(sb["zip"], "d3"); wait_idle(rid)
    D.pick(rid, 1); wait_idle(rid)
    D.reject(rid, "not now")
    d = D.run_detail(rid)
    assert d["summary"]["status"] == "rejected" and d["stages"][-1]["status"] == "cancelled"
    with pytest.raises(AgentError, match="bad run id"):
        D.run_detail("../etc")
    with pytest.raises(AgentError, match="give a GitHub URL"):
        D.start("  ")

def test_errors_surface_not_crash(sb):
    rid = D.start(sb["zip"], "d4"); wait_idle(rid)
    D.approve(rid)                                   # wrong gate, runs in background
    while rid in D._busy:
        time.sleep(0.05)
    assert "not at G2" in D.run_detail(rid)["error"]
    D._errors.pop(rid)

def test_http_smoke(sb):
    httpd = D.serve(0)
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        html = urllib.request.urlopen(f"http://127.0.0.1:{port}/").read().decode()
        assert "content-os" in html
        assert json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/api/runs").read()) == []
        req = urllib.request.Request(f"http://127.0.0.1:{port}/api/runs", data=b'{"source": ""}',
                                     method="POST", headers={"Content-Type": "application/json"})
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(req)
        assert e.value.code == 400
    finally:
        httpd.shutdown()
