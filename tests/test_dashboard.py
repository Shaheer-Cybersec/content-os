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
    assert "Content OS stage: analyzer" in D.get_prompt(rid)["text"]
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


def test_mode_switch_and_status(sb, monkeypatch, tmp_path):
    from core import llm
    monkeypatch.setattr(D, "SETTINGS_FILE", tmp_path / "dashboard.json")
    assert D.get_mode() == "mock"
    with pytest.raises(AgentError, match="mock"):
        D.set_mode("auto")
    monkeypatch.setattr(llm, "BACKEND", "claude")
    assert D.set_mode("auto")["backend"] == "claude_cli" and llm.BACKEND == "claude_cli"
    assert json.loads((tmp_path / "dashboard.json").read_text())["mode"] == "auto"
    assert D.set_mode("manual")["mode"] == "manual"
    with pytest.raises(AgentError, match="auto or manual"):
        D.set_mode("turbo")
    assert set(D.claude_status()) == {"found", "path", "version"}

def test_live_feed_and_activity_log(sb, monkeypatch):
    from core import llm
    monkeypatch.setattr(llm, "BACKEND", "claude")
    rid = D.start(sb["zip"], "d5"); wait_idle(rid)
    (sb["root"] / "runs" / rid / "analyzer.live.txt").write_text('{"summary": "stream', encoding="utf-8")
    d = D.run_detail(rid)
    assert d["live"]["tag"] == "A03" and d["live"]["text"].endswith("stream")
    kinds = {(e["kind"], e["tag"]) for e in d["events"]}
    assert ("stage", "A01") in kinds and ("llm", "A03") in kinds

def test_g1_shows_confidence_why_and_recommendation(sb):
    rid = D.start(sb["zip"], "d6"); wait_idle(rid)
    s = D.run_detail(rid)["summary"]
    assert s["recommendation"]["title"] == s["angles"][0]["title"]
    assert s["angles"][0]["recommended"] is True and s["angles"][0]["why"]


def test_linkedin_kit_view_and_save(sb):
    import base64 as b64
    import io
    from pathlib import Path

    from PIL import Image
    rid = D.start(sb["zip"], "d7"); wait_idle(rid)
    D.pick(rid, 1); wait_idle(rid)
    k = D.run_detail(rid)["kit"]
    assert k["saved"] and k["has_pdf"] and not k["has_png"]           # server graphic first, canvas not yet
    assert k["caption"] == D.run_detail(rid)["summary"]["package"]["text"]
    buf = io.BytesIO(); Image.new("RGB", (1200, 627), "#123456").save(buf, "PNG")
    png = "data:image/png;base64," + b64.b64encode(buf.getvalue()).decode()
    k2 = D.save_kit(rid, png)
    assert k2["has_png"] is True and k2["has_pdf"] is True
    assert (Path(k2["dir"]) / "03_graphic.png").read_bytes() == buf.getvalue()
    assert D.report_pdf(rid).startswith(b"%PDF")
    with pytest.raises(AgentError, match="PNG"):
        D.save_kit(rid, "data:image/png;base64," + b64.b64encode(b"GIF89a").decode())
