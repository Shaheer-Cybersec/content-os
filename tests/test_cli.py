"""[I09] CLI tests: drive the orchestrator through run.py, offline, mock backend."""
import json

import run
from tests.test_orchestrator import sb  # noqa: F401  (reuse the sandbox fixture)


def test_full_run_through_cli(sb, capsys):
    assert run.main(["start", sb["zip"], "--id", "c1"]) == 0
    out = capsys.readouterr().out
    assert "gate G1" in out and "python run.py pick c1" in out

    assert run.main(["pick", "c1", "1"]) == 0
    assert "G2  review" in capsys.readouterr().out

    assert run.main(["approve", "c1"]) == 0
    out = capsys.readouterr().out
    assert "status done" in out and "saved    post 1" in out

def test_json_output_and_list(sb, capsys):
    run.main(["--json", "start", sb["zip"], "--id", "c2"])
    s = json.loads(capsys.readouterr().out)
    assert s["gate"] == "G1" and s["angles"]
    run.main(["--json", "list"])
    rows = json.loads(capsys.readouterr().out)
    assert rows[0]["run_id"] == "c2"

def test_errors_exit_1(sb, capsys):
    assert run.main(["status", "missing"]) == 1
    assert "no run" in capsys.readouterr().out
    run.main(["start", sb["zip"], "--id", "c3"])
    capsys.readouterr()
    assert run.main(["--json", "approve", "c3"]) == 1
    assert "not at G2" in json.loads(capsys.readouterr().out)["error"]

def test_reject(sb, capsys):
    run.main(["start", sb["zip"], "--id", "c4"])
    run.main(["pick", "c4", "1"])
    capsys.readouterr()
    assert run.main(["reject", "c4", "not now"]) == 0
    assert "note     not now" in capsys.readouterr().out