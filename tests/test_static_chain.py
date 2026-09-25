"""
[M1] static chain: A01 -> A02 -> (fixture draft) -> A10 -> (approve) -> A11 -> A02
No LLM anywhere. Proves the four static agents hand data to each other correctly.

Offline by default (a zip built in the test). Set CONTENTOS_LIVE=1 to also run
the same chain against a real GitHub repo over the network.
"""
import os
import zipfile

import pytest

from agents.memory_retrieve.agent import MemoryRetrieve
from agents.memory_writer import agent as MW
from agents.memory_writer.agent import MemoryWriter
from agents.packager import agent as PK
from agents.packager.agent import Packager
from agents.repo_ingest import agent as RI
from agents.repo_ingest.agent import RepoIngest
from memory import db
from memory import embeddings as E

SHA = "672971d66a2ef9f85151e53283113f33d642dabd"
FILES = {
    "README.md": "# chain demo",
    "requirements.txt": "fastapi\n",
    "app/main.py": "TOP_K = 8\n",
    "app/retriever.py": "def retrieve(q, k=8): ...\n",
}


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    for mod, attr, sub in [(RI, "CACHE_DIR", "cache"), (MW, "OBSIDIAN_DIR", "obsidian"),
                           (PK, "OUTPUTS_DIR", "outputs")]:
        d = tmp_path / sub
        d.mkdir()
        monkeypatch.setattr(mod, attr, d)
    monkeypatch.setattr(MW, "DATA_DIR", tmp_path)
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "chain.db")
    monkeypatch.setattr(E, "embed", lambda text: [0.6, 0.8])
    zp = tmp_path / "chain-demo-main.zip"
    with zipfile.ZipFile(zp, "w") as z:
        for rel, text in FILES.items():
            z.writestr("chain-demo-main/" + rel, text)
        z.comment = SHA.encode()
    return {"zip": zp, "root": tmp_path}


def step(agent, ctx):
    rec = agent.execute(ctx)
    assert rec["status"] == "success", f"{rec['id']} {rec['agent']}: {rec['error']}"
    return rec


def run_chain(source, cite_path, cite_ref):
    ctx = {"source": source, "run_id": "m1-run-001"}
    trail = [step(RepoIngest(), ctx), step(MemoryRetrieve(), ctx)]
    assert ctx["memory"]["past_post_count"] == 0            # fresh database

    # Stand-in for A03-A09: a draft citing a file A01 really saw
    angles = [{"title": "Top-k of 8 widens the injection surface", "score": 0.9, "dedup_status": "ok"},
              {"title": "A RAG lab in four files", "score": 0.7, "dedup_status": "ok"}]
    ctx["chosen_angle"], ctx["angles"] = angles[0], angles
    ctx["draft"] = {"text": "My RAG lab retrieves 8 chunks. That is 8 chances for a poisoned doc.",
                    "claims": [{"claim": "top-k is 8", "source_path": cite_path, "source_ref": cite_ref}]}
    ctx["visual_plan"] = [{"shot": "top_k line", "how": "VS Code crop"}]
    trail.append(step(Packager(), ctx))

    ctx["approved_post"] = ctx["package"]["post"]           # G2: you said yes
    trail.append(step(MemoryWriter(), ctx))

    again = {"repo": ctx["repo"]}                           # the NEXT run on this repo
    step(MemoryRetrieve(), again)
    return ctx, again["memory"], trail


def test_static_chain_offline(sandbox):
    ctx, mem, trail = run_chain(str(sandbox["zip"]), "app/main.py", SHA[:7])

    assert [r["id"] for r in trail] == ["A01", "A02", "A10", "A11"]
    assert ctx["repo"]["file_count"] == 4
    assert len(list((sandbox["root"] / "outputs").glob("*.md"))) == 1
    assert ctx["memory_write"]["post_id"] == 1
    # The next run sees the post (for dedup) and the unused angle (queued)
    assert mem["past_post_count"] == 1
    assert [a["title"] for a in mem["queued_angles"]] == ["A RAG lab in four files"]


def test_chain_stops_at_packager_on_invented_file(sandbox):
    ctx = {"source": str(sandbox["zip"]), "run_id": "m1-bad"}
    step(RepoIngest(), ctx)
    ctx["draft"] = {"text": "Guardrails block it.",
                    "claims": [{"claim": "has guardrails", "source_path": "app/guardrails.py",
                                "source_ref": SHA[:7]}]}
    rec = Packager().execute(ctx)
    assert rec["status"] == "failed" and "path not in repo" in rec["error"]
    assert "package" not in ctx                             # nothing reaches approval


@pytest.mark.skipif(os.getenv("CONTENTOS_LIVE") != "1", reason="set CONTENTOS_LIVE=1 for the network run")
def test_static_chain_live_git(sandbox):
    ctx, mem, _ = run_chain("https://github.com/pallets/itsdangerous",
                            "src/itsdangerous/signer.py", "672971d")
    assert ctx["repo"]["source_type"] == "git" and len(ctx["repo"]["commits"]) == 30
    assert mem["past_post_count"] == 1