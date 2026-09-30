"""[A10] checkpoint tests. Output goes to a temp folder."""
import copy
from datetime import datetime
from pathlib import Path

import pytest

from agents.packager import agent as PK
from agents.packager.agent import FIXTURE, Packager


@pytest.fixture(autouse=True)
def out_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(PK, "OUTPUTS_DIR", tmp_path)
    return tmp_path

def ctx():
    return copy.deepcopy(FIXTURE)

def run(c):
    rec = Packager().execute(c)
    return rec, c.get("package")


def test_packages_valid_draft(out_dir):
    rec, pkg = run(ctx())
    assert rec["status"] == "success"
    assert pkg["evidence_checked"] == 2 and pkg["warnings"] == []
    d = out_dir / "damn-vulnerable-rag" / f"{datetime.now():%Y-%m-%d}_fixture-run-001"
    assert pkg["folder"] == d.as_posix() and pkg["path"] == (d / "00_START_HERE.md").as_posix()
    assert "`app/retriever.py`" in (d / "06_evidence.md").read_text(encoding="utf-8")
    assert "- [ ] **retriever.py top_k line**" in (d / "05_visuals.md").read_text(encoding="utf-8")

def test_final_draft_wins_over_draft():
    c = ctx()
    c["final_draft"] = {"text": "Revised version.", "claims": []}
    _, pkg = run(c)
    assert pkg["post"]["text"] == "Revised version."

def test_invented_file_blocks_packaging(out_dir):
    c = ctx()
    c["draft"]["claims"][0]["source_path"] = "app/guardrails.py"
    rec, pkg = run(c)
    assert rec["status"] == "failed" and "path not in repo" in rec["error"]
    assert pkg is None and list(out_dir.iterdir()) == []       # nothing written

def test_invented_commit_blocks_packaging():
    c = ctx()
    c["draft"]["claims"][1]["source_ref"] = "deadbee"
    rec, _ = run(c)
    assert rec["status"] == "failed" and "unknown commit" in rec["error"]

def test_over_linkedin_limit_fails():
    c = ctx()
    c["draft"]["text"] = "x" * 3001
    rec, _ = run(c)
    assert rec["status"] == "failed" and "3001 chars" in rec["error"]

def test_links_and_hashtags_warn_but_package():
    c = ctx()
    c["draft"]["text"] += "\nhttps://github.com/x/y #a #b #c"
    rec, pkg = run(c)
    assert rec["status"] == "success"
    assert len(pkg["warnings"]) == 2

def test_no_draft_fails_cleanly():
    c = ctx()
    del c["draft"]
    rec, _ = run(c)
    assert rec["status"] == "failed" and "no final_draft or draft" in rec["error"]

def test_post_handed_on_for_approval():
    _, pkg = run(ctx())
    assert set(pkg["post"]) == {"text", "evidence"}

def test_accepts_a06_dict_shape(out_dir):
    c = ctx()
    c["visual_plan"] = {"shots": [{"shot": "decode in terminal", "how": "run two lines"}],
                        "rationale": "r", "dropped": []}
    rec, pkg = run(c)
    assert rec["status"] == "success"
    body = (Path(pkg["folder"]) / "05_visuals.md").read_text(encoding="utf-8")
    assert "- [ ] **decode in terminal**" in body and "run two lines" in body


def test_folder_layout_and_kit(out_dir):
    c = ctx()
    c["visual_plan"] = {"shots": [], "rationale": "r",
                        "graphic": {"style": "quote", "headline": "Top-k 8 widens it", "subline": "sub"},
                        "image_prompts": [{"tool": "Canva", "prompt": "dark banner, one number"}]}
    rec, pkg = run(c)
    kit, d = pkg["kit"], Path(pkg["folder"])
    assert d.parent == out_dir / "damn-vulnerable-rag"
    names = sorted(p.name for p in d.iterdir())
    for n in ["00_START_HERE.md", "01_caption.md", "02_first_comment.md", "03_graphic.png", "04_alt_text.md",
              "05_visuals.md", "06_evidence.md", "07_run_summary.md", "copy-paste.html", "manifest.json"]:
        assert n in names, n
    assert (d / "01_caption.md").read_text(encoding="utf-8").strip() == pkg["post"]["text"]
    assert kit["first_comment"].startswith("Code for this post: https://github.com/")
    assert "app/retriever.py" in kit["first_comment"] and kit["alt_text"] == "Graphic: Top-k 8 widens it. sub"
    assert "dark banner, one number" in (d / "05_visuals.md").read_text(encoding="utf-8")


def test_graphic_png_is_linkedin_size(out_dir):
    PIL = pytest.importorskip("PIL.Image")
    for style in ("terminal", "quote", "stat"):
        c = ctx()
        c["visual_plan"] = {"shots": [], "graphic": {"style": style, "headline": "Top-k 8 widens the attack surface",
                                                    "subline": "More chunks, more chances", "stat": "8"}}
        _, pkg = run(c)
        img = PIL.open(Path(pkg["folder"]) / "03_graphic.png")
        assert img.size == (1200, 627)


def test_pdf_and_html_are_built(out_dir):
    pytest.importorskip("reportlab")
    _, pkg = run(ctx())
    d = Path(pkg["folder"])
    pdf = (d / "00_REPORT.pdf").read_bytes()
    assert pdf.startswith(b"%PDF") and len(pdf) > 8000
    html = (d / "copy-paste.html").read_text(encoding="utf-8")
    assert html.count("Copy</button>") >= 3 and pkg["post"]["text"].splitlines()[0] in html


def test_long_post_with_odd_characters_still_makes_pdf(out_dir):
    pytest.importorskip("reportlab")
    c = ctx()
    c["draft"]["text"] = "\n\n".join(f"Paragraph {i}: caf\u00e9 <b>&</b> arrow \u2192 done" + " word" * 40 for i in range(12))
    rec, pkg = run(c)
    assert rec["status"] == "success" and (Path(pkg["folder"]) / "00_REPORT.pdf").stat().st_size > 8000


def test_missing_pdf_or_png_libs_never_block_the_post(out_dir, monkeypatch):
    from core import report
    def boom(*a, **k):
        raise ImportError("nope")
    monkeypatch.setattr(report, "render_graphic", boom)
    monkeypatch.setattr(report, "_pdf", boom)
    rec, pkg = run(ctx())
    d = Path(pkg["folder"])
    assert rec["status"] == "success" and (d / "01_caption.md").exists()
    assert not (d / "00_REPORT.pdf").exists() and not (d / "03_graphic.png").exists()
    assert any("pip install pillow" in w for w in pkg["warnings"]) and any("reportlab" in w for w in pkg["warnings"])


def test_records_are_drawn_but_never_saved_in_state():
    from core import orchestrator as O
    src = open(O.__file__, encoding="utf-8").read()
    assert 'ctx.pop("_records"' in src
