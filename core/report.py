"""
[I10] report - the finished-post folder for one run. Pure code, no LLM.

    outputs/<repo>/<date>_<run>/
        00_REPORT.pdf        ONE visual PDF with everything (preview, kit, visuals, evidence, how it was made)
        00_START_HERE.md     file index + the 5 posting steps
        01_caption.md        the post text, exactly as LinkedIn should get it
        02_first_comment.md  paste as your first comment
        03_graphic.png       1200x627 image
        04_alt_text.md       paste as the image's alt text
        05_visuals.md        screenshots to take + prompts for Canva / image AIs
        06_evidence.md       every claim -> file -> commit
        07_run_summary.md    angle chosen, confidence, critic scores, pipeline timings
        copy-paste.html      one page with a Copy button per block (open in any browser)
        manifest.json        machine-readable index

PNG + PDF need `pip install pillow reportlab`. Without them everything else is still
written and `manifest.json` says what was skipped.
"""
from __future__ import annotations

import base64
import io
import json
import re
from datetime import datetime
from html import escape as hesc
from pathlib import Path
from xml.sax.saxutils import escape as xesc

LINKEDIN_MAX_CHARS = 3000
GFX_W, GFX_H = 1200, 627

HOW_TO_POST = [
    "LinkedIn -> Start a post. Paste 01_caption.md as the text.",
    "Add 03_graphic.png with the image button. Paste 04_alt_text.md as its alt text.",
    "Post it.",
    "Right away, comment on your own post with 02_first_comment.md (links in the body cut reach).",
    "Reply to every comment in the first hour.",
]

FILES = [
    ("00_REPORT.pdf", "everything in one visual PDF"),
    ("00_START_HERE.md", "this index + the posting steps"),
    ("01_caption.md", "post text, paste as-is"),
    ("02_first_comment.md", "first comment, paste after posting"),
    ("03_graphic.png", "image, 1200x627"),
    ("04_alt_text.md", "image alt text"),
    ("05_visuals.md", "screenshots to take + image-AI prompts"),
    ("06_evidence.md", "claim -> file -> commit"),
    ("07_run_summary.md", "angle, confidence, critic scores, timings"),
    ("copy-paste.html", "one page with Copy buttons"),
]

STAGE_NAMES = {
    "A01": "repo ingest", "A02": "memory", "A03": "analyzer", "A04": "angles", "G1": "you pick",
    "A05": "strategist", "A06": "visuals", "A07": "writer", "A08": "critic", "A09": "reviser",
    "A10": "packager", "G2": "you approve", "A11": "memory write",
}


# ---------------------------------------------------------------- names

def safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", name or "").strip("-.") or "repo"


def run_folder(outputs_dir, repo: dict, run_id: str, date: str | None = None) -> Path:
    """outputs/<repo>/<date>_<run>/"""
    date = date or f"{datetime.now():%Y-%m-%d}"
    return Path(outputs_dir) / safe(repo.get("repo", "")) / f"{date}_{safe(run_id)}"


# ---------------------------------------------------------------- model

def linkedin_kit(repo: dict, text: str, claims: list[dict], visual_plan) -> dict:
    """Caption, first comment and alt text for one post."""
    url = repo.get("url") or ""
    sha = (repo.get("head_sha") or "")[:7]
    files = list(dict.fromkeys(c["source_path"] for c in claims))
    first = []
    if url.startswith("https://github.com/"):
        first.append(f"Code for this post: {url}")
    if files:
        first.append("Files referenced: " + ", ".join(files[:4]) + (f" (commit {sha})" if sha else ""))
    first.append("Every technical claim in the post points at one of these files.")
    g = (visual_plan or {}).get("graphic") if isinstance(visual_plan, dict) else None
    if g:
        alt = f"Graphic: {g['headline']}" + (f". {g['subline']}" if g.get("subline") else "")
    else:
        alt = "Graphic: " + (text.splitlines() or [""])[0][:200]
    return {"caption": text, "first_comment": "\n".join(first), "alt_text": alt[:1000]}


def collect(ctx: dict, records: dict | None = None, date: str | None = None) -> dict:
    """Everything the report shows, gathered once from the run context."""
    pkg = ctx.get("package") or {}
    post = pkg.get("post") or {}
    fd = ctx.get("final_draft") or ctx.get("draft") or {}
    text = (post.get("text") or fd.get("text") or "").strip()
    claims = post.get("evidence") or fd.get("claims") or []
    repo = ctx["repo"]
    vp = ctx.get("visual_plan") or {}
    vp = vp if isinstance(vp, dict) else {"shots": vp}
    kit = linkedin_kit(repo, text, claims, vp)
    kit.update({k: v for k, v in (pkg.get("kit") or {}).items() if k in kit})
    angle = ctx.get("chosen_angle") or {}
    angles = (ctx.get("angle_set") or {}).get("angles") or []
    crit = ctx.get("critique") or {}
    graphic = vp.get("graphic") or {"style": "terminal", "headline": (text.splitlines() or [""])[0][:90],
                                    "subline": None, "stat": None}
    recs = records or ctx.get("_records") or {}
    pipeline = []
    for tag in ("A01", "A02", "A03", "A04", "G1", "A05", "A06", "A07", "A08", "A09", "A10", "G2", "A11"):
        r = recs.get(tag) or {}
        if tag.startswith("G"):
            done = (tag == "G1" and "chosen_angle" in ctx) or (tag == "G2" and "approved_post" in ctx)
            pipeline.append({"tag": tag, "name": STAGE_NAMES[tag], "status": "you" if done else "pending", "secs": None})
        else:
            pipeline.append({"tag": tag, "name": STAGE_NAMES[tag],
                             "status": r.get("status") or ("success" if tag == "A10" else "pending"),
                             "secs": (r["duration_ms"] / 1000) if r.get("duration_ms") is not None else None})
    return {
        "repo": repo, "run_id": ctx["run_id"], "date": date or f"{datetime.now():%Y-%m-%d}",
        "title": angle.get("title") or (angles[0]["title"] if angles else "Untitled angle"),
        "angle": angle, "angles": angles, "recommendation": (ctx.get("angle_set") or {}).get("recommendation"),
        "strategy": ctx.get("strategy") or {},
        "text": text, "claims": claims, "kit": kit,
        "shots": vp.get("shots") or [], "image_prompts": vp.get("image_prompts") or [],
        "shots_rationale": vp.get("rationale") or "",
        "graphic": graphic,
        "stats": {"chars": len(text), "words": len(text.split()),
                  "hashtags": len(re.findall(r"(?<!\w)#\w+", text)),
                  "links": len(re.findall(r"https?://\S+", text))},
        "warnings": pkg.get("warnings") or [],
        "scores": crit.get("scores") or {}, "flags": crit.get("flags") or [],
        "changes": (ctx.get("final_draft") or {}).get("changes") or [],
        "unresolved": (ctx.get("final_draft") or {}).get("unresolved") or [],
        "pipeline": pipeline,
        "total_secs": sum(p["secs"] for p in pipeline if p["secs"]),
    }


def _fmt_secs(x) -> str:
    return "" if x is None else f"{x:.1f}s"


def _repo_label(m: dict) -> str:
    r = m["repo"]
    return f"{r.get('owner', '')}/{r.get('repo', '')}".strip("/")


# ---------------------------------------------------------------- fonts

def _first(paths):
    for p in paths:
        if Path(p).exists():
            return p
    return None


_WIN = "C:/Windows/Fonts/"
_DJ = "/usr/share/fonts/truetype/dejavu/"
_MAC = "/System/Library/Fonts/Supplemental/"
FONT_FILES = {
    "sans": [_WIN + "segoeui.ttf", _WIN + "arial.ttf", _DJ + "DejaVuSans.ttf", _MAC + "Arial.ttf"],
    "sansb": [_WIN + "segoeuib.ttf", _WIN + "arialbd.ttf", _DJ + "DejaVuSans-Bold.ttf", _MAC + "Arial Bold.ttf"],
    "mono": [_WIN + "consola.ttf", _WIN + "cour.ttf", _DJ + "DejaVuSansMono.ttf", _MAC + "Courier New.ttf"],
    "monob": [_WIN + "consolab.ttf", _WIN + "courbd.ttf", _DJ + "DejaVuSansMono-Bold.ttf", _MAC + "Courier New Bold.ttf"],
}


# ---------------------------------------------------------------- graphic (Pillow)

def render_graphic(m: dict, style: str | None = None) -> bytes:
    """1200x627 PNG drawn from A06's graphic plan. Same three templates as the dashboard."""
    from PIL import Image, ImageDraw, ImageFont

    g = m["graphic"]
    style = style or g.get("style") or "terminal"
    W, H = GFX_W, GFX_H

    def font(kind, size):
        p = _first(FONT_FILES[kind])
        try:
            return ImageFont.truetype(p, size) if p else ImageFont.load_default(size)
        except Exception:
            return ImageFont.load_default(size)

    def wrap(d, text, f, maxw, n):
        words, lines, cur = str(text or "").split(), [], ""
        for w in words:
            t = f"{cur} {w}".strip()
            if d.textlength(t, font=f) > maxw and cur:
                lines.append(cur)
                cur = w
            else:
                cur = t
        if cur:
            lines.append(cur)
        if len(lines) > n:
            lines = lines[:n]
            lines[-1] = lines[-1].rstrip(" .,;:") + "..."
        return lines

    img = Image.new("RGB", (W, H), "#070b10")
    d = ImageDraw.Draw(img)
    repo = _repo_label(m)
    ACC, TXT, MUT = "#3ee6a8", "#e6edf3", "#8b9bb0"
    head, sub = g.get("headline") or "", g.get("subline")

    if style == "terminal":
        d.rounded_rectangle((60, 60, W - 60, H - 90), 18, fill="#0e151e")
        d.rounded_rectangle((60, 60, W - 60, 106), 18, fill="#131c27")
        d.rectangle((60, 90, W - 60, 106), fill="#131c27")
        for i, c in enumerate(("#ff5f57", "#febc2e", "#28c840")):
            d.ellipse((85 + i * 26, 76, 99 + i * 26, 90), fill=c)
        d.text((190, 83), repo or "shaheersec", font=font("mono", 18), fill="#5b6b80", anchor="lm")
        f = font("monob", 40)
        y = 130
        d.text((100, y), "$", font=f, fill=ACC)
        for line in wrap(d, head, f, W - 270, 4):
            d.text((145, y), line, font=f, fill=TXT)
            y += 56
        if sub:
            f2 = font("mono", 24)
            for line in wrap(d, "# " + sub, f2, W - 230, 2):
                d.text((100, y + 14), line, font=f2, fill=MUT)
                y += 34
        d.rectangle((100, min(y + 30, H - 130), 122, min(y + 30, H - 130) + 34), fill=ACC)
    elif style == "quote":
        for y in range(H):                      # diagonal-ish vertical gradient
            t = y / H
            d.line((0, y, W, y), fill=(int(11 + 2 * t), int(19 + 23 * t), int(32 + 8 * t)))
        d.rectangle((80, 110, 88, 410), fill=ACC)
        f = font("sansb", 52)
        y = 120
        for line in wrap(d, head, f, W - 240, 4):
            d.text((120, y), line, font=f, fill="#ffffff")
            y += 66
        if sub:
            f2 = font("sans", 28)
            for line in wrap(d, sub, f2, W - 240, 2):
                d.text((120, y + 20), line, font=f2, fill="#9fb3c8")
                y += 40
    else:                                        # stat
        stat = g.get("stat") or (re.search(r"\$?\d[\d,.%]*|\b[A-Z]{2,}\b", head) or [""])[0] or "!"
        d.ellipse((W * .75 - 380, H * .3 - 380, W * .75 + 380, H * .3 + 380), fill="#0c1f1f")
        d.ellipse((W * .75 - 220, H * .3 - 220, W * .75 + 220, H * .3 + 220), fill="#0f2a27")
        d.text((90, 90), stat, font=font("monob", 150), fill=ACC)
        f = font("sansb", 44)
        y = 330
        for line in wrap(d, head, f, W - 180, 3):
            d.text((95, y), line, font=f, fill=TXT)
            y += 56
        if sub:
            f2 = font("sans", 26)
            for line in wrap(d, sub, f2, W - 180, 1):
                d.text((95, y + 10), line, font=f2, fill=MUT)

    d.text((80, H - 40), "@ShaheerSec", font=font("monob", 22), fill=ACC, anchor="lm")
    r = ("github.com/" + repo) if repo else ""
    d.text((W - 80, H - 40), r, font=font("mono", 18), fill="#5b6b80", anchor="rm")
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


# ---------------------------------------------------------------- markdown files

def _md_files(m: dict, notes: list[str]) -> dict[str, str]:
    kit, s = m["kit"], m["stats"]
    files = {}
    files["01_caption.md"] = kit["caption"].rstrip() + "\n"
    files["02_first_comment.md"] = kit["first_comment"].rstrip() + "\n"
    files["04_alt_text.md"] = kit["alt_text"].rstrip() + "\n"

    idx = "\n".join(f"| `{n}` | {d} |" for n, d in FILES)
    steps = "\n".join(f"{i}. {t}" for i, t in enumerate(HOW_TO_POST, 1))
    warn = ("\n## Warnings\n\n" + "\n".join(f"- {w}" for w in m["warnings"]) + "\n") if m["warnings"] else ""
    note = ("\n## Skipped\n\n" + "\n".join(f"- {n}" for n in notes) + "\n") if notes else ""
    files["00_START_HERE.md"] = (
        f"# {m['title']}\n\n"
        f"- Repo: {m['repo'].get('url', '')} @ `{(m['repo'].get('head_sha') or '')[:12]}`\n"
        f"- Run: `{m['run_id']}` on {m['date']}\n"
        f"- Length: {s['chars']} / {LINKEDIN_MAX_CHARS} chars, {s['words']} words, "
        f"{s['hashtags']} hashtags, {s['links']} links\n{warn}\n"
        f"## Post it in 5 steps\n\n{steps}\n\n## What is in this folder\n\n"
        f"| File | What it is |\n|---|---|\n{idx}\n{note}")

    shots = "\n".join(
        f"- [ ] **{x.get('shot', '?')}**" + (f" (`{x['file_path']}`{', ' + x['lines'] if x.get('lines') else ''})" if x.get("file_path") else "")
        + (f", about {x['minutes']} min" if x.get("minutes") else "") + f"\n  {x.get('how', '')}"
        for x in m["shots"]) or "- (none planned)"
    prompts = "\n\n".join(f"### {p.get('tool', 'Image AI')}\n\n```text\n{p.get('prompt', '')}\n```"
                          for p in m["image_prompts"]) or "(this run has no image prompts)"
    files["05_visuals.md"] = (
        f"# Visuals\n\n## Graphic\n\n![graphic](03_graphic.png)\n\n"
        f"Headline: {m['graphic'].get('headline', '')}\n\n## Screenshots to take\n\n{shots}\n\n"
        f"{('Why these: ' + m['shots_rationale'] + chr(10) + chr(10)) if m['shots_rationale'] else ''}"
        f"## Prompts for Canva / image AIs\n\n{prompts}\n")

    rows = "\n".join(f"| {c['claim']} | `{c['source_path']}` | `{c['source_ref']}` |" for c in m["claims"]) \
        or "| (no factual claims) | | |"
    files["06_evidence.md"] = (
        f"# Evidence\n\nEvery item below was checked against the repo at commit "
        f"`{(m['repo'].get('head_sha') or '')[:12]}`. A post with an invented file or commit is never packaged.\n\n"
        f"| Claim | File | Commit |\n|---|---|---|\n{rows}\n")

    ang = "\n".join(
        f"| {'**' + a['title'] + '**' if a['title'] == m['title'] else a['title']} | {round(a.get('score', 0) * 100)}% | "
        f"{a.get('post_type', '')} | {a.get('dedup_status', '')} |" for a in m["angles"]) or "| | | | |"
    sc = ", ".join(f"{k} {v}/5" for k, v in m["scores"].items()) or "n/a"
    fl = "\n".join(f"- [{f.get('severity')}] {f.get('quote', '')}: {f.get('issue', '')}" for f in m["flags"]) or "- none"
    pl = "\n".join(f"| {p['tag']} | {p['name']} | {p['status']} | "
                   f"{_fmt_secs(p['secs'])} |" for p in m["pipeline"])
    files["07_run_summary.md"] = (
        f"# Run summary\n\n## Angles considered\n\n| Angle | Confidence | Type | Dedup |\n|---|---|---|---|\n{ang}\n\n"
        + (f"Claude recommended: {m['recommendation']['title']}. {m['recommendation'].get('reason', '')}\n\n"
           if m.get("recommendation") else "")
        + f"## Critic\n\nScores: {sc}\n\n{fl}\n\n## Pipeline\n\n| Stage | Agent | Status | Time |\n|---|---|---|---|\n{pl}\n"
        + (f"\nTotal agent time: {m['total_secs']:.0f}s\n" if m["total_secs"] else ""))
    return files


# ---------------------------------------------------------------- copy-paste.html

def _html(m: dict, png: bytes | None) -> str:
    kit = m["kit"]
    blocks = [("Caption", "The post text", kit["caption"]),
              ("First comment", "Post it right after publishing", kit["first_comment"]),
              ("Alt text", "Paste into the image's alt-text field", kit["alt_text"])]
    for p in m["image_prompts"]:
        blocks.append((f"Image prompt: {p.get('tool', '')}", "For Canva or an image AI", p.get("prompt", "")))
    cards = []
    for i, (t, sub, body) in enumerate(blocks):
        cards.append(f'<section><div class="h"><div><h2>{hesc(t)}</h2><span>{hesc(sub)}</span></div>'
                     f'<button onclick="cp({i},this)">Copy</button></div><pre id="b{i}">{hesc(body)}</pre></section>')
    img = ""
    if png:
        uri = "data:image/png;base64," + base64.b64encode(png).decode()
        img = (f'<section><div class="h"><div><h2>Graphic</h2><span>1200 x 627, attach it to the post</span></div>'
               f'<a class="btn" download="graphic.png" href="{uri}">Download PNG</a></div><img src="{uri}" alt=""></section>')
    steps = "".join(f"<li>{hesc(t)}</li>" for t in HOW_TO_POST)
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{hesc(m['title'])} - copy and paste</title><style>
:root{{color-scheme:dark}}body{{margin:0;background:#0b1118;color:#e6edf3;font:15px/1.55 "Segoe UI",system-ui,sans-serif}}
main{{max-width:760px;margin:0 auto;padding:28px 18px 60px}}h1{{font-size:20px;margin:0 0 4px}}.meta{{color:#8b9bb0;font-size:13px;margin-bottom:22px}}
section{{background:#111a24;border:1px solid #1f2d3d;border-radius:12px;padding:16px;margin-bottom:16px}}
.h{{display:flex;justify-content:space-between;align-items:center;gap:12px;margin-bottom:10px}}h2{{font-size:15px;margin:0}}.h span{{color:#8b9bb0;font-size:12.5px}}
pre{{white-space:pre-wrap;word-wrap:break-word;margin:0;font:14px/1.55 "Segoe UI",system-ui,sans-serif;background:#0b1118;border-radius:8px;padding:12px}}
button,.btn{{background:#3ee6a8;color:#04140d;border:0;border-radius:8px;padding:8px 14px;font-weight:600;cursor:pointer;text-decoration:none;font-size:13px;white-space:nowrap}}
img{{width:100%;border-radius:8px;display:block}}ol{{margin:0;padding-left:20px;color:#b7c4d3}}
</style></head><body><main><h1>{hesc(m['title'])}</h1><div class="meta">{hesc(_repo_label(m))} · run {hesc(m['run_id'])} · {hesc(m['date'])}</div>
<section><div class="h"><div><h2>Post it in 5 steps</h2></div></div><ol>{steps}</ol></section>
{cards[0]}{img}{"".join(cards[1:])}
</main><script>
function cp(i,b){{var t=document.getElementById("b"+i).textContent;
function done(){{var o=b.textContent;b.textContent="Copied";setTimeout(function(){{b.textContent=o}},1200)}}
if(navigator.clipboard&&window.isSecureContext){{navigator.clipboard.writeText(t).then(done,fb)}}else fb();
function fb(){{var a=document.createElement("textarea");a.value=t;document.body.appendChild(a);a.select();try{{document.execCommand("copy");done()}}catch(e){{}}a.remove()}}}}
</script></body></html>"""


# ---------------------------------------------------------------- PDF (reportlab)

def _pdf(m: dict, png: bytes | None, path: Path) -> None:
    from reportlab.graphics.shapes import Drawing, Line, Rect, String
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import (BaseDocTemplate, Frame, Image, KeepTogether, PageBreak, PageTemplate,
                                    Paragraph, Spacer, Table, TableStyle)

    # fonts: real TTFs when found (unicode), else built-in Helvetica/Courier (latin-1 only)
    F = {"sans": "Helvetica", "sansb": "Helvetica-Bold", "mono": "Courier", "monob": "Courier-Bold"}
    unicode_ok = False
    for kind in ("sans", "sansb", "mono", "monob"):
        p = _first(FONT_FILES[kind])
        if p:
            try:
                pdfmetrics.registerFont(TTFont(f"CO-{kind}", p))
                F[kind] = f"CO-{kind}"
                unicode_ok = True if kind == "sans" else unicode_ok
            except Exception:
                pass
    if F["sans"] == "Helvetica":
        unicode_ok = False

    def T(s) -> str:                                   # text -> safe paragraph markup
        s = str(s if s is not None else "")
        if not unicode_ok:
            s = s.encode("cp1252", "replace").decode("cp1252")
        return xesc(s).replace("\n", "<br/>")

    NAVY, TEAL, INK, MUTE = colors.HexColor("#0b1320"), colors.HexColor("#1fbf8a"), colors.HexColor("#1b2733"), colors.HexColor("#5f6f80")
    PAGE, CARD, LINE = colors.HexColor("#f3f2ef"), colors.white, colors.HexColor("#d9dee4")
    OKC, BADC, AMB, VIO = colors.HexColor("#1fbf8a"), colors.HexColor("#e5484d"), colors.HexColor("#e5a00d"), colors.HexColor("#7c5cff")

    def st(name, font="sans", size=10, lead=None, color=INK, **kw):
        return ParagraphStyle(name, fontName=F[font], fontSize=size, leading=lead or size * 1.35, textColor=color, **kw)

    body, small = st("body", size=10, lead=14), st("small", size=8.5, color=MUTE)
    h1 = st("h1", "sansb", 20, 24, colors.white)
    h2 = st("h2", "sansb", 13, 17, INK, spaceBefore=10, spaceAfter=6)
    lab = st("lab", "sansb", 7.5, 10, MUTE)
    mono = st("mono", "mono", 8.5, 11.5)
    W, H = A4
    M = 36
    CW = W - 2 * M

    # ----- page chrome
    def chrome(c, doc):
        c.saveState()
        c.setFillColor(PAGE)
        c.rect(0, 0, W, H, stroke=0, fill=1)
        c.setFillColor(NAVY)
        c.rect(0, 0, W, 24, stroke=0, fill=1)
        c.setFillColor(colors.HexColor("#9fb3c8"))
        c.setFont(F["mono"], 7.5)
        c.drawString(M, 9, f"Content OS  |  {_repo_label(m)}  |  run {m['run_id']}")
        c.drawRightString(W - M, 9, f"page {doc.page}")
        c.restoreState()

    doc = BaseDocTemplate(str(path), pagesize=A4, leftMargin=M, rightMargin=M, topMargin=M, bottomMargin=40,
                          title=f"{m['title']} - LinkedIn kit", author="Content OS")
    doc.addPageTemplates([PageTemplate(id="p", frames=[Frame(M, 40, CW, H - M - 40, 0, 0, 0, 0)], onPage=chrome)])
    S = []

    def band(title, sub):
        t = Table([[Paragraph(f'<font size="8" color="#1fbf8a">CONTENT OS  |  {T(sub)}</font>', st("b0", "monob", 8, 10, colors.white))],
                   [Paragraph(T(title), h1)]], colWidths=[CW])
        t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), NAVY), ("LEFTPADDING", (0, 0), (-1, -1), 16),
                               ("RIGHTPADDING", (0, 0), (-1, -1), 16), ("TOPPADDING", (0, 0), (0, 0), 12),
                               ("BOTTOMPADDING", (0, 1), (0, 1), 14), ("TOPPADDING", (0, 1), (0, 1), 2),
                               ("LINEBELOW", (0, -1), (-1, -1), 3, TEAL)]))
        return t

    def card(rows, pad=14, bg=CARD):
        t = Table([[r] for r in rows], colWidths=[CW])
        t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), bg), ("LEFTPADDING", (0, 0), (-1, -1), pad),
                               ("RIGHTPADDING", (0, 0), (-1, -1), pad), ("TOPPADDING", (0, 0), (-1, -1), 3),
                               ("BOTTOMPADDING", (0, 0), (-1, -1), 3), ("LINEBEFORE", (0, 0), (0, -1), .6, LINE),
                               ("LINEAFTER", (-1, 0), (-1, -1), .6, LINE), ("LINEABOVE", (0, 0), (-1, 0), .6, LINE),
                               ("LINEBELOW", (0, -1), (-1, -1), .6, LINE), ("TOPPADDING", (0, 0), (0, 0), 12),
                               ("BOTTOMPADDING", (0, -1), (-1, -1), 12)]), )
        return t

    def avatar():
        d = Drawing(38, 38)
        d.add(Rect(0, 0, 38, 38, rx=19, ry=19, fillColor=NAVY, strokeColor=None))
        d.add(String(19, 12, "S", fontName=F["sansb"], fontSize=17, fillColor=TEAL, textAnchor="middle"))
        return d

    def img_flow(width):
        if not png:
            return Paragraph(T("(graphic not rendered: pip install pillow)"), small)
        return Image(io.BytesIO(png), width=width, height=width * GFX_H / GFX_W)

    def bar(pct, w=150, color=TEAL, h=8):
        d = Drawing(w, h + 2)
        d.add(Rect(0, 1, w, h, rx=4, ry=4, fillColor=colors.HexColor("#e6eaef"), strokeColor=None))
        d.add(Rect(0, 1, max(w * pct, 4) if pct > 0 else 0, h, rx=4, ry=4, fillColor=color, strokeColor=None))
        return d

    s = m["stats"]
    # ===== PAGE 1: what it will look like
    S.append(band(m["title"], f"{_repo_label(m)}  |  {m['date']}"))
    S.append(Spacer(1, 10))
    S.append(Paragraph(T("This is what your post will look like. Everything you need to publish it is on the next pages, "
                         "and the same text sits in the numbered files next to this PDF."), small))
    S.append(Spacer(1, 8))
    hdr = Table([[avatar(), Paragraph(f'<b>Shaheer</b>  <font color="#5f6f80" size="8">| ShaheerSec</font><br/>'
                                      f'<font color="#5f6f80" size="8">Cybersecurity | ethical hacking | AI security  ·  Just now</font>',
                                      st("who", size=10, lead=13))]],
                colWidths=[46, CW - 28 - 46])
    hdr.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("LEFTPADDING", (0, 0), (-1, -1), 0)]))
    cap = st("cap", size=9.4, lead=12.8)
    paras = [Paragraph(T(para.strip()), cap) for para in re.split(r"\n\s*\n", m["text"])]
    reacts = Table([[Paragraph('<font color="#5f6f80" size="8">Like  ·  Comment  ·  Repost  ·  Send</font>', small)]], colWidths=[CW - 28])
    reacts.setStyle(TableStyle([("LINEABOVE", (0, 0), (-1, 0), .6, LINE), ("TOPPADDING", (0, 0), (-1, -1), 8)]))
    head_h = band(m["title"], "x").wrap(CW, 999)[1] + 60                    # band + the note line above the card
    avail = (H - M - 40) - head_h - 4
    img_w = CW - 28
    while True:                                                              # shrink the image until the card fits page 1
        rows = [hdr, Spacer(1, 4)]
        cap_h = sum(p_.wrap(CW - 28, 9999)[1] + 5 for p_ in paras)
        if cap_h < avail * .75:                        # one cell = no per-row padding, fits more on page 1
            rows.append([x for p_ in paras for x in (p_, Spacer(1, 5))])
        else:                                          # very long post: one row per paragraph so it can split
            for p_ in paras:
                rows += [p_, Spacer(1, 4)]
        if png:
            im = img_flow(img_w)
            box = Table([[im]], colWidths=[CW - 28], hAlign="CENTER")
            box.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER"), ("LEFTPADDING", (0, 0), (-1, -1), 0),
                                     ("RIGHTPADDING", (0, 0), (-1, -1), 0), ("TOPPADDING", (0, 0), (-1, -1), 0),
                                     ("BOTTOMPADDING", (0, 0), (-1, -1), 0)]))
            rows.append(box)
        rows += [Spacer(1, 6), reacts]
        card_t = card(rows)
        if not png or img_w <= 170 or card_t.wrap(CW, 9999)[1] <= avail:
            break
        img_w -= 15
    S.append(card_t)
    S.append(PageBreak())

    # ===== PAGE 2: at a glance + first comment + steps
    S.append(Paragraph("At a glance", h2))
    conf = None
    for a in m["angles"]:
        if a["title"] == m["title"]:
            conf = a.get("score")
    acc = m["scores"].get("accuracy")
    tiles = [(f"{s['chars']}", f"of {LINKEDIN_MAX_CHARS} characters"), (f"{s['words']}", "words"),
             (f"{len(m['claims'])}", "claims, all verified in repo"),
             (f"{round(conf * 100)}%" if conf is not None else "-", "Claude's confidence in the angle"),
             (f"{acc}/5" if acc is not None else "-", "critic accuracy score"),
             (f"{s['hashtags']}", "hashtags")]
    tt = Table([[Paragraph(f'<font name="{F["sansb"]}" size="17" color="#0b1320">{T(n)}</font><br/>'
                           f'<font size="7.5" color="#5f6f80">{T(l)}</font>', st("tile", size=8, lead=12)) for n, l in tiles[:3]],
                [Paragraph(f'<font name="{F["sansb"]}" size="17" color="#0b1320">{T(n)}</font><br/>'
                           f'<font size="7.5" color="#5f6f80">{T(l)}</font>', st("tile", size=8, lead=12)) for n, l in tiles[3:]]],
               colWidths=[CW / 3] * 3)
    tt.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), CARD), ("BOX", (0, 0), (-1, -1), .6, LINE),
                            ("INNERGRID", (0, 0), (-1, -1), .6, LINE), ("TOPPADDING", (0, 0), (-1, -1), 9),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 9), ("LEFTPADDING", (0, 0), (-1, -1), 12)]))
    S.append(tt)
    if m["warnings"]:
        S.append(Spacer(1, 6))
        S.append(Paragraph("<b>Warnings:</b> " + T("; ".join(m["warnings"])), st("w", size=9, color=colors.HexColor("#8a5a00"))))

    S.append(Paragraph("First comment (post it right after publishing)", h2))
    S.append(card([Paragraph(f'<font color="#5f6f80" size="8">Shaheer  ·  your comment</font>', small),
                   Paragraph(T(m["kit"]["first_comment"]), st("fc", size=9.5, lead=13.5))], bg=colors.HexColor("#f8fafb")))
    S.append(Paragraph("Image alt text", h2))
    S.append(card([Paragraph(T(m["kit"]["alt_text"]), st("alt", size=9.5, lead=13.5))], bg=colors.HexColor("#f8fafb")))

    S.append(Paragraph("Post it in 5 steps", h2))
    stp = Table([[Paragraph(f'<font name="{F["sansb"]}" color="#1fbf8a">{i}</font>', st("n", size=12)),
                  Paragraph(T(t), body), Paragraph("[  ]", st("cb", "mono", 10, color=MUTE))]
                 for i, t in enumerate(HOW_TO_POST, 1)], colWidths=[22, CW - 60, 30])
    stp.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), CARD), ("BOX", (0, 0), (-1, -1), .6, LINE),
                             ("LINEBELOW", (0, 0), (-1, -2), .4, LINE), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                             ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                             ("LEFTPADDING", (0, 0), (0, -1), 12)]))
    S.append(stp)
    S.append(PageBreak())

    # ===== PAGE 3: visuals
    S.append(Paragraph("Graphic", h2))
    S.append(img_flow(CW))
    S.append(Paragraph(T("1200 x 627 px. Saved as 03_graphic.png. Prefer a different look? Use a prompt below in Canva or an image AI."), small))
    S.append(Paragraph("Prompts for Canva / image AIs", h2))
    for p in m["image_prompts"]:
        S.append(KeepTogether([card([Paragraph(f'<font name="{F["sansb"]}" size="8" color="#7c5cff">{T((p.get("tool") or "IMAGE AI").upper())}</font>', small),
                       Paragraph(T(p.get("prompt", "")), st("ip", "mono", 8, 11))], bg=colors.HexColor("#f8fafb"))]))
        S.append(Spacer(1, 5))
    if not m["image_prompts"]:
        S.append(Paragraph(T("(this run has no image prompts)"), small))
    S.append(PageBreak())
    S.append(Paragraph("Screenshots to take", h2))
    for x in m["shots"]:
        chip = (f'  <font name="{F["mono"]}" size="7.5" color="#1fbf8a">{T(x["file_path"])}'
                f'{" " + T(x["lines"]) if x.get("lines") else ""}</font>') if x.get("file_path") else ""
        mins = f'  <font size="7.5" color="#5f6f80">~{x["minutes"]} min</font>' if x.get("minutes") else ""
        S.append(KeepTogether([card([Paragraph(f'<font name="{F["mono"]}">[  ]</font>  <b>{T(x.get("shot", ""))}</b>{chip}{mins}', body),
                       Paragraph(T(x.get("how", "")), st("how", size=8.6, lead=12, color=MUTE))])]))
        S.append(Spacer(1, 5))
    if not m["shots"]:
        S.append(Paragraph(T("(none planned)"), small))

    # ===== PAGE 4: evidence
    S.append(Paragraph("Evidence: every claim points at a real file", h2))
    S.append(Paragraph(T(f"Each item was checked against the repo at commit {(m['repo'].get('head_sha') or '')[:12]}. "
                         "A post with an invented file or commit is never packaged."), small))
    S.append(Spacer(1, 6))
    ev = [[Paragraph("CLAIM", lab), Paragraph("FILE", lab), Paragraph("COMMIT", lab)]]
    for c in m["claims"]:
        ev.append([Paragraph(T(c["claim"]), st("c", size=8.8, lead=12)),
                   Paragraph(T(c["source_path"]), st("f", "mono", 8, 10.5, colors.HexColor("#0f8f66"))),
                   Paragraph(T(c["source_ref"]), st("r", "mono", 8, 10.5))])
    et = Table(ev, colWidths=[CW * .50, CW * .32, CW * .18], repeatRows=1)
    et.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), CARD), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef1f4")),
                            ("BOX", (0, 0), (-1, -1), .6, LINE), ("LINEBELOW", (0, 0), (-1, -2), .4, LINE),
                            ("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 5),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
    S.append(et)
    S.append(PageBreak())

    # ===== PAGE 5: how it was made
    S.append(Paragraph("How this post was made", h2))
    cols, gap, bh = 5, 10, 40
    bw = (CW - (cols - 1) * gap) / cols
    rows_n = -(-len(m["pipeline"]) // cols)
    dh = rows_n * (bh + 22) + 4
    dr = Drawing(CW, dh)
    col = {"success": OKC, "skipped": colors.HexColor("#9aa7b5"), "you": VIO, "pending": colors.HexColor("#c8d0d9"),
           "failed": BADC, "waiting": AMB}
    for i, p in enumerate(m["pipeline"]):
        r, cidx = divmod(i, cols)
        x, y = cidx * (bw + gap), dh - (r + 1) * (bh + 22) + 12
        c_ = col.get(p["status"], col["pending"])
        dr.add(Rect(x, y, bw, bh, rx=7, ry=7, fillColor=colors.white, strokeColor=c_, strokeWidth=1.6))
        dr.add(Rect(x, y, 5, bh, fillColor=c_, strokeColor=None))
        dr.add(String(x + 12, y + bh - 15, p["tag"], fontName=F["monob"], fontSize=9, fillColor=INK))
        dr.add(String(x + 12, y + bh - 27, p["name"], fontName=F["sans"], fontSize=8, fillColor=MUTE))
        tm = "you" if p["status"] == "you" else (f"{p['secs']:.1f}s" if p["secs"] is not None else p["status"])
        dr.add(String(x + bw - 8, y + 9, tm, fontName=F["mono"], fontSize=7.5, fillColor=c_, textAnchor="end"))
        if cidx < cols - 1 and i < len(m["pipeline"]) - 1:
            dr.add(Line(x + bw + 2, y + bh / 2, x + bw + gap - 2, y + bh / 2, strokeColor=colors.HexColor("#9aa7b5"), strokeWidth=1))
    S.append(dr)
    if m["total_secs"]:
        S.append(Paragraph(T(f"Agent time: {m['total_secs']:.0f}s. Purple stages are your decisions (pick the angle, approve the post)."), small))

    if m["angles"]:
        S.append(Paragraph("Angles Claude considered (confidence)", h2))
        arows = []
        for a in m["angles"][:6]:
            chosen = a["title"] == m["title"]
            arows.append([Paragraph(("<b>" if chosen else "") + T(a["title"]) + ("</b>  <font color='#1fbf8a' size='7.5'>CHOSEN</font>" if chosen else ""),
                                    st("at", size=8.8, lead=12)),
                          bar(a.get("score", 0), 90, TEAL if chosen else colors.HexColor("#9aa7b5")),
                          Paragraph(f"{round(a.get('score', 0) * 100)}%", st("pc", "monob", 8.5))])
        at = Table(arows, colWidths=[CW - 150, 100, 40])
        at.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), CARD), ("BOX", (0, 0), (-1, -1), .6, LINE),
                                ("LINEBELOW", (0, 0), (-1, -2), .4, LINE), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                                ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
        S.append(at)
        if m.get("recommendation"):
            S.append(Spacer(1, 4))
            S.append(Paragraph("<b>Claude recommended:</b> " + T(m["recommendation"].get("reason") or m["recommendation"].get("title", "")), small))

    if m["scores"]:
        S.append(Paragraph("Critic scores", h2))
        srows = [[Paragraph(T(k), body), bar(v / 5, 140, OKC if v >= 4 else AMB), Paragraph(f"{v}/5", st("sc", "monob", 8.5))]
                 for k, v in m["scores"].items()]
        sc_t = Table(srows, colWidths=[80, 150, 40], hAlign="LEFT")
        sc_t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 3)]))
        S.append(sc_t)
    fixed = m["changes"]
    if m["flags"]:
        S.append(Paragraph("Issues the critic found, and what the reviser did", h2))
        for f in m["flags"]:
            sev = f.get("severity", "")
            c_ = BADC if sev == "block" else AMB
            S.append(KeepTogether([card([Paragraph(f'<font name="{F["monob"]}" size="7.5" color="{c_.hexval().replace("0x", "#")}">{T(sev.upper())}</font>  '
                                     f'<font name="{F["mono"]}" size="8">{T(f.get("quote", ""))}</font>', small),
                           Paragraph(T(f.get("issue", "")), st("iss", size=8.6, lead=12, color=MUTE))], pad=10)]))
            S.append(Spacer(1, 4))
        S.append(Paragraph(T(f"{len(fixed)} change(s) applied, {len(m['unresolved'])} unresolved."), small))

    doc.build(S)


# ---------------------------------------------------------------- orchestration

def build_folder(ctx: dict, folder, records: dict | None = None, png: bytes | None = None,
                 date: str | None = None) -> dict:
    """Write the whole run folder. `png` = a graphic drawn elsewhere (the dashboard canvas)."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    m = collect(ctx, records, date)
    notes: list[str] = []
    source = "dashboard" if png else "server"
    if png is None:
        keep = folder / "03_graphic.png"
        try:
            png = render_graphic(m)
        except Exception as e:                                  # Pillow missing or font trouble
            if keep.exists():
                png = keep.read_bytes()
                source = "kept"
            else:
                notes.append(f"03_graphic.png skipped ({type(e).__name__}: {e}). Run: pip install pillow")
    if png:
        (folder / "03_graphic.png").write_bytes(png)

    for name, text in _md_files(m, notes).items():
        (folder / name).write_text(text, encoding="utf-8")
    (folder / "copy-paste.html").write_text(_html(m, png), encoding="utf-8")
    try:
        _pdf(m, png, folder / "00_REPORT.pdf")
    except Exception as e:
        notes.append(f"00_REPORT.pdf skipped ({type(e).__name__}: {e}). Run: pip install reportlab")
        (folder / "00_START_HERE.md").write_text(
            (folder / "00_START_HERE.md").read_text(encoding="utf-8") + "\n" + "\n".join(f"- {n}" for n in notes) + "\n",
            encoding="utf-8")

    manifest = {"repo": _repo_label(m), "run_id": m["run_id"], "date": m["date"], "title": m["title"],
                "graphic_source": source, "written": datetime.now().isoformat(timespec="seconds"),
                "files": sorted(p.name for p in folder.iterdir() if p.name != "manifest.json"), "skipped": notes}
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"folder": folder.as_posix(), "pdf": (folder / "00_REPORT.pdf").as_posix(),
            "graphic_source": source, "skipped": notes, "kit": m["kit"]}
