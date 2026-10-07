"""Step 7: technical checks, QA frames + contact sheet, caption safety checks, README.md."""

from __future__ import annotations

import json
import shutil
import struct
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw

from .cache import atomic_write_text, read_json
from .episode import Episode
from .reframe import OUT_H, OUT_W
from .util import console, ffprobe_json, fmt_ts, log, warn

TILE_W, TILE_H = 270, 480


def top_level_atoms(path: Path) -> list[str]:
    atoms = []
    size_total = path.stat().st_size
    with open(path, "rb") as f:
        pos = 0
        while pos < size_total:
            f.seek(pos)
            head = f.read(8)
            if len(head) < 8:
                break
            size, kind = struct.unpack(">I4s", head)
            if size == 1:
                size = struct.unpack(">Q", f.read(8))[0]
            elif size == 0:
                size = size_total - pos
            atoms.append(kind.decode("latin-1"))
            if size < 8:
                break
            pos += size
    return atoms


def check_file(mp4: Path, expected_dur: float) -> list[str]:
    problems = []
    p = ffprobe_json(mp4)
    v = next((s for s in p["streams"] if s["codec_type"] == "video"), None)
    a = next((s for s in p["streams"] if s["codec_type"] == "audio"), None)
    if not v:
        return ["no video stream"]
    if (v["width"], v["height"]) != (OUT_W, OUT_H):
        problems.append(f"size {v['width']}x{v['height']}")
    if v.get("profile") != "High":
        problems.append(f"profile {v.get('profile')}")
    if v.get("pix_fmt") != "yuv420p":
        problems.append(f"pix_fmt {v.get('pix_fmt')}")
    if not a or a.get("codec_name") != "aac" or a.get("sample_rate") != "48000":
        problems.append(f"audio {a and a.get('codec_name')} {a and a.get('sample_rate')}")
    dur = float(p["format"]["duration"])
    if abs(dur - expected_dur) > 0.25:
        problems.append(f"duration {dur:.2f}s vs {expected_dur:.2f}s")
    atoms = top_level_atoms(mp4)
    if "moov" in atoms and "mdat" in atoms and atoms.index("moov") > atoms.index("mdat"):
        problems.append("moov after mdat (no faststart)")
    return problems


def caption_problems(cw: Path, brand) -> list[str]:
    meta_p, rf_p = cw / "captions" / "captions.json", cw / "reframe.json"
    if not meta_p.exists() or not rf_p.exists():
        return ["caption/reframe metadata missing"]
    segs = read_json(rf_p)["segments"]
    out = []
    for p in read_json(meta_p)["pages"]:
        x1, y1, x2, y2 = p["bbox"]
        if y1 < brand.safe_top or y2 > OUT_H - brand.safe_bottom or x2 > OUT_W - brand.safe_right + brand.outline_px:
            out.append(f"caption outside safe zone at {p['start']:.1f}s: “{p['text']}”")
        for s in segs:
            if s["start"] < p["end"] and s["end"] > p["start"]:
                for fx, fy, fw, fh in s.get("faces", []):
                    if x1 < fx + fw and fx < x2 and y1 < fy + fh and fy < y2:
                        out.append(f"caption over a face at {p['start']:.1f}s: “{p['text']}”")
                        break
    return sorted(set(out))


def grab(mp4: Path, t: float, dest: Path) -> None:
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{t:.2f}", "-i", str(mp4), "-frames:v", "1", "-q:v", "3",
                    str(dest)], check=True)


def contact_sheet(rows: list[tuple[str, list[Path]]], dest: Path) -> None:
    cols = 2  # clips per row, 3 frames each
    label_h = 36
    n_rows = (len(rows) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * 3 * TILE_W + (cols - 1) * 20, n_rows * (TILE_H + label_h)), (10, 10, 11))
    d = ImageDraw.Draw(sheet)
    for i, (label, frames) in enumerate(rows):
        ox = (i % cols) * (3 * TILE_W + 20)
        oy = (i // cols) * (TILE_H + label_h)
        d.text((ox + 6, oy + 8), label[:70], fill=(255, 199, 0), font_size=20)
        for k, f in enumerate(frames):
            if f.exists():
                sheet.paste(Image.open(f).convert("RGB").resize((TILE_W, TILE_H)), (ox + k * TILE_W, oy + label_h))
    sheet.save(dest, quality=88)


def ai_review(ep: Episode, sheet: Path) -> str:
    cfg = ep.cfg
    if not shutil.which(cfg.claude_bin):
        return "claude CLI not found; skipped"
    prompt = (f"Use the Read tool to look at {sheet}. Each block is one vertical short: frames at 15%, 50% and 85%, "
              f"titled with the clip number. Check for each clip: is a person well framed (not cut off, not an empty "
              f"crop)? Are captions readable, inside the frame, and not covering faces? Anything visually broken? "
              f"Answer with a short markdown list per clip: OK or the problem.")
    cmd = [cfg.claude_bin, "-p", "--allowedTools", "Read", "--add-dir", str(sheet.parent)]
    if cfg.claude_model:
        cmd += ["--model", cfg.claude_model]
    proc = subprocess.run(cmd, input=prompt, capture_output=True, text=True, cwd=sheet.parent, timeout=600)
    return proc.stdout.strip() or proc.stderr.strip()


def write_readme(ep: Episode, data: dict, results: dict[int, dict]) -> None:
    lines = [f"# Shorts — {ep.dir.name}", "", f"Source: `{ep.source}`  ", f"Host: {ep.cfg.host} · Guest: {ep.cfg.guest}",
             "", "| # | Clip | Cover | Start | Length | Score | Title | QA |", "|---|---|---|---|---|---|---|---|"]
    for c in data["clips"]:
        r = results.get(c["rank"], {})
        qa = "✅" if not r.get("problems") else "⚠️ " + "; ".join(r["problems"][:2])
        lines.append(f"| {c['rank']} | [{c['slug']}.mp4]({c['slug']}.mp4) | [jpg]({c['slug']}.jpg) | "
                     f"{fmt_ts(c['start'])} | {c['duration']:.0f}s | {c['virality_score']:.1f} | {c['title']} | {qa} |")
    lines += ["", "## Copy-paste", ""]
    for c in data["clips"]:
        lines += [f"### {c['rank']:02d} — {c['title']}", "",
                  f"*Hook:* {c['hook_text']}  ", f"*Why it works:* {c['why_it_works']}", "",
                  "```text", c["title"], "", c["description"], "", " ".join(c["hashtags"]), "```", ""]
    if data.get("runners_up"):
        lines += ["## Runners-up", ""]
        for r in data["runners_up"]:
            lines.append(f"- {fmt_ts(r['start'])} ({r['duration']:.0f}s, score {r['virality_score']:.1f}): {r['title']}")
        lines.append("")
    lines += ["## Caption corrections", ""] + [f"- {k} → {v}" for k, v in data["corrections"].items()] + [""]
    atomic_write_text(ep.dir / "README.md", "\n".join(lines))


def run(ep: Episode, ai: bool = False) -> bool:
    data = read_json(ep.work / "clips.json")
    qa_dir = ep.dir / "_qa"
    qa_dir.mkdir(exist_ok=True)
    results: dict[int, dict] = {}
    rows = []
    all_ok = True
    for c in data["clips"]:
        mp4 = ep.dir / f"{c['slug']}.mp4"
        cw = ep.work / "clips" / f"{c['rank']:02d}"
        if not mp4.exists():
            results[c["rank"]] = {"problems": ["not rendered"]}
            all_ok = False
            continue
        rf = read_json(cw / "reframe.json")
        expected = rf["segments"][-1]["end"]
        problems = check_file(mp4, expected) + caption_problems(cw, ep.cfg.brand)
        frames = []
        for pct in (15, 50, 85):
            f = qa_dir / f"{c['slug']}_{pct}.jpg"
            grab(mp4, expected * pct / 100, f)
            frames.append(f)
        kinds = sorted({s["kind"] for s in rf["segments"]})
        rows.append((f"{c['rank']:02d} {c['title']} [{len(rf['segments'])} seg, {'/'.join(kinds)}]", frames))
        results[c["rank"]] = {"problems": problems}
        if problems:
            all_ok = False
            warn(f"QA {c['slug']}: " + "; ".join(problems))
        else:
            log(f"QA {c['slug']}: ok")
    sheet = qa_dir / "contact_sheet.jpg"
    if rows:
        contact_sheet(rows, sheet)
        log(f"QA contact sheet → {sheet}")
    if ai and rows:
        log("QA: asking Claude to review the contact sheet…")
        review = ai_review(ep, sheet)
        atomic_write_text(qa_dir / "ai_review.md", review + "\n")
        console.print(review)
    write_readme(ep, data, results)
    atomic_write_text(qa_dir / "qa.json", json.dumps(results, indent=2))
    log(f"README → {ep.dir / 'README.md'}")
    return all_ok
