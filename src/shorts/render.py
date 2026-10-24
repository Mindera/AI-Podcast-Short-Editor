"""Step 6: reframe + captions + one ffmpeg encode per clip, plus the cover. Clips render in parallel."""

from __future__ import annotations

import json
import os
import re
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .assets import ensure_assets, font_path
from .cache import atomic_write_json, file_sig, is_fresh, make_key, mark_done, part_path, read_json, supersede
from .captions import build_captions, clip_words
from .episode import Episode
from .probe import video_info
from .reframe import OUT_H, OUT_W, reframe_clip, snap_to_frames
from .util import cpu_count, hex_rgb, log, run_ffmpeg, warn
from .words import transcript_words

LOUDNORM = "loudnorm=I=-14:TP=-1.5:LRA=11"


def out_fps(info: dict, max_fps: int) -> float:
    return min(info["fps"], float(max_fps)) if info["fps"] > 1 else float(max_fps)


def seg_chain(seg: dict, tag: str) -> str:
    """Filter chain turning one segment's source frames into a 1080×1920 frame."""
    if seg["kind"] == "split":
        (ax, ay, aw, ah), (bx, by, bw, bh) = seg["rects"]
        return (f"split[{tag}a][{tag}b];"
                f"[{tag}a]crop={aw}:{ah}:{ax}:{ay},scale={OUT_W}:{OUT_H // 2}:flags=lanczos[{tag}ta];"
                f"[{tag}b]crop={bw}:{bh}:{bx}:{by},scale={OUT_W}:{OUT_H // 2}:flags=lanczos[{tag}tb];"
                f"[{tag}ta][{tag}tb]vstack,setsar=1,format=yuv420p")
    x, y, w, h = seg["rects"][0]
    return f"crop={w}:{h}:{x}:{y},scale={OUT_W}:{OUT_H}:flags=lanczos,setsar=1,format=yuv420p"


def measure_loudness(src: Path, start: float, dur: float, log_path: Path, stall_s: int) -> dict | None:
    text = run_ffmpeg(["-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", str(src), "-map", "0:a:0", "-vn",
                       "-af", f"{LOUDNORM}:print_format=json", "-f", "null", "-"], log_path, stall_s, "loudnorm pass 1")
    m = re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", text, re.S)
    if not m:
        return None
    data = json.loads(m.group(0))
    if "inf" in str(data.get("input_i")):
        return None
    return data


def encode(src: Path, clip: dict, segs: list[dict], caps: Path, fps: float, loud: dict | None, out: Path,
           log_path: Path, crf: int, stall_s: int) -> None:
    start, dur = clip["start"], segs[-1]["end"]
    args: list[str] = []
    for s in segs:
        # a little extra input so trim=end_frame always has enough frames
        args += ["-ss", f"{start + s['start']:.4f}", "-t", f"{s['end'] - s['start'] + 3 / fps:.4f}", "-i", str(src)]
    n = len(segs)
    args += ["-ss", f"{start:.4f}", "-t", f"{dur:.4f}", "-i", str(src)]
    args += ["-f", "concat", "-safe", "0", "-i", str(caps)]

    parts = []
    for i, s in enumerate(segs):
        parts.append(f"[{i}:v:0]fps={fps},trim=end_frame={s['frames']},setpts=PTS-STARTPTS,{seg_chain(s, f's{i}')}[v{i}]")
    parts.append("".join(f"[v{i}]" for i in range(n)) + f"concat=n={n}:v=1:a=0[base]")
    parts.append(f"[{n + 1}:v]format=rgba,fps={fps}[cap]")
    parts.append("[base][cap]overlay=0:0:eof_action=repeat:format=auto,format=yuv420p[v]")
    if loud:
        an = (f"{LOUDNORM}:measured_I={loud['input_i']}:measured_TP={loud['input_tp']}:"
              f"measured_LRA={loud['input_lra']}:measured_thresh={loud['input_thresh']}:"
              f"offset={loud['target_offset']}:linear=true")
    else:
        an = "anull"
    parts.append(f"[{n}:a:0]{an},aresample=48000[a]")

    args += ["-filter_complex", ";".join(parts), "-map", "[v]", "-map", "[a]", "-t", f"{dur:.4f}",
             "-c:v", "libx264", "-profile:v", "high", "-preset", "medium", "-crf", str(crf), "-pix_fmt", "yuv420p",
             "-r", f"{fps}", "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-movflags", "+faststart",
             "-f", "mp4", str(out)]
    run_ffmpeg(args, log_path, stall_s, f"encode {out.name}")


def _wrap_title(title: str, font: ImageFont.FreeTypeFont, max_w: float) -> list[list[str]]:
    lines: list[list[str]] = [[]]
    for w in title.split():
        if lines[-1] and font.getlength(" ".join(lines[-1] + [w])) > max_w:
            lines.append([w])
        else:
            lines[-1].append(w)
    return lines


def make_cover(src: Path, clip: dict, segs: list[dict], font_file: Path, brand, out: Path) -> None:
    t_rel = min(max(0.0, clip["suggested_cover_timestamp"] - clip["start"]), segs[-1]["end"] - 0.05)
    seg = next((s for s in segs if s["start"] <= t_rel < s["end"]), segs[-1])
    chain = seg_chain(seg, "c")
    frame = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", f"{clip['start'] + t_rel:.3f}", "-i", str(src), "-frames:v", "1",
         "-filter_complex", f"[0:v]{chain},format=rgb24[o]", "-map", "[o]", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True, check=True,
    ).stdout
    img = Image.frombytes("RGB", (OUT_W, OUT_H), frame[: OUT_W * OUT_H * 3])
    dark = Image.new("RGB", img.size, hex_rgb(brand.outline_color))
    img = Image.blend(img, dark, brand.cover_darken)
    # extra darkening toward the bottom so the title reads
    grad = Image.linear_gradient("L").resize(img.size)
    img = Image.composite(dark, img, grad.point(lambda v: int(v * 0.5)))

    size = brand.cover_font_size
    max_w = brand.max_text_width
    while True:
        font = ImageFont.truetype(str(font_file), size)
        lines = _wrap_title(clip["title"], font, max_w)
        if len(lines) <= 3 or size <= 56:
            break
        size -= 8
    ascent, descent = font.getmetrics()
    lh = (ascent + descent) * 1.1
    block = lh * len(lines)
    cy = min(brand.cover_title_y * OUT_H, OUT_H - brand.safe_bottom - block / 2)
    top = cy - block / 2
    key = re.sub(r"[^\w]", "", clip.get("key_word", "").lower())
    d = ImageDraw.Draw(img)
    space = font.getlength(" ")
    for li, line in enumerate(lines):
        lw = font.getlength(" ".join(line))
        x = OUT_W / 2 - lw / 2
        for w in line:
            color = brand.highlight_color if key and re.sub(r"[^\w]", "", w.lower()) == key else brand.text_color
            d.text((x, top + li * lh + ascent), w, font=font, anchor="ls", fill=hex_rgb(color),
                   stroke_width=brand.outline_px // 2, stroke_fill=hex_rgb(brand.outline_color))
            x += font.getlength(w) + space
    tmp = part_path(out)
    img.save(tmp, format="JPEG", quality=92)
    os.replace(tmp, out)


def render_one(ep_dir: str, clip: dict, host_side: str, corrections: dict) -> dict:
    """Runs in a worker process. Returns a small status dict."""
    ep = Episode.load(Path(ep_dir))
    cfg, work, src = ep.cfg, ep.work, ep.source_file
    info = video_info(read_json(work / "probe.json"))
    fps = out_fps(info, cfg.max_fps)
    cw = ep.clip_work(clip["rank"])
    font_file = font_path(cfg.brand.font)
    mp4, jpg, ass = ep.dir / f"{clip['slug']}.mp4", ep.dir / f"{clip['slug']}.jpg", ep.dir / f"{clip['slug']}.ass"

    # reframe (cached)
    rf_path = cw / "reframe.json"
    rf_key = make_key("reframe-v1", file_sig(src), clip["start"], clip["end"], clip["speakers"], host_side,
                      cfg.analysis_fps, cfg.min_segment_s, cfg.rapid_turn_s, cfg.split_screen, fps)
    if is_fresh(cw, "reframe", rf_key, [rf_path]):
        rf = read_json(rf_path)
    else:
        rf = reframe_clip(src, info, clip["start"], clip["end"], clip["speakers"], host_side, cfg)
        rf["segments"] = snap_to_frames(rf["segments"], fps, clip["end"] - clip["start"])
        atomic_write_json(rf_path, rf)
        mark_done(cw, "reframe", rf_key)
    segs = rf["segments"]
    dur = segs[-1]["end"]

    # captions (cached)
    caps_dir = cw / "captions"
    cap_key = make_key("captions-v4", file_sig(work / "transcript.json"), clip["start"], clip["end"], corrections,
                       cfg.terms, cfg.fillers, cfg.brand.model_dump(), str(font_file), rf_key)
    if not is_fresh(cw, "captions", cap_key, [caps_dir / "caps.ffconcat", ass]):
        words = clip_words(transcript_words(read_json(work / "transcript.json")), clip["start"], clip["end"], cfg,
                           corrections)
        build_captions(words, dur, segs, font_file, cfg.brand, caps_dir, ass)
        mark_done(cw, "captions", cap_key)

    # encode + cover (cached)
    enc_key = make_key("encode-v1", rf_key, cap_key, cfg.crf, fps, clip["title"], clip.get("key_word"),
                       clip["suggested_cover_timestamp"])
    if is_fresh(cw, "encode", enc_key, [mp4, jpg]):
        return {"rank": clip["rank"], "file": mp4.name, "status": "cached"}
    for old in (mp4, jpg):
        supersede(old, work)
    loud = measure_loudness(src, clip["start"], dur, cw / "loudnorm.log", cfg.stall_s)
    tmp = part_path(mp4)
    encode(src, clip, segs, caps_dir / "caps.ffconcat", fps, loud, tmp, cw / "encode.log", cfg.crf, cfg.stall_s)
    os.replace(tmp, mp4)
    make_cover(src, clip, segs, font_file, cfg.brand, jpg)
    mark_done(cw, "encode", enc_key)
    kinds = {s["kind"] for s in segs}
    return {"rank": clip["rank"], "file": mp4.name, "status": "rendered", "segments": len(segs),
            "split": "split" in kinds}


def run(ep: Episode, only: list[int] | None = None) -> list[dict]:
    clips_path = ep.work / "clips.json"
    if not clips_path.exists():
        raise SystemExit(f"no clips picked yet ({clips_path} is missing) — run `shorts run <source>` first")
    data = read_json(clips_path)
    clips = [c for c in data["clips"] if not only or c["rank"] in only]
    if only and not clips:
        raise SystemExit(f"--only {','.join(map(str, only))} matched no clips "
                         f"(clips.json has ranks 1–{len(data['clips'])})")
    ensure_assets(ep.cfg.brand.font)

    # move outputs of clips that are no longer in clips.json out of the way (never delete)
    current = {c["slug"] for c in data["clips"]}
    for f in ep.dir.glob("[0-9][0-9]_*.*"):
        if f.suffix in {".mp4", ".jpg", ".ass"} and f.stem not in current:
            supersede(f, ep.work)

    jobs = ep.cfg.jobs or max(1, cpu_count() // 6)
    log(f"render: {len(clips)} clips, {jobs} parallel job(s)")
    results = []
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        futs = {pool.submit(render_one, str(ep.dir), c, data["host_side"], data["corrections"]): c for c in clips}
        for fut in as_completed(futs):
            c = futs[fut]
            try:
                r = fut.result()
                log(f"  [{c['rank']:02d}] {r['status']}: {r['file']}")
                results.append(r)
            except Exception as e:  # keep going; report at the end
                warn(f"  [{c['rank']:02d}] FAILED: {e}")
                results.append({"rank": c["rank"], "file": f"{c['slug']}.mp4", "status": "failed", "error": str(e)})
    return sorted(results, key=lambda r: r["rank"])
