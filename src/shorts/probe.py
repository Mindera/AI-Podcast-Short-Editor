"""Step 1: ffprobe, 16 kHz mono WAV, chunk plan, stereo-mic check, layout contact sheet."""

from __future__ import annotations

import subprocess
import wave
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .cache import atomic_write_json, file_sig, is_fresh, make_key, mark_done, read_json
from .episode import Episode
from .faces import FaceDetector, analysis_size
from .util import ffprobe_json, fmt_ts, log, run_ffmpeg, warn

SR = 16000


def video_info(probe: dict) -> dict:
    v = next(s for s in probe["streams"] if s["codec_type"] == "video")
    num, den = (v.get("avg_frame_rate") or v.get("r_frame_rate") or "30/1").split("/")
    fps = float(num) / float(den) if float(den) else 30.0
    a = next((s for s in probe["streams"] if s["codec_type"] == "audio"), None)
    return {
        "width": int(v["width"]),
        "height": int(v["height"]),
        "fps": fps,
        "duration": float(probe["format"]["duration"]),
        "audio_channels": int(a["channels"]) if a else 0,
    }


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path)) as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)


def plan_chunks(audio: np.ndarray, duration: float, over_s: float, every_s: float) -> list[tuple[float, float]]:
    """Cut at the quietest 0.5 s within ±60 s of each 30-min mark, only if the file is over 1 h."""
    if duration <= over_s:
        return [(0.0, duration)]
    x = audio.astype(np.float32)
    win, hop = int(0.5 * SR), int(0.1 * SR)
    cuts = []
    mark = every_s
    while mark < duration - 60:
        lo, hi = int(max(0, mark - 60) * SR), int(min(duration, mark + 60) * SR)
        seg = x[lo:hi]
        n = max(1, (len(seg) - win) // hop)
        energy = [float(np.mean(seg[i * hop:i * hop + win] ** 2)) for i in range(n)]
        best = int(np.argmin(energy))
        cuts.append((lo + best * hop + win // 2) / SR)
        mark += every_s
    edges = [0.0, *cuts, duration]
    return list(zip(edges[:-1], edges[1:]))


def stereo_check(src: Path, work: Path, channels: int) -> dict:
    """If L/R correlate below ~0.9 they're probably separate mics: a free speaker signal."""
    if channels < 2:
        return {"separate_mics": False, "correlation": None, "reason": "mono audio"}
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(src), "-map", "0:a:0", "-vn", "-ac", "2", "-ar", "8000", "-f", "s16le", "-"],
        capture_output=True, check=True,
    ).stdout
    lr = np.frombuffer(raw, dtype=np.int16).reshape(-1, 2).astype(np.float32)
    if len(lr) < 8000 or lr.std(axis=0).min() < 1:
        return {"separate_mics": False, "correlation": None, "reason": "silent channel"}
    corr = float(np.corrcoef(lr[:, 0], lr[:, 1])[0, 1])
    hop = 800  # 10 Hz RMS envelope
    n = len(lr) // hop
    rms = np.sqrt((lr[: n * hop].reshape(n, hop, 2) ** 2).mean(axis=1))
    np.save(work / "stereo_rms.npy", rms.astype(np.float32))
    return {"separate_mics": corr < 0.9, "correlation": round(corr, 3), "rms_hz": 10}


def layout_sheet(src: Path, info: dict, work: Path) -> dict:
    """Sample 10 frames across the episode, detect faces, and build a labelled contact sheet."""
    dur, n = info["duration"], 10
    aw, ah = analysis_size(info["width"], info["height"])
    det = FaceDetector(aw, ah)
    tiles, frames = [], []
    for i in range(n):
        t = dur * (i + 0.5) / n
        raw = subprocess.run(
            ["ffmpeg", "-v", "error", "-ss", f"{t:.2f}", "-i", str(src), "-frames:v", "1",
             "-vf", f"scale={aw}:{ah}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
            capture_output=True,
        ).stdout
        if len(raw) < aw * ah * 3:
            continue
        rgb = np.frombuffer(raw[: aw * ah * 3], dtype=np.uint8).reshape(ah, aw, 3)
        faces = det(np.ascontiguousarray(rgb[:, :, ::-1]))
        img = Image.fromarray(rgb)
        d = ImageDraw.Draw(img)
        for f in faces:
            d.rectangle([f.x, f.y, f.x + f.w, f.y + f.h], outline=(255, 199, 0), width=3)
        d.rectangle([0, 0, 200, 34], fill=(10, 10, 11))
        d.text((8, 6), f"{fmt_ts(t)}  faces={len(faces)}", fill=(255, 255, 255), font_size=20)
        tiles.append(img.resize((aw // 2, ah // 2)))
        frames.append({"t": round(t, 2), "faces": [{"cx": round(f.cx / aw, 3), "cy": round(f.cy / ah, 3),
                                                    "w": round(f.w / aw, 3)} for f in faces]})
    if tiles:
        tw, th = tiles[0].size
        sheet = Image.new("RGB", (tw * 5, th * ((len(tiles) + 4) // 5)), (10, 10, 11))
        for i, tile in enumerate(tiles):
            sheet.paste(tile, ((i % 5) * tw, (i // 5) * th))
        sheet.save(work / "layout_sheet.jpg", quality=88)
    counts = [len(f["faces"]) for f in frames]
    summary = {
        "frames": frames,
        "max_faces": max(counts, default=0),
        "two_shot_ratio": round(sum(c >= 2 for c in counts) / max(1, len(counts)), 2),
        "single_ratio": round(sum(c == 1 for c in counts) / max(1, len(counts)), 2),
    }
    atomic_write_json(work / "layout.json", summary)
    return summary


def run(ep: Episode, force: bool = False) -> dict:
    work, src = ep.work, ep.source_file
    outputs = [work / "probe.json", work / "audio16k.wav", work / "chunks.json", work / "stereo.json", work / "layout.json"]
    key = make_key("probe-v1", file_sig(src), ep.cfg.chunk_over_s, ep.cfg.chunk_every_s)
    if not force and is_fresh(work, "probe", key, outputs):
        return video_info(read_json(work / "probe.json"))

    log("probe: ffprobe")
    probe = ffprobe_json(src)
    atomic_write_json(work / "probe.json", probe)
    info = video_info(probe)
    log(f"  {info['width']}x{info['height']} @ {info['fps']:.2f} fps, {fmt_ts(info['duration'])}, "
        f"{info['audio_channels']} audio ch")
    if info["audio_channels"] == 0:
        raise SystemExit("source has no audio stream")

    log("probe: extracting 16 kHz mono WAV")
    wav = work / "audio16k.wav"
    tmp = work / "audio16k.part.wav"
    run_ffmpeg(["-i", str(src), "-map", "0:a:0", "-vn", "-ac", "1", "-ar", str(SR), "-c:a", "pcm_s16le", str(tmp)],
               work / "logs" / "probe_wav.log", ep.cfg.stall_s, "wav extract")
    tmp.replace(wav)

    chunks = plan_chunks(read_wav(wav), info["duration"], ep.cfg.chunk_over_s, ep.cfg.chunk_every_s)
    atomic_write_json(work / "chunks.json", chunks)
    if len(chunks) > 1:
        log(f"  {len(chunks)} transcription chunks, cuts at " + ", ".join(fmt_ts(c[0]) for c in chunks[1:]))

    log("probe: stereo mic check")
    stereo = stereo_check(src, work, info["audio_channels"])
    atomic_write_json(work / "stereo.json", stereo)
    if stereo["separate_mics"]:
        log(f"  L/R correlation {stereo['correlation']}: separate mics, using as a speaker hint")

    log("probe: layout contact sheet (10 frames)")
    lay = layout_sheet(src, info, work)
    log(f"  max faces {lay['max_faces']}, two-shot {lay['two_shot_ratio']:.0%}, single {lay['single_ratio']:.0%} "
        f"→ {work / 'layout_sheet.jpg'}")
    if lay["max_faces"] == 0:
        warn("no faces found in the layout sample; clips will use a centre crop where faces are missing")

    mark_done(work, "probe", key)
    return info
