"""Step 4: decide a fixed 9:16 crop (or a 2-panel split) per segment of a clip.

Signals, in priority order: the speaker turns from clips.json, then mouth motion as a fallback.
Crops never pan within a segment; segments shorter than 0.6 s (shot-detection artefacts) are merged away.
"""

from __future__ import annotations

import subprocess
from collections.abc import Hashable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .faces import Face, FaceDetector, analysis_size, mouth_motion

OUT_W, OUT_H = 1080, 1920


@dataclass
class Run:
    start: float
    end: float
    label: Hashable

    @property
    def dur(self) -> float:
        return self.end - self.start


def coalesce(runs: list[Run]) -> list[Run]:
    out: list[Run] = []
    for r in runs:
        if out and out[-1].label == r.label:
            out[-1] = Run(out[-1].start, r.end, r.label)
        else:
            out.append(Run(r.start, r.end, r.label))
    return out


def merge_short_runs(runs: list[Run], min_len: float) -> list[Run]:
    """Merge runs shorter than `min_len` into a neighbour.

    Terminates: every pass removes at least one run (the shortest short run is absorbed by a neighbour,
    and coalescing can only remove more), and the loop stops at one run or when no run is short.
    """
    runs = coalesce(runs)
    while len(runs) > 1:
        short = [i for i, r in enumerate(runs) if r.dur < min_len - 1e-9]
        if not short:
            break
        i = min(short, key=lambda k: runs[k].dur)
        if i == 0:
            j = 1
        elif i == len(runs) - 1:
            j = i - 1
        else:
            j = i - 1 if runs[i - 1].dur >= runs[i + 1].dur else i + 1
        a, b = min(i, j), max(i, j)
        merged = Run(runs[a].start, runs[b].end, runs[j].label)
        before = len(runs)
        runs = coalesce(runs[:a] + [merged] + runs[b + 1:])
        assert len(runs) < before
    return runs


def _even(v: float) -> int:
    return max(2, int(round(v / 2)) * 2)


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


@dataclass
class Track:
    cx: float  # source pixels
    cy: float
    w: float
    h: float
    frames: dict[int, Face]  # analysis-frame faces by frame index

    def box(self) -> tuple[float, float, float, float]:
        return self.cx - self.w / 2, self.cy - self.h / 2, self.w, self.h


def full_rect(cx: float | None, cy: float | None, W: int, H: int) -> tuple[int, int, int, int]:
    cw = _even(min(W, H * 9 / 16))
    ch = _even(min(H, cw * 16 / 9))
    if cx is None:
        return (_even((W - cw) / 2) if W > cw else 0, _even((H - ch) / 2) if H > ch else 0, cw, ch)
    x = _clamp(cx - cw / 2, 0, W - cw)
    y = _clamp(cy - ch * 0.4, 0, H - ch)
    return _even(x) if x > 0 else 0, _even(y) if y > 0 else 0, cw, ch


def panel_rect(t: Track, W: int, H: int) -> tuple[int, int, int, int]:
    """A 9:8 crop for one 1080×960 panel of a split screen."""
    ph = _clamp(t.h * 4.0, 0.45 * H, H)
    pw = ph * 9 / 8
    if pw > W:
        pw, ph = W, W * 8 / 9
    pw, ph = _even(pw), _even(ph)
    x = _clamp(t.cx - pw / 2, 0, W - pw)
    y = _clamp(t.cy - 0.42 * ph, 0, H - ph)
    return _even(x) if x > 0 else 0, _even(y) if y > 0 else 0, min(pw, W), min(ph, H)


def read_frames(src: Path, start: float, dur: float, aw: int, ah: int, fps: int):
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", str(src),
           "-vf", f"fps={fps},scale={aw}:{ah}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    size = aw * ah * 3
    assert p.stdout is not None
    while True:
        buf = p.stdout.read(size)
        if len(buf) < size:
            break
        yield np.frombuffer(buf, dtype=np.uint8).reshape(ah, aw, 3)
    p.stdout.close()
    p.wait()


def detect_shots(thumbs: list[np.ndarray]) -> list[tuple[int, int]]:
    """Shot cuts from the mean abs difference of 64×36 grayscale thumbnails."""
    n = len(thumbs)
    if n == 0:
        return []
    diffs = np.array([0.0] + [float(np.abs(thumbs[i].astype(np.int16) - thumbs[i - 1]).mean()) for i in range(1, n)])
    thr = max(18.0, 5.0 * float(np.median(diffs[1:])) if n > 1 else 18.0)
    cuts = [0] + [i for i in range(1, n) if diffs[i] > thr] + [n]
    return [(a, b) for a, b in zip(cuts, cuts[1:]) if b > a]


def build_tracks(faces: list[list[Face]], a: int, b: int, scale: float, aw: int) -> list[Track]:
    """Group the faces in a shot by x position into at most 2 tracks."""
    dets = [(i, f) for i in range(a, b) for f in faces[i]]
    if not dets:
        return []
    xs = sorted(f.cx for _, f in dets)
    split_x = None
    if len(xs) > 1:
        gaps = np.diff(xs)
        g = int(np.argmax(gaps))
        if gaps[g] > 0.12 * aw:
            split_x = (xs[g] + xs[g + 1]) / 2
    groups: list[list[tuple[int, Face]]] = [[], []] if split_x is not None else [[]]
    for i, f in dets:
        groups[1 if split_x is not None and f.cx > split_x else 0].append((i, f))
    min_count = max(2, 0.25 * (b - a))
    tracks = []
    for grp in groups:
        if len(grp) < min_count:
            continue
        per_frame: dict[int, Face] = {}
        for i, f in grp:
            if i not in per_frame or f.score > per_frame[i].score:
                per_frame[i] = f
        fs = list(per_frame.values())
        tracks.append(Track(
            cx=float(np.median([f.cx for f in fs])) * scale, cy=float(np.median([f.cy for f in fs])) * scale,
            w=float(np.median([f.w for f in fs])) * scale, h=float(np.median([f.h for f in fs])) * scale,
            frames=per_frame,
        ))
    return sorted(tracks, key=lambda t: t.cx)


def role_at(turns: list[list], t: float) -> str | None:
    for a, b, r in turns:
        if a <= t < b:
            return r
    return None


def map_box(box, rect, oy: float, out_h: int) -> list[float] | None:
    bx, by, bw, bh = box
    rx, ry, rw, rh = rect
    sx, sy = OUT_W / rw, out_h / rh
    x, y, w, h = (bx - rx) * sx, (by - ry) * sy + oy, bw * sx, bh * sy
    if x + w < 0 or x > OUT_W or y + h < oy or y > oy + out_h:
        return None
    return [round(x, 1), round(y, 1), round(w, 1), round(h, 1)]


def faces_out(label: tuple, boxes: list) -> list[list[float]]:
    out = []
    if label[0] == "split":
        for rect, oy in ((label[1], 0), (label[2], OUT_H // 2)):
            out += [m for b in boxes if (m := map_box(b, rect, oy, OUT_H // 2))]
    else:
        out += [m for b in boxes if (m := map_box(b, label[1], 0, OUT_H))]
    return out


def reframe_clip(src: Path, info: dict, start: float, end: float, turns_src: list[list], host_side: str,
                 cfg) -> dict:
    W, H = info["width"], info["height"]
    aw, ah = analysis_size(W, H)
    scale = W / aw
    fps = cfg.analysis_fps
    dur = end - start
    det = FaceDetector(aw, ah)

    faces: list[list[Face]] = []
    thumbs: list[np.ndarray] = []
    motion: list[list[float]] = []
    prev = None
    for frame in read_frames(src, start, dur, aw, ah, fps):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fs = det(frame)
        faces.append(fs)
        motion.append([mouth_motion(f, gray, prev) for f in fs])
        thumbs.append(cv2.resize(gray, (64, 36), interpolation=cv2.INTER_AREA))
        prev = gray
    n = len(faces)
    if n == 0:
        rect = full_rect(None, None, W, H)
        return {"segments": [{"start": 0.0, "end": round(dur, 3), "kind": "crop", "rects": [list(rect)], "faces": []}],
                "shots": [], "frames": 0}

    turns = [[a - start, b - start, r] for a, b, r in turns_src]
    shots = detect_shots(thumbs)
    labels: list[tuple] = [("crop", full_rect(None, None, W, H))] * n
    frame_boxes: list[list] = [[] for _ in range(n)]

    for a, b in shots:
        tracks = build_tracks(faces, a, b, scale, aw)
        for i in range(a, b):
            frame_boxes[i] = [t.box() for t in tracks]
        if not tracks:
            continue
        if len(tracks) == 1:
            t = tracks[0]
            for i in range(a, b):
                labels[i] = ("crop", full_rect(t.cx, t.cy, W, H))
            continue

        host, guest = (tracks[0], tracks[-1]) if host_side == "left" else (tracks[-1], tracks[0])
        roles: list[str] = []
        for i in range(a, b):
            r = role_at(turns, i / fps)
            if r is None:  # fallback: mouth motion over a ±1 s window
                lo, hi = max(a, i - fps), min(b, i + fps + 1)

                def energy(t: Track) -> float:
                    vals = []
                    for k in range(lo, hi):
                        if k in t.frames and t.frames[k] in faces[k]:
                            vals.append(motion[k][faces[k].index(t.frames[k])])
                    return float(np.mean(vals)) if vals else 0.0

                r = "host" if energy(host) >= energy(guest) else "guest"
            roles.append(r)
        for k, i in enumerate(range(a, b)):
            t = host if roles[k] == "host" else guest
            labels[i] = ("crop", full_rect(t.cx, t.cy, W, H))

        if cfg.split_screen:
            # quick back-and-forth: ≥3 consecutive turns each shorter than rapid_turn_s → split screen
            role_runs = coalesce([Run(k, k + 1, roles[k]) for k in range(len(roles))])
            short = [r.dur / fps < cfg.rapid_turn_s for r in role_runs]
            k = 0
            split_label = ("split", panel_rect(tracks[0], W, H), panel_rect(tracks[-1], W, H))
            while k < len(role_runs):
                if not short[k]:
                    k += 1
                    continue
                m = k
                while m < len(role_runs) and short[m]:
                    m += 1
                if m - k >= 3:
                    for rr in role_runs[k:m]:
                        for kk in range(int(rr.start), int(rr.end)):
                            labels[a + kk] = split_label
                k = m

    frame_runs = [Run(i / fps, min(dur, (i + 1) / fps), labels[i]) for i in range(n)]
    frame_runs[-1] = Run(frame_runs[-1].start, dur, frame_runs[-1].label)
    runs = merge_short_runs(frame_runs, cfg.min_segment_s)

    segments = []
    for r in runs:
        i0, i1 = int(r.start * fps), max(int(r.start * fps) + 1, int(round(r.end * fps)))
        boxes = {tuple(round(v) for v in b) for i in range(i0, min(i1, n)) for b in frame_boxes[i]}
        label = r.label
        segments.append({
            "start": round(r.start, 3),
            "end": round(r.end, 3),
            "kind": label[0],
            "rects": [list(x) for x in label[1:]],
            "faces": faces_out(label, sorted(boxes)),
        })
    return {"segments": segments, "shots": [[a / fps, b / fps] for a, b in shots], "frames": n}


def snap_to_frames(segments: list[dict], fps: float, total: float) -> list[dict]:
    """Align segment boundaries to the output frame grid so per-segment frame counts add up exactly."""
    out = []
    for s in segments:
        a, b = round(s["start"] * fps) / fps, round(s["end"] * fps) / fps
        if s is segments[-1]:
            b = round(total * fps) / fps
        if b - a >= 1 / fps - 1e-9:
            out.append({**s, "start": round(a, 4), "end": round(b, 4), "frames": int(round((b - a) * fps))})
    if out:
        out[0]["start"] = 0.0
        out[0]["frames"] = int(round(out[0]["end"] * fps))
    for prev, cur in zip(out, out[1:]):
        if abs(cur["start"] - prev["end"]) > 1e-6:
            cur["start"] = prev["end"]
            cur["frames"] = int(round((cur["end"] - cur["start"]) * fps))
    return [s for s in out if s["frames"] > 0]
