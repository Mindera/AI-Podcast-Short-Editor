"""Step 2: faster-whisper transcription with word timestamps, cached per chunk."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .cache import atomic_write_json, atomic_write_text, file_sig, is_fresh, make_key, mark_done, read_json
from .episode import Episode
from .probe import SR, read_wav
from .util import cpu_count, fmt_ts, log


def initial_prompt(ep: Episode) -> str:
    cfg = ep.cfg
    return (f"A podcast conversation between {cfg.host} and {cfg.guest}. "
            f"Names and terms: {', '.join(cfg.terms)}.")


def _srt_ts(t: float) -> str:
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def channel_hint(work: Path, start: float, end: float) -> str:
    """'L'/'R' when one mic clearly dominates this segment (separate-mic recordings only)."""
    stereo = work / "stereo.json"
    if not stereo.exists() or not read_json(stereo).get("separate_mics"):
        return ""
    rms = np.load(work / "stereo_rms.npy")
    a, b = int(start * 10), max(int(start * 10) + 1, int(end * 10))
    seg = rms[a:b]
    if len(seg) == 0:
        return ""
    l, r = float(seg[:, 0].mean()), float(seg[:, 1].mean())
    if l > 1.5 * r:
        return "L"
    if r > 1.5 * l:
        return "R"
    return ""


def write_outputs(work: Path, transcript: dict) -> None:
    atomic_write_json(work / "transcript.json", transcript)
    srt, txt = [], []
    for i, s in enumerate(transcript["segments"], 1):
        srt.append(f"{i}\n{_srt_ts(s['start'])} --> {_srt_ts(s['end'])}\n{s['text'].strip()}\n")
        ch = channel_hint(work, s["start"], s["end"])
        tag = f" | mic {ch}" if ch else ""
        txt.append(f"[{s['start']:.2f}-{s['end']:.2f} | {fmt_ts(s['start'])}{tag}] {s['text'].strip()}")
    atomic_write_text(work / "transcript.srt", "\n".join(srt))
    atomic_write_text(work / "transcript.txt", "\n".join(txt) + "\n")


def _transcribe_chunk(model, audio: np.ndarray, offset: float, ep: Episode, total: float) -> list[dict]:
    lang = None if ep.cfg.language == "auto" else ep.cfg.language
    segments, _ = model.transcribe(
        audio,
        language=lang,
        word_timestamps=True,
        vad_filter=True,
        condition_on_previous_text=False,
        initial_prompt=initial_prompt(ep),
        beam_size=5,
    )
    out, t0, last_log = [], time.time(), 0.0
    for s in segments:
        out.append({
            "start": round(s.start + offset, 3),
            "end": round(s.end + offset, 3),
            "text": s.text,
            "words": [{"start": round(w.start + offset, 3), "end": round(w.end + offset, 3), "word": w.word,
                       "prob": round(w.probability, 3)} for w in (s.words or [])],
        })
        if time.time() - last_log > 20:
            done = s.end + offset
            speed = (s.end) / max(1e-6, time.time() - t0)
            log(f"  transcribed {fmt_ts(done)} / {fmt_ts(total)} ({speed:.2f}× real-time)")
            last_log = time.time()
    return out


def run(ep: Episode, force: bool = False) -> dict:
    work, cfg = ep.work, ep.cfg
    wav = work / "audio16k.wav"
    chunks = read_json(work / "chunks.json")
    outputs = [work / "transcript.json", work / "transcript.srt", work / "transcript.txt"]
    params = ("whisper-v1", cfg.whisper_model, cfg.whisper_compute, cfg.language, initial_prompt(ep))
    key = make_key(params, file_sig(wav), chunks, file_sig(work / "stereo.json"))
    if not force and is_fresh(work, "transcribe", key, outputs):
        return _require_speech(read_json(work / "transcript.json"))

    audio = read_wav(wav)
    total = len(audio) / SR
    model = None
    segments: list[dict] = []
    for i, (a, b) in enumerate(chunks):
        chunk_path = work / "chunks" / f"chunk_{i:02d}.json"
        ckey = make_key(params, file_sig(wav), a, b)
        if not force and chunk_path.exists() and read_json(chunk_path).get("key") == ckey:
            segments += read_json(chunk_path)["segments"]
            continue
        if model is None:
            from faster_whisper import WhisperModel

            log(f"transcribe: loading faster-whisper {cfg.whisper_model} ({cfg.whisper_compute}, CPU)")
            model = WhisperModel(cfg.whisper_model, device="cpu", compute_type=cfg.whisper_compute,
                                 cpu_threads=cpu_count())
        log(f"transcribe: chunk {i + 1}/{len(chunks)} {fmt_ts(a)}–{fmt_ts(b)}")
        pcm = audio[int(a * SR):int(b * SR)].astype(np.float32) / 32768.0
        segs = _transcribe_chunk(model, pcm, a, ep, total)
        atomic_write_json(chunk_path, {"key": ckey, "segments": segs})
        segments += segs

    transcript = {"language": cfg.language, "duration": total, "segments": segments}
    write_outputs(work, transcript)
    mark_done(work, "transcribe", key)
    log(f"transcribe: {len(segments)} segments → {work / 'transcript.txt'}")
    return _require_speech(transcript)


def _require_speech(transcript: dict) -> dict:
    if not transcript["segments"]:
        raise SystemExit("transcription found no speech in the source audio — nothing to clip. "
                         "Wrong file, or wrong language? (--lang auto to detect)")
    return transcript
