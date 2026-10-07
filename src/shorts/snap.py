"""Snap picked boundaries to exact word timestamps, pad, rank, and print the approval summary."""

from __future__ import annotations

import difflib

from rich.table import Table

from .cache import atomic_write_json, file_sig, is_fresh, make_key, mark_done, read_json, supersede
from .episode import Episode
from .pick import Boundary, PickedClip, PickResult
from .util import console, fmt_ts, slugify, warn
from .words import Word, norm, transcript_words

SEARCH_WINDOW_S = 30.0


def _match_score(words: list[Word], i: int, toks: list[str]) -> float:
    cand = [norm(w.text) for w in words[i:i + len(toks)]]
    if len(cand) < len(toks):
        return 0.0
    return sum(difflib.SequenceMatcher(None, a, b).ratio() for a, b in zip(cand, toks)) / len(toks)


def find_phrase(words: list[Word], phrase: str, approx: float, at_end: bool) -> int:
    """Index of the first word (start) or last word (end) of `phrase`, nearest to `approx`."""
    toks = [t for t in (norm(x) for x in phrase.split()) if t]
    if not words:
        raise ValueError("transcript has no words")
    if not toks:
        key = (lambda w: abs(w.end - approx)) if at_end else (lambda w: abs(w.start - approx))
        return min(range(len(words)), key=lambda i: key(words[i]))
    candidates = [i for i, w in enumerate(words) if abs(w.start - approx) <= SEARCH_WINDOW_S] or range(len(words))
    best, best_key = None, None
    for i in candidates:
        score = _match_score(words, i, toks)
        k = (round(score, 2), -abs(words[i].start - approx))
        if best_key is None or k > best_key:
            best, best_key = i, k
    assert best is not None
    if best_key[0] < 0.6:
        warn(f"snap: weak match for {'end' if at_end else 'start'} phrase “{phrase}” near {fmt_ts(approx)}; "
             "using nearest word")
        if at_end:
            return min(range(len(words)), key=lambda i: abs(words[i].end - approx))
        return min(range(len(words)), key=lambda i: abs(words[i].start - approx))
    return best + len(toks) - 1 if at_end else best


def snap_bounds(words: list[Word], b: Boundary, pad: float) -> tuple[float, float, int, int]:
    i = find_phrase(words, b.start_phrase, b.start, at_end=False)
    j = find_phrase(words, b.end_phrase, b.end, at_end=True)
    if j < i:
        i, j = j, i
    start, end = words[i].start, words[j].end
    # pad ~150 ms, never past the midpoint of the gap to the neighbouring word
    lo = (words[i - 1].end + start) / 2 if i > 0 else 0.0
    hi = (end + words[j + 1].start) / 2 if j + 1 < len(words) else end + pad
    return round(max(lo, start - pad, 0.0), 3), round(min(hi, end + pad), 3), i, j


def clean_hashtags(tags: list[str], required: list[str]) -> list[str]:
    out: list[str] = []
    for t in [*required, *tags]:
        t = "#" + t.lstrip("#").replace(" ", "")
        if len(t) > 1 and t.lower() not in {x.lower() for x in out}:
            out.append(t)
    return out[: len(required) + 7]


def normalize_turns(turns, start: float, end: float) -> list[list]:
    ts = sorted(([max(start, float(a)), min(end, float(b)), r] for a, b, r in turns), key=lambda t: t[0])
    ts = [t for t in ts if t[1] > t[0]]
    if not ts:
        return [[start, end, "guest"]]
    ts[0][0] = start
    for a, b in zip(ts, ts[1:]):
        a[1] = b[0]  # fill gaps / remove overlaps: each turn runs until the next one starts
    ts[-1][1] = end
    return [[round(a, 3), round(b, 3), r] for a, b, r in ts if b > a]


def _clip_entry(c: PickedClip, words: list[Word], ep: Episode) -> dict:
    start, end, i, j = snap_bounds(words, c, ep.cfg.pad_s)
    entry = {
        "start": start, "end": end, "duration": round(end - start, 3),
        "hook_text": c.hook_text, "why_it_works": c.why_it_works, "virality_score": c.virality_score,
        "title": c.title.strip()[:60], "key_word": c.key_word, "description": c.description.strip(),
        "hashtags": clean_hashtags(c.hashtags, ep.cfg.brand.hashtags),
        "suggested_cover_timestamp": min(max(c.suggested_cover_timestamp, start + 0.5), end - 0.5),
        "speakers": normalize_turns(c.speakers, start, end),
        "opening_line": " ".join(w.text for w in words[i:i + 14]),
        "shorter_cut": None,
    }
    if c.shorter_cut:
        s2, e2, _, _ = snap_bounds(words, c.shorter_cut, ep.cfg.pad_s)
        entry["shorter_cut"] = {"start": s2, "end": e2, "duration": round(e2 - s2, 3)}
    return entry


def run(ep: Episode, force: bool = False) -> dict:
    work, cfg = ep.work, ep.cfg
    raw_path, out = work / "clips_raw.json", work / "clips.json"
    key = make_key("snap-v1", file_sig(raw_path), file_sig(work / "transcript.json"), cfg.pad_s, cfg.host_side,
                   cfg.corrections, cfg.brand.hashtags)
    if not force and is_fresh(work, "snap", key, [out]):
        return read_json(out)

    result = PickResult.model_validate(read_json(raw_path))
    words = transcript_words(read_json(work / "transcript.json"))
    clips = [_clip_entry(c, words, ep) for c in result.clips]
    clips.sort(key=lambda c: -c["virality_score"])
    for rank, c in enumerate(clips, 1):
        c["rank"] = rank
        c["slug"] = f"{rank:02d}_{slugify(c['title'], 40)}"
        if c["duration"] > cfg.max_clip_s:
            warn(f"clip {rank} is {c['duration']:.0f}s (> {cfg.max_clip_s:.0f}s); consider its shorter cut")
    for a in clips:
        for b in clips:
            if a["rank"] < b["rank"] and a["start"] < b["end"] and b["start"] < a["end"]:
                warn(f"clips {a['rank']} and {b['rank']} overlap")
    runners = [_clip_entry(c, words, ep) for c in result.runners_up]

    host_side = cfg.host_side
    if host_side == "auto":
        host_side = result.layout.host_side if result.layout.host_side != "unknown" else "left"
    data = {
        "host_side": host_side,
        "layout": result.layout.model_dump(),
        "corrections": {**cfg.corrections, **result.corrections},
        "clips": clips,
        "runners_up": runners,
    }
    if out.exists():
        supersede(out, work)
    atomic_write_json(out, data)
    mark_done(work, "snap", key)
    return data


def print_summary(ep: Episode, data: dict) -> None:
    cfg = ep.cfg
    console.print()
    console.print(f"[bold]Layout:[/bold] {data['layout'].get('description') or 'n/a'}  "
                  f"[bold]host side:[/bold] {data['host_side']}")
    t = Table(title=f"Picked clips ({len(data['clips'])})", show_lines=False)
    for col in ("#", "score", "start", "len", "title", "opening line"):
        t.add_column(col, overflow="fold")
    for c in data["clips"]:
        flag = " ⚠" if c["duration"] > cfg.ideal_max_s else ""
        t.add_row(str(c["rank"]), f"{c['virality_score']:.1f}", fmt_ts(c["start"]), f"{c['duration']:.0f}s{flag}",
                  c["title"], c["opening_line"])
    console.print(t)

    long = [c for c in data["clips"] if c["duration"] > cfg.ideal_max_s]
    if long:
        console.print(f"\n[bold]Over {cfg.ideal_max_s:.0f}s[/bold] (suggested shorter cut):")
        for c in long:
            sc = c.get("shorter_cut")
            alt = f"{fmt_ts(sc['start'])}–{fmt_ts(sc['end'])} ({sc['duration']:.0f}s)" if sc else "none suggested"
            console.print(f"  {c['rank']:>2}. {c['title']} — {c['duration']:.0f}s → {alt}")

    if data["runners_up"]:
        console.print("\n[bold]Runners-up[/bold]")
        for r in data["runners_up"]:
            console.print(f"  • {r['virality_score']:.1f}  {fmt_ts(r['start'])}  {r['duration']:.0f}s  {r['title']}")

    console.print("\n[bold]Caption corrections[/bold]")
    for k, v in data["corrections"].items():
        console.print(f"  {k} → {v}")
    console.print(f"\nEdit [cyan]{ep.work / 'clips.json'}[/cyan] to change anything (times, titles, speakers).")
