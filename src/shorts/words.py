"""Word-level helpers shared by snap and captions."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class Word:
    start: float
    end: float
    text: str


def norm(text: str) -> str:
    return re.sub(r"[^\w]", "", text.lower())


def merge_tokens(raw: list[dict]) -> list[Word]:
    """Merge Whisper continuation tokens (no leading space, or punctuation only) into the previous word."""
    out: list[Word] = []
    for tok in raw:
        t = tok["word"]
        stripped = t.strip()
        if not stripped:
            continue
        continuation = out and (not t.startswith(" ") or not re.search(r"\w", stripped))
        if continuation:
            out[-1].text += stripped
            out[-1].end = max(out[-1].end, float(tok["end"]))
        else:
            out.append(Word(float(tok["start"]), float(tok["end"]), stripped))
    return out


def transcript_words(transcript: dict) -> list[Word]:
    raw = [w for seg in transcript["segments"] for w in seg.get("words", [])]
    return merge_tokens(raw)


def _split_punct(text: str) -> tuple[str, str, str]:
    m = re.match(r"^(\W*)(.*?)(\W*)$", text, flags=re.S)
    return (m.group(1), m.group(2), m.group(3)) if m else ("", text, "")


def _spread(start: float, end: float, text: str) -> list[Word]:
    """Split a (possibly multi-word) replacement back into words, timing them by character length,
    so captions can still wrap and highlight word by word."""
    parts = text.split()
    total = sum(len(p) for p in parts) or 1
    out, t = [], start
    for p in parts:
        dt = (end - start) * len(p) / total
        out.append(Word(round(t, 3), round(t + dt, 3), p))
        t += dt
    return out


def apply_corrections(words: list[Word], corrections: dict[str, str], terms: list[str]) -> list[Word]:
    """Whole-word replacements, including multi-word ones ("my nearer" → "Mindera"), plus term casing."""
    rules: list[tuple[tuple[str, ...], str]] = []
    for wrong, right in corrections.items():
        toks = tuple(norm(t) for t in wrong.split() if norm(t))
        if toks:
            rules.append((toks, right))
    for term in terms:
        toks = tuple(norm(t) for t in term.split() if norm(t))
        if len(toks) > 1:
            rules.append((toks, term))
    rules.sort(key=lambda r: -len(r[0]))
    single_terms = {norm(t): t for t in terms if len(t.split()) == 1}

    out: list[Word] = []
    i = 0
    while i < len(words):
        for toks, right in rules:
            n = len(toks)
            if tuple(norm(w.text) for w in words[i:i + n]) == toks:
                lead, _, _ = _split_punct(words[i].text)
                _, _, trail = _split_punct(words[i + n - 1].text)
                out += _spread(words[i].start, words[i + n - 1].end, f"{lead}{right}{trail}")
                i += n
                break
        else:
            w = words[i]
            if (canon := single_terms.get(norm(w.text))) is not None:
                lead, core, trail = _split_punct(w.text)
                if core.lower() == canon.lower():
                    w = Word(w.start, w.end, f"{lead}{canon}{trail}")
            out.append(w)
            i += 1
    return out


def drop_fillers(words: list[Word], fillers: list[str]) -> list[Word]:
    bad = {norm(f) for f in fillers}
    return [w for w in words if norm(w.text) not in bad]


def capitalize_first(words: list[Word]) -> list[Word]:
    if words:
        lead, core, trail = _split_punct(words[0].text)
        if core:
            words[0] = Word(words[0].start, words[0].end, f"{lead}{core[0].upper()}{core[1:]}{trail}")
    return words
