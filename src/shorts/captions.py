"""Step 5: word-by-word captions.

This ffmpeg build may lack libass, so captions are always drawn as transparent full-frame PNGs with Pillow
and overlaid from an ffconcat timeline. An .ass file is still written as a deliverable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

from .cache import atomic_write_json, atomic_write_text
from .config import Brand, Config
from .reframe import OUT_H, OUT_W
from .util import hex_rgb
from .words import Word, apply_corrections, capitalize_first, drop_fillers

SENTENCE_END = re.compile(r"[.?!…]['\"”’)]*$")


@dataclass
class Page:
    lines: list[list[Word]]
    start: float = 0.0
    end: float = 0.0
    y: float = 0.0  # centre of the text block
    bbox: list[float] = field(default_factory=list)

    @property
    def words(self) -> list[Word]:
        return [w for line in self.lines for w in line]


def clip_words(all_words: list[Word], start: float, end: float, cfg: Config, corrections: dict[str, str]) -> list[Word]:
    """Words inside the clip, clip-relative, corrected, fillers dropped (captions only), first word capitalised."""
    ws = [Word(max(0.0, w.start - start), min(end - start, w.end - start), w.text)
          for w in all_words if w.end > start + 0.02 and w.start < end - 0.02]
    ws = apply_corrections(ws, corrections, cfg.terms)
    ws = drop_fillers(ws, cfg.fillers)
    return capitalize_first(ws)


def layout_pages(words: list[Word], font: ImageFont.FreeTypeFont, brand: Brand, clip_dur: float) -> list[Page]:
    max_w = brand.max_text_width - 2 * brand.outline_px
    space = font.getlength(" ")

    def width(line: list[Word]) -> float:
        return sum(font.getlength(w.text) for w in line) + space * max(0, len(line) - 1)

    pages: list[Page] = []
    lines: list[list[Word]] = [[]]
    prev: Word | None = None
    for w in words:
        new_page = prev is not None and (w.start - prev.end > brand.pause_break or SENTENCE_END.search(prev.text))
        if new_page and lines[0]:
            pages.append(Page(lines))
            lines = [[]]
        line = lines[-1]
        if line and (len(line) >= brand.words_per_line or width(line + [w]) > max_w):
            if len(lines) >= brand.max_lines:
                pages.append(Page(lines))
                lines = [[w]]
            else:
                lines.append([w])
        else:
            line.append(w)
        prev = w
    if lines[0]:
        pages.append(Page(lines))

    for i, p in enumerate(pages):
        p.start = p.words[0].start
        last_end = p.words[-1].end + 0.5
        nxt = pages[i + 1].words[0].start if i + 1 < len(pages) else clip_dur
        gap = nxt - p.words[-1].end
        p.end = nxt if gap <= brand.pause_break else min(nxt, last_end)
        p.end = max(p.end, p.words[-1].end)
    return pages


def _overlaps(a: list[float], b: list[float], margin: float = 20) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx, by, bw, bh = b
    return ax1 < bx + bw + margin and bx - margin < ax2 and ay1 < by + bh + margin and by - margin < ay2


def place_pages(pages: list[Page], font: ImageFont.FreeTypeFont, brand: Brand, segments: list[dict]) -> None:
    """Pick a y for each page: default 64 % (or the seam in split-screen), moved off faces, inside safe zones."""
    ascent, descent = font.getmetrics()
    lh = (ascent + descent) * 1.08
    space = font.getlength(" ")
    pad = brand.outline_px + brand.shadow_px
    top_limit, bottom_limit = brand.safe_top, OUT_H - brand.safe_bottom
    for p in pages:
        block_h = lh * len(p.lines)
        widest = max(sum(font.getlength(w.text) for w in ln) + space * (len(ln) - 1) for ln in p.lines)
        segs = [s for s in segments if s["start"] < p.end and s["end"] > p.start]
        split = any(s["kind"] == "split" for s in segs)
        faces = [f for s in segs for f in s.get("faces", [])]
        default = OUT_H / 2 if split else brand.caption_y * OUT_H
        lo, hi = top_limit + block_h / 2 + pad, bottom_limit - block_h / 2 - pad
        candidates = [default] + [default + d for k in range(1, 12) for d in (k * 40, -k * 40)]
        x1, x2 = OUT_W / 2 - widest / 2 - pad, OUT_W / 2 + widest / 2 + pad
        chosen = None
        for y in candidates:
            if not lo <= y <= hi:
                continue
            box = [x1, y - block_h / 2 - pad, x2, y + block_h / 2 + pad]
            if not any(_overlaps(box, f) for f in faces):
                chosen = y
                break
        p.y = chosen if chosen is not None else _clamp(default, lo, hi)
        p.bbox = [round(x1, 1), round(p.y - block_h / 2 - pad, 1), round(x2, 1), round(p.y + block_h / 2 + pad, 1)]


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def draw_page(page: Page, active: int | None, font: ImageFont.FreeTypeFont, brand: Brand) -> Image.Image:
    ascent, descent = font.getmetrics()
    lh = (ascent + descent) * 1.08
    space = font.getlength(" ")
    top = page.y - lh * len(page.lines) / 2
    outline = hex_rgb(brand.outline_color)
    text_c, hi_c = hex_rgb(brand.text_color), hex_rgb(brand.highlight_color)

    placements = []  # (x, baseline_y, text, is_active)
    k = 0
    for li, line in enumerate(page.lines):
        lw = sum(font.getlength(w.text) for w in line) + space * (len(line) - 1)
        x = OUT_W / 2 - lw / 2
        base = top + li * lh + ascent
        for w in line:
            placements.append((x, base, w.text, k == active))
            x += font.getlength(w.text) + space
            k += 1

    shadow = Image.new("RGBA", (OUT_W, OUT_H), (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow)
    for x, y, t, _ in placements:
        sd.text((x, y + brand.shadow_px), t, font=font, anchor="ls", fill=(*outline, 170),
                stroke_width=brand.outline_px, stroke_fill=(*outline, 170))
    img = shadow.filter(ImageFilter.GaussianBlur(brand.shadow_px))
    d = ImageDraw.Draw(img)
    for x, y, t, is_active in placements:
        d.text((x, y), t, font=font, anchor="ls", fill=(*(hi_c if is_active else text_c), 255),
               stroke_width=brand.outline_px, stroke_fill=(*outline, 255))
    return img


def _ass_color(hex_color: str) -> str:
    r, g, b = hex_rgb(hex_color)
    return f"&H00{b:02X}{g:02X}{r:02X}"


def _ass_ts(t: float) -> str:
    cs = int(round(t * 100))
    h, cs = divmod(cs, 360000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def write_ass(path: Path, pages: list[Page], font: ImageFont.FreeTypeFont, brand: Brand) -> None:
    family = font.getname()[0]
    head = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {OUT_W}
PlayResY: {OUT_H}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Caption,{family},{brand.font_size},{_ass_color(brand.text_color)},{_ass_color(brand.highlight_color)},{_ass_color(brand.outline_color)},&H80{_ass_color(brand.outline_color)[4:]},0,0,0,0,100,100,0,0,1,{brand.outline_px},{brand.shadow_px},5,0,{brand.safe_right},0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    events = []
    hi = _ass_color(brand.highlight_color)
    base = _ass_color(brand.text_color)
    for p in pages:
        ws = p.words
        for k, w in enumerate(ws):
            s = p.start if k == 0 else w.start
            e = ws[k + 1].start if k + 1 < len(ws) else p.end
            parts, idx = [], 0
            for line in p.lines:
                toks = []
                for lw in line:
                    t = lw.text.replace("{", "(").replace("}", ")")
                    toks.append(f"{{\\c{hi}}}{t}{{\\c{base}}}" if idx == k else t)
                    idx += 1
                parts.append(" ".join(toks))
            text = f"{{\\an5\\pos({OUT_W // 2},{int(p.y)})}}" + "\\N".join(parts)
            events.append(f"Dialogue: 0,{_ass_ts(s)},{_ass_ts(e)},Caption,,0,0,0,,{text}")
    atomic_write_text(path, head + "\n".join(events) + "\n")


def build_captions(words: list[Word], clip_dur: float, segments: list[dict], font_file: Path, brand: Brand,
                   out_dir: Path, ass_path: Path) -> dict:
    """Render caption PNG states + caps.ffconcat in out_dir. Returns metadata for QA."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.png"):
        old.unlink()  # regenerated intermediates inside _work only
    font = ImageFont.truetype(str(font_file), brand.font_size)
    pages = layout_pages(words, font, brand, clip_dur)
    place_pages(pages, font, brand, segments)

    blank = out_dir / "blank.png"
    Image.new("RGBA", (OUT_W, OUT_H), (0, 0, 0, 0)).save(blank, compress_level=1)
    timeline: list[tuple[float, str]] = []  # (start time, file)
    t = 0.0
    for pi, p in enumerate(pages):
        if p.start > t + 1e-3:
            timeline.append((t, blank.name))
        ws = p.words
        for k, w in enumerate(ws):
            s = p.start if k == 0 else w.start
            name = f"p{pi:03d}_{k:02d}.png"
            draw_page(p, k, font, brand).save(out_dir / name, compress_level=1)
            timeline.append((s, name))
        t = p.end
    if t < clip_dur:
        timeline.append((t, blank.name))
    if not timeline:
        timeline.append((0.0, blank.name))

    lines = ["ffconcat version 1.0"]
    for i, (s, name) in enumerate(timeline):
        e = timeline[i + 1][0] if i + 1 < len(timeline) else clip_dur
        lines += [f"file '{name}'", f"duration {max(0.001, e - s):.3f}"]
    lines.append(f"file '{timeline[-1][1]}'")  # concat demuxer ignores the last duration otherwise
    atomic_write_text(out_dir / "caps.ffconcat", "\n".join(lines) + "\n")
    write_ass(ass_path, pages, font, brand)

    meta = {"pages": [{"start": p.start, "end": p.end, "y": p.y, "bbox": p.bbox,
                       "text": " / ".join(" ".join(w.text for w in ln) for ln in p.lines)} for p in pages]}
    atomic_write_json(out_dir / "captions.json", meta)
    return meta
