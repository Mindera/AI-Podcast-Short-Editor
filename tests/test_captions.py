from PIL import ImageFont

from shorts.captions import build_captions, layout_pages
from shorts.config import Brand
from shorts.words import Word


def _words(text, gap_at=None):
    ws, t = [], 0.0
    for i, tok in enumerate(text.split()):
        if gap_at is not None and i == gap_at:
            t += 1.0
        ws.append(Word(t, t + 0.3, tok))
        t += 0.35
    return ws


def test_layout_limits_and_pause_breaks(font_file):
    brand = Brand()
    font = ImageFont.truetype(str(font_file), brand.font_size)
    ws = _words("one two three four five six seven eight nine ten eleven", gap_at=4)
    pages = layout_pages(ws, font, brand, 10)
    assert pages[0].words[-1].text == "four"  # pause > 0.6 s starts a new screen
    for p in pages:
        assert len(p.lines) <= 2 and all(len(l) <= 3 for l in p.lines)
    for a, b in zip(pages, pages[1:]):
        assert a.end <= b.start + 1e-9


def test_build_captions_avoids_faces_and_safe_zones(tmp_path, font_file):
    brand = Brand()
    ws = _words("this caption should move away from the face below")
    face = [300, 1180, 480, 300]  # sits right where 64 % would be
    segs = [{"start": 0, "end": 10, "kind": "crop", "faces": [face]}]
    meta = build_captions(ws, 10, segs, font_file, brand, tmp_path / "caps", tmp_path / "x.ass")
    for p in meta["pages"]:
        x1, y1, x2, y2 = p["bbox"]
        assert y1 >= brand.safe_top and y2 <= 1920 - brand.safe_bottom
        fx, fy, fw, fh = face
        assert not (y1 < fy + fh and fy < y2)
    assert (tmp_path / "caps" / "caps.ffconcat").read_text().startswith("ffconcat version 1.0")
    assert "Dialogue:" in (tmp_path / "x.ass").read_text()


def test_overwide_single_word_shrinks_to_fit(tmp_path, font_file):
    brand = Brand()
    ws = _words("Rindfleischetikettierungsueberwachungsaufgaben then normal words again")
    segs = [{"start": 0, "end": 10, "kind": "crop", "faces": []}]
    meta = build_captions(ws, 10, segs, font_file, brand, tmp_path / "caps", tmp_path / "x.ass")
    for p in meta["pages"]:
        x1, y1, x2, y2 = p["bbox"]
        assert x1 >= 0 and x2 <= 1080 - brand.safe_right + brand.outline_px, p["text"]
    # the shrunken page carries its size into the .ass deliverable
    assert "\\fs" in (tmp_path / "x.ass").read_text()
