"""Project-local assets: caption font and the YuNet face model, downloaded on first use."""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path

from .util import log, tool_home

MONTSERRAT_URL = "https://github.com/JulietaUla/Montserrat/raw/master/fonts/ttf/Montserrat-ExtraBold.ttf"
YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
# Preference order among PP Neue Machina weights, heaviest first.
MACHINA_WEIGHTS = ("Ultrabold", "UltraBold", "Extrabold", "ExtraBold", "Bold", "Black", "Medium", "Regular")


def _download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    log(f"downloading {url.rsplit('/', 1)[-1]} → {dest}")
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 16):
            f.write(chunk)
    os.replace(tmp, dest)
    return dest


def font_path(explicit: str | None = None) -> Path:
    if explicit:
        p = Path(explicit).expanduser()
        if not p.exists():
            raise SystemExit(f"font not found: {p}")
        return p
    for fonts in (Path.cwd() / "fonts", tool_home() / "fonts"):
        machina = sorted(fonts.glob("PPNeueMachina-*.otf"))
        for weight in MACHINA_WEIGHTS:
            for f in machina:
                if weight in f.stem:
                    return f
        if machina:
            return machina[0]
    dest = tool_home() / "fonts" / "Montserrat-ExtraBold.ttf"
    return dest if dest.exists() else _download(MONTSERRAT_URL, dest)


def yunet_path() -> Path:
    dest = tool_home() / "models" / "face_detection_yunet_2023mar.onnx"
    return dest if dest.exists() else _download(YUNET_URL, dest)


def ensure_assets(explicit_font: str | None = None) -> None:
    font_path(explicit_font)
    yunet_path()
