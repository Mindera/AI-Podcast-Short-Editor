"""Project-local assets: caption font and the YuNet face model, downloaded on first use."""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path

from .util import log, tool_home

# Each asset has a mirror: some networks block one host (e.g. GitHub raw content behind a proxy).
MONTSERRAT_URLS = (
    "https://github.com/JulietaUla/Montserrat/raw/master/fonts/ttf/Montserrat-ExtraBold.ttf",
    "https://cdn.jsdelivr.net/gh/JulietaUla/Montserrat@master/fonts/ttf/Montserrat-ExtraBold.ttf",
)
YUNET_URLS = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "https://huggingface.co/opencv/face_detection_yunet/resolve/main/face_detection_yunet_2023mar.onnx",
)
# Preference order among PP Neue Machina weights, heaviest first.
MACHINA_WEIGHTS = ("Ultrabold", "UltraBold", "Extrabold", "ExtraBold", "Bold", "Black", "Medium", "Regular")


def _download(urls: tuple[str, ...], dest: Path, min_bytes: int = 100_000) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    errors = []
    for url in urls:
        log(f"downloading {dest.name} ← {url.split('/')[2]}")
        try:
            with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
                while chunk := r.read(1 << 16):
                    f.write(chunk)
            if tmp.stat().st_size < min_bytes:  # an error page or a git-lfs pointer, not the asset
                raise OSError(f"got {tmp.stat().st_size} bytes, expected at least {min_bytes}")
            os.replace(tmp, dest)
            return dest
        except Exception as e:
            errors.append(f"{url}: {e}")
    tmp.unlink(missing_ok=True)
    raise RuntimeError(f"could not download {dest.name}:\n  " + "\n  ".join(errors))


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
    return dest if dest.exists() else _download(MONTSERRAT_URLS, dest)


def yunet_path() -> Path:
    dest = tool_home() / "models" / "face_detection_yunet_2023mar.onnx"
    return dest if dest.exists() else _download(YUNET_URLS, dest)


def ensure_assets(explicit_font: str | None = None) -> None:
    font_path(explicit_font)
    yunet_path()
