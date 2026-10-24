"""Shared fixtures. Tests prefer an installed system font so they run without network access."""

from pathlib import Path

import pytest

SYSTEM_FONTS = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",  # Debian/Ubuntu
    "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",  # Fedora
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",  # macOS
    "/System/Library/Fonts/Helvetica.ttc",  # macOS
    "C:/Windows/Fonts/arialbd.ttf",  # Windows
)


@pytest.fixture(scope="session")
def font_file() -> Path:
    for p in map(Path, SYSTEM_FONTS):
        if p.exists():
            return p
    from shorts.assets import font_path

    return font_path()  # last resort: the brand font (downloads on first use)
