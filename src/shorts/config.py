"""Configuration: defaults < ./shorts.toml < --config file < CLI flags."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

DEFAULT_TERMS = [
    "Mindera",
    "Minders",
    "Mindera Yellow Box",
    "Jira",
    "MAUI",
    "Kotlin Multiplatform",
    "React Native",
    "phygital",
]


class Brand(BaseModel):
    name: str = "Mindera"
    font: str | None = None  # explicit font path; otherwise ./fonts/PPNeueMachina-* or Montserrat ExtraBold
    text_color: str = "#FFFFFF"
    highlight_color: str = "#FFC700"
    outline_color: str = "#0A0A0B"
    outline_px: int = 6
    shadow_px: int = 3
    font_size: int = 80
    max_text_width: int = 780
    caption_y: float = 0.64  # centre of the caption block, fraction of height
    safe_top: int = 250
    safe_bottom: int = 400
    safe_right: int = 150
    words_per_line: int = 3
    max_lines: int = 2
    pause_break: float = 0.6  # seconds of silence that start a new caption screen
    cover_title_y: float = 0.70
    cover_font_size: int = 96
    cover_darken: float = 0.55
    hashtags: list[str] = Field(default_factory=lambda: ["#Mindera"])


class Config(BaseModel):
    language: str = "en"  # whisper language code, or "auto"
    host: str = "Host"
    guest: str = "Guest"
    host_side: Literal["auto", "left", "right"] = "auto"  # auto = ask the picker, fall back to left
    clips: int = 8
    out: str = "~/Shorts"
    terms: list[str] = Field(default_factory=lambda: list(DEFAULT_TERMS))
    corrections: dict[str, str] = Field(default_factory=lambda: {"my nearer": "Mindera"})
    fillers: list[str] = Field(default_factory=lambda: ["um", "uh", "uhm", "umm", "erm", "er", "hmm", "mm", "ah"])

    whisper_model: str = "medium"
    whisper_compute: str = "int8"
    chunk_over_s: float = 3600.0
    chunk_every_s: float = 1800.0

    claude_bin: str = "claude"
    claude_model: str | None = None
    claude_timeout_s: int = 1800

    min_clip_s: float = 20.0
    ideal_max_s: float = 60.0
    max_clip_s: float = 90.0
    pad_s: float = 0.15

    analysis_fps: int = 10
    min_segment_s: float = 0.6
    rapid_turn_s: float = 2.5
    split_screen: bool = True

    crf: int = 19
    max_fps: int = 30
    jobs: int = 0  # 0 = cores // 6
    stall_s: int = 300  # kill ffmpeg if it makes no progress for this long
    drive_cookies: str | None = None  # browser to borrow Google cookies from for private Drive files, e.g. "chrome"

    brand: Brand = Field(default_factory=Brand)


def _deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: Path | None = None, overrides: dict[str, Any] | None = None, base: dict[str, Any] | None = None) -> Config:
    data: dict[str, Any] = dict(base or {})
    local = Path.cwd() / "shorts.toml"
    for p in (local, path):
        if p and p.exists():
            data = _deep_merge(data, tomllib.loads(p.read_text()))
        elif p is path and p is not None:
            raise FileNotFoundError(f"config file not found: {p}")
    clean = {k: v for k, v in (overrides or {}).items() if v is not None}
    data = _deep_merge(data, clean)
    return Config.model_validate(data)
