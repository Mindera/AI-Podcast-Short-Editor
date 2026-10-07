"""Shared helpers: logging, paths, ffmpeg/ffprobe runners with a stall watchdog."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
import unicodedata
from pathlib import Path

from rich.console import Console

console = Console(stderr=True, highlight=False)


def log(msg: str) -> None:
    console.print(f"[dim]{time.strftime('%H:%M:%S')}[/dim] {msg}")


def warn(msg: str) -> None:
    console.print(f"[dim]{time.strftime('%H:%M:%S')}[/dim] [yellow]warning:[/yellow] {msg}")


def tool_home() -> Path:
    """Where fonts/ and models/ live: $SHORTS_HOME, the repo checkout, or ~/.local/share/shorts."""
    if env := os.environ.get("SHORTS_HOME"):
        return Path(env).expanduser()
    repo = Path(__file__).resolve().parents[2]
    if (repo / "pyproject.toml").exists():
        return repo
    return Path.home() / ".local" / "share" / "shorts"


def slugify(text: str, max_len: int = 50) -> str:
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_text.lower()).strip("-")
    return slug[:max_len].rstrip("-") or "clip"


def fmt_ts(t: float) -> str:
    t = max(0, int(t))
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def hex_rgb(color: str) -> tuple[int, int, int]:
    c = color.lstrip("#")
    return int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)


def require_tools() -> None:
    missing = [t for t in ("ffmpeg", "ffprobe") if not shutil.which(t)]
    if missing:
        raise SystemExit(f"missing required tools: {', '.join(missing)} (brew install ffmpeg)")


def ffmpeg_filters() -> set[str]:
    out = subprocess.run(["ffmpeg", "-hide_banner", "-filters"], capture_output=True, text=True).stdout
    names = set()
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 3 and "->" in parts[2]:
            names.add(parts[1])
    return names


def has_libass() -> bool:
    f = ffmpeg_filters()
    return "subtitles" in f and "ass" in f


def ffprobe_json(path: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout
    return json.loads(out)


class StallError(RuntimeError):
    pass


def run_ffmpeg(args: list[str], log_path: Path, stall_s: float = 300, label: str = "ffmpeg") -> str:
    """Run ffmpeg with -progress; kill it if the progress file stops changing for `stall_s` seconds.

    Returns the stderr log text. macOS has no `timeout`, so this is our hang detector.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    prog = log_path.with_suffix(".progress")
    prog.unlink(missing_ok=True)
    cmd = ["ffmpeg", "-hide_banner", "-nostdin", "-y", "-nostats", "-progress", str(prog), *args]
    with open(log_path, "w") as lf:
        lf.write(" ".join(cmd) + "\n\n")
        lf.flush()
        p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=lf)
        last_sig, last_change = None, time.time()
        while p.poll() is None:
            time.sleep(0.5)
            sig = prog.stat().st_mtime_ns if prog.exists() else None
            if sig != last_sig:
                last_sig, last_change = sig, time.time()
            elif time.time() - last_change > stall_s:
                p.kill()
                p.wait()
                raise StallError(f"{label}: no progress for {stall_s}s, killed (log: {log_path})")
    text = log_path.read_text(errors="replace")
    if p.returncode != 0:
        tail = "\n".join(text.splitlines()[-15:])
        raise RuntimeError(f"{label} failed (exit {p.returncode}), log: {log_path}\n{tail}")
    return text


def cpu_count() -> int:
    return os.cpu_count() or 4
