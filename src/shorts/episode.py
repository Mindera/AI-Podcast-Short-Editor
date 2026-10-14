"""An episode = one source video and its output folder (<out>/<slug>/ with _work/ inside)."""

from __future__ import annotations

import html
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from .cache import atomic_write_json, read_json
from .config import Config, load_config
from .util import slugify

DRIVE_PATTERNS = (r"/file/d/([A-Za-z0-9_-]{10,})", r"[?&]id=([A-Za-z0-9_-]{10,})", r"/open\?id=([A-Za-z0-9_-]{10,})")


def drive_id(source: str) -> str | None:
    if "drive.google.com" not in source and "docs.google.com" not in source:
        return None
    for pat in DRIVE_PATTERNS:
        if m := re.search(pat, source):
            return m.group(1)
    return None


# filename tokens that say nothing about the episode (export/version/resolution noise)
_NOISE = {"final", "copy", "export", "exported", "master", "render", "edit", "edited", "raw", "full", "video",
          "mp4", "mov", "mkv", "hd", "uhd", "4k", "1080p", "720p", "2160p", "h264", "h265", "hevc", "prores"}
_NOISE_RE = re.compile(r"^(v\d+|\d+fps)$")


def short_slug(filename: str, max_len: int = 40) -> str:
    """A short folder name from a video filename: its distinctive words, cut at a word boundary.

    "Podcast EP12 - Jane Doe on AI (FINAL v3) 1080p.mp4" → "podcast-ep12-jane-doe-on-ai".
    """
    stem = re.sub(r"\.(mp4|mov|mkv|m4v|webm|avi)$", "", filename.strip(), flags=re.I)
    words = [w for w in slugify(stem, max_len=200).split("-") if w and w not in _NOISE and not _NOISE_RE.match(w)]
    out = ""
    for w in words:
        if out and len(out) + 1 + len(w) > max_len:
            break
        out = f"{out}-{w}" if out else w[:max_len]
    return out or "episode"


def drive_title(did: str, timeout: float = 10) -> str | None:
    """The Drive file's name from its public view page, or None (private file, offline…)."""
    req = urllib.request.Request(f"https://drive.google.com/file/d/{did}/view", headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            page = r.read(200_000).decode("utf-8", "replace")
    except Exception:
        return None
    m = re.search(r"<title>(.*?) - Google Drive</title>", page, re.S)
    return (html.unescape(m.group(1)).strip() or None) if m else None


def default_slug(source: str) -> str:
    if did := drive_id(source):
        title = drive_title(did)
        return short_slug(title) if title else f"drive-{did[:10]}"
    if source.startswith(("http://", "https://")):
        return short_slug(source.rstrip("/").rsplit("/", 1)[-1].split("?")[0])
    return short_slug(Path(source).name)


def _existing_dir(out: Path, source: str) -> Path | None:
    """An episode folder already made for this source, so re-runs reuse its caches."""
    for state in out.glob("*/_work/episode.json"):
        try:
            if read_json(state).get("source") == source:
                return state.parent.parent
        except Exception:
            continue
    return None


def _free_dir(out: Path, slug: str) -> Path:
    """<out>/<slug>, or <slug>-2, -3… if that name belongs to a different source."""
    d, n = out / slug, 2
    while (d / "_work" / "episode.json").exists():
        d, n = out / f"{slug}-{n}", n + 1
    return d


@dataclass
class Episode:
    dir: Path
    source: str  # original argument (URL or path)
    cfg: Config

    @property
    def work(self) -> Path:
        return self.dir / "_work"

    @property
    def state_path(self) -> Path:
        return self.work / "episode.json"

    @property
    def source_file(self) -> Path:
        """The local video file. Local sources are used in place, read-only."""
        if drive_id(self.source) or self.source.startswith(("http://", "https://")):
            return self.work / "source.mp4"
        return Path(self.source).expanduser().resolve()

    def save(self) -> None:
        atomic_write_json(self.state_path, {"source": self.source, "config": self.cfg.model_dump()})

    def clip_work(self, rank: int) -> Path:
        p = self.work / "clips" / f"{rank:02d}"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @classmethod
    def create(cls, source: str, cfg: Config, name: str | None = None) -> Episode:
        out = Path(cfg.out).expanduser().resolve()
        if name:
            d = out / slugify(name)
        else:
            d = _existing_dir(out, source) or _free_dir(out, default_slug(source))
        ep = cls(dir=d, source=source, cfg=cfg)
        ep.work.mkdir(parents=True, exist_ok=True)
        ep.save()
        return ep

    @classmethod
    def load(cls, episode_dir: Path, config: Path | None = None, overrides: dict | None = None) -> Episode:
        episode_dir = episode_dir.expanduser().resolve()
        state_path = episode_dir / "_work" / "episode.json"
        if not state_path.exists():
            raise SystemExit(f"not an episode folder (no _work/episode.json): {episode_dir}")
        state = read_json(state_path)
        cfg = load_config(config, overrides, base=state["config"])
        ep = cls(dir=episode_dir, source=state["source"], cfg=cfg)
        if overrides or config:
            ep.save()
        return ep
