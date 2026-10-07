"""An episode = one source video and its output folder (<out>/<slug>/ with _work/ inside)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .cache import atomic_write_json, read_json
from .config import Config, load_config
from .util import slugify

DRIVE_PATTERNS = (r"/file/d/([A-Za-z0-9_-]{10,})", r"[?&]id=([A-Za-z0-9_-]{10,})", r"/open\?id=([A-Za-z0-9_-]{10,})")


def drive_id(source: str) -> str | None:
    import re

    if "drive.google.com" not in source and "docs.google.com" not in source:
        return None
    for pat in DRIVE_PATTERNS:
        if m := re.search(pat, source):
            return m.group(1)
    return None


def default_slug(source: str) -> str:
    if did := drive_id(source):
        return f"drive-{did[:10]}"
    if source.startswith(("http://", "https://")):
        return slugify(source.rstrip("/").rsplit("/", 1)[-1].split("?")[0]) or "episode"
    return slugify(Path(source).stem)


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
        ep = cls(dir=out / (slugify(name) if name else default_slug(source)), source=source, cfg=cfg)
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
