"""Step caching keyed on inputs, atomic writes, and never-delete supersession."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any


def file_sig(path: Path) -> dict[str, Any]:
    """Cheap identity for big files (size + mtime); content hash for small ones."""
    st = path.stat()
    if st.st_size < 8 * 1024 * 1024:
        return {"path": str(path), "sha": hashlib.sha256(path.read_bytes()).hexdigest()}
    return {"path": str(path), "size": st.st_size, "mtime": st.st_mtime_ns}


def make_key(*parts: Any) -> str:
    blob = json.dumps(parts, sort_keys=True, default=str, ensure_ascii=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:20]


def _stamp(work: Path, step: str) -> Path:
    return work / ".cache" / f"{step}.json"


def is_fresh(work: Path, step: str, key: str, outputs: list[Path]) -> bool:
    stamp = _stamp(work, step)
    if not stamp.exists() or not all(p.exists() for p in outputs):
        return False
    try:
        return json.loads(stamp.read_text()).get("key") == key
    except json.JSONDecodeError:
        return False


def mark_done(work: Path, step: str, key: str) -> None:
    atomic_write_json(_stamp(work, step), {"key": key, "at": time.strftime("%Y-%m-%dT%H:%M:%S")})


def part_path(path: Path) -> Path:
    return path.with_name(path.name + ".part")


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = part_path(path)
    tmp.write_text(text)
    os.replace(tmp, path)


def atomic_write_json(path: Path, data: Any) -> None:
    atomic_write_text(path, json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def supersede(path: Path, work: Path) -> Path | None:
    """Move an outdated output into _work/_superseded/<timestamp>/ instead of deleting it."""
    if not path.exists():
        return None
    dest_dir = work / "_superseded" / time.strftime("%Y%m%d-%H%M%S")
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / path.name
    n = 1
    while dest.exists():
        dest = dest_dir / f"{path.stem}.{n}{path.suffix}"
        n += 1
    shutil.move(str(path), dest)
    return dest
