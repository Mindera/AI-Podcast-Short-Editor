"""Get the source video onto disk. Local files are never copied, moved or modified."""

from __future__ import annotations

import subprocess
import tempfile
import time
from pathlib import Path

from .cache import atomic_write_json
from .episode import Episode, drive_id
from .util import log, warn

# gdown's messages when the file isn't shared publicly (or the quota/permission check fails)
_DRIVE_DENIED = ("Cannot retrieve the public link", "Failed to retrieve file url", "Access denied", "permission")


def _watch(cmd: list[str], dest: Path, log_path: Path, stall_s: int) -> None:
    """Run a downloader in the background and poll that the file keeps growing."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as lf:
        p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT)
        last_size, last_change, last_report = -1, time.time(), 0.0
        while p.poll() is None:
            time.sleep(2)
            size = max((f.stat().st_size for f in dest.parent.glob(dest.name + "*") if f.is_file()), default=0)
            if size != last_size:
                last_size, last_change = size, time.time()
            elif time.time() - last_change > stall_s:
                p.kill()
                raise RuntimeError(f"download stalled for {stall_s}s (log: {log_path}); re-run to resume")
            if time.time() - last_report > 15:
                log(f"  downloading… {size / 1e9:.2f} GB")
                last_report = time.time()
    if p.returncode != 0:
        tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-10:])
        raise RuntimeError(f"download failed (log: {log_path})\n{tail}")


def fetch(ep: Episode) -> Path:
    dest = ep.source_file
    done = ep.work / ".cache" / "fetch.json"
    if did := drive_id(ep.source):
        if dest.exists() and done.exists():
            return dest
        log(f"downloading Drive file {did} (original upload) → {dest}")
        _fetch_drive(did, dest, ep)
    elif ep.source.startswith(("http://", "https://")):
        if dest.exists() and done.exists():
            return dest
        warn("non-Drive URL: downloading with yt-dlp. Re-encoded uploads (e.g. YouTube) lose quality when cropped to 9:16.")
        cmd = ["uvx", "yt-dlp", "-f", "bv*+ba/b", "--merge-output-format", "mp4", "-o", str(dest), ep.source]
        _watch(cmd, dest, ep.work / "logs" / "fetch.log", ep.cfg.stall_s)
    else:
        if not dest.exists():
            raise SystemExit(f"source not found: {dest}")
        return dest
    if not dest.exists() or dest.stat().st_size == 0:
        raise RuntimeError(f"download produced no file at {dest}")
    atomic_write_json(done, {"source": ep.source, "size": dest.stat().st_size})
    return dest


def _fetch_drive(did: str, dest: Path, ep: Episode) -> None:
    url = f"https://drive.google.com/uc?id={did}"
    cmd = ["uvx", "gdown>=6.4", "--continue", url, "-O", str(dest)]
    log_path = ep.work / "logs" / "fetch.log"
    browser = ep.cfg.drive_cookies
    if not browser:
        try:
            _watch(cmd + ["--no-cookies"], dest, log_path, ep.cfg.stall_s)
        except RuntimeError as e:
            if any(m.lower() in str(e).lower() for m in _DRIVE_DENIED):
                raise SystemExit(
                    f"Drive file {did} isn't publicly shared. To download it as your signed-in Google account, "
                    "re-run with --drive-cookies chrome (or brave/firefox/edge/safari…). Your Google cookies are "
                    "copied to a temp file for the download only and deleted right after."
                ) from e
            raise
        return
    # cookies live in a private temp dir for this download only, never in ~/.cache/gdown
    log(f"using your {browser} Google session for the Drive download (macOS may ask for Keychain access)")
    with tempfile.TemporaryDirectory(prefix="shorts-gdown-") as tmp:
        jar = Path(tmp) / "cookies.txt"
        _watch(cmd + ["--cookies", str(jar), "--cookies-from-browser", browser], dest, log_path, ep.cfg.stall_s)
