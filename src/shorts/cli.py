"""`shorts` CLI: one-shot `run`, plus one command per pipeline step."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import Annotated, Optional

import typer

from . import costs, fetch, pick, probe, qa, render, snap, transcribe
from .assets import ensure_assets
from .config import load_config
from .episode import Episode
from .util import console, has_libass, log, require_tools, warn

app = typer.Typer(add_completion=False, no_args_is_help=True, rich_markup_mode="rich",
                  help="Turn a long podcast video into branded vertical shorts with word-by-word captions.")

EpisodeDir = Annotated[Path, typer.Argument(help="Episode folder (<out>/<slug>) created by `shorts run`.")]
ConfigOpt = Annotated[Optional[Path], typer.Option("--config", "-c", help="TOML config file.")]
ForceOpt = Annotated[bool, typer.Option("--force", help="Ignore the cache for this step.")]


def _overrides(**kw) -> dict:
    o = {k: v for k, v in kw.items() if v is not None}
    if isinstance(o.get("terms"), str):
        o["terms"] = [t.strip() for t in o["terms"].split(",") if t.strip()]
    return o


def _env_checks() -> None:
    require_tools()
    if not has_libass():
        log("ffmpeg has no libass: captions are drawn with Pillow and overlaid (an .ass file is still written)")
    if os.environ.get("HF_TOKEN"):
        log("HF_TOKEN is set, but diarization isn't implemented; using transcript turns + stereo mic check")


def _require_picker(cfg) -> None:
    """The pick step runs `claude -p` long after the slow steps; fail now if it's missing."""
    if os.environ.get("SHORTS_PICKER", "claude") != "stub":
        require_tools(cfg.claude_bin)


def _require_fetcher(source: str) -> None:
    """URL and Drive sources are downloaded with `uvx gdown` / `uvx yt-dlp`."""
    from .episode import drive_id

    if drive_id(source) or source.startswith(("http://", "https://")):
        require_tools("uv", "uvx")


@app.command()
def run(
    source: Annotated[str, typer.Argument(help="Google Drive link, other video URL, or local path.")],
    config: ConfigOpt = None,
    out: Annotated[Optional[str], typer.Option("--out", "-o", help="Output root (default ~/Shorts).")] = None,
    name: Annotated[Optional[str], typer.Option("--name", "-n", help="Episode folder name.")] = None,
    host: Annotated[Optional[str], typer.Option(help="Host name.")] = None,
    guest: Annotated[Optional[str], typer.Option(help="Guest name.")] = None,
    clips: Annotated[Optional[int], typer.Option("--clips", "-k", help="Number of clips.")] = None,
    language: Annotated[Optional[str], typer.Option("--lang", "-l", help="Language code, or 'auto'.")] = None,
    terms: Annotated[Optional[str], typer.Option(help="Comma-separated proper nouns/terms to spell right.")] = None,
    host_side: Annotated[Optional[str], typer.Option(help="auto | left | right")] = None,
    model: Annotated[Optional[str], typer.Option("--model", help="Claude model for picking (e.g. opus).")] = None,
    whisper_model: Annotated[Optional[str], typer.Option(help="faster-whisper model (default medium).")] = None,
    jobs: Annotated[Optional[int], typer.Option(help="Parallel renders (default cores/6).")] = None,
    drive_cookies: Annotated[Optional[str], typer.Option(
        "--drive-cookies", help="Browser whose Google login to use for a private Drive file (e.g. chrome).")] = None,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip the approval pause and render straight away.")] = False,
    ai_qa: Annotated[bool, typer.Option("--ai-qa", help="Ask Claude to review the QA contact sheet.")] = False,
) -> None:
    """Do everything: fetch → probe → transcribe → pick → snap → [approve] → render → qa."""
    _env_checks()
    cfg = load_config(config, _overrides(out=out, host=host, guest=guest, clips=clips, language=language, terms=terms,
                                         host_side=host_side, claude_model=model, whisper_model=whisper_model,
                                         jobs=jobs, drive_cookies=drive_cookies))
    _require_fetcher(source)
    _require_picker(cfg)  # pick runs ~30 min in on a long episode; fail now, not then
    ep = Episode.create(source, cfg, name)
    log(f"episode folder: {ep.dir}")
    costs.start(ep, "run")
    try:
        _run_steps(ep, yes, ai_qa)
    finally:
        costs.finish()
        log(f"costs → {ep.dir / 'run.md'}")


def _run_steps(ep: Episode, yes: bool, ai_qa: bool) -> None:
    # fonts + face model download while the source downloads / transcribes
    assets_err: list[BaseException] = []

    def _assets() -> None:
        try:
            ensure_assets(ep.cfg.brand.font)
        except BaseException as e:  # surfaced after join; a daemon thread's traceback would be lost
            assets_err.append(e)

    assets = threading.Thread(target=_assets, daemon=True)
    assets.start()
    with costs.step("fetch"):
        fetch.fetch(ep)
        assets.join()
    if assets_err:
        raise SystemExit(f"asset download failed: {assets_err[0]}")
    with costs.step("probe"):
        probe.run(ep)
    # separate process: OpenCV and PyAV (faster-whisper) bundle clashing libavdevice copies on macOS
    with costs.step("transcribe"):
        if subprocess.run([sys.executable, "-m", "shorts", "transcribe", str(ep.dir)]).returncode != 0:
            raise typer.Exit(1)
    with costs.step("pick"):
        pick.run(ep)
    with costs.step("snap"):
        data = snap.run(ep)
    snap.print_summary(ep, data)

    if not yes:
        if not sys.stdin.isatty():
            console.print(f"\nStopped for approval. Render with: [cyan]shorts render {ep.dir}[/cyan] (or re-run with --yes)")
            raise typer.Exit(0)
        answer = console.input("\nType [bold]go[/bold] to render (anything else stops here): ").strip().lower()
        if answer != "go":
            console.print(f"Stopped. Edit {ep.work / 'clips.json'} if needed, then: [cyan]shorts render {ep.dir}[/cyan]")
            raise typer.Exit(0)
    _render_and_qa(ep, None, ai_qa)


def _render_and_qa(ep: Episode, only: list[int] | None, ai_qa: bool) -> None:
    with costs.step("render"):
        results = render.run(ep, only)
    with costs.step("qa"):
        ok = qa.run(ep, ai=ai_qa)
    failed = [r for r in results if r["status"] == "failed"]
    console.print(f"\n[bold]Done.[/bold] {len(results) - len(failed)}/{len(results)} clips in {ep.dir}")
    if failed or not ok:
        console.print("[yellow]Some clips need attention — see README.md and _qa/qa.json.[/yellow]")
        raise typer.Exit(1)


@app.command("probe")
def probe_cmd(episode: EpisodeDir, config: ConfigOpt = None, force: ForceOpt = False) -> None:
    """Step 1: ffprobe, 16 kHz WAV, chunk plan, stereo check, layout contact sheet."""
    _env_checks()
    ep = Episode.load(episode, config)
    _require_fetcher(ep.source)
    fetch.fetch(ep)
    probe.run(ep, force)


@app.command("transcribe")
def transcribe_cmd(episode: EpisodeDir, config: ConfigOpt = None, force: ForceOpt = False) -> None:
    """Step 2: faster-whisper with word timestamps → transcript.json/.srt/.txt."""
    transcribe.run(Episode.load(episode, config), force)


@app.command("pick")
def pick_cmd(episode: EpisodeDir, config: ConfigOpt = None, force: ForceOpt = False,
             clips: Annotated[Optional[int], typer.Option("--clips", "-k")] = None,
             model: Annotated[Optional[str], typer.Option("--model")] = None) -> None:
    """Step 3: Claude reads the whole transcript and picks clips → clips_raw.json."""
    ep = Episode.load(episode, config, _overrides(clips=clips, claude_model=model) or None)
    _require_picker(ep.cfg)
    costs.start(ep, "pick")
    try:
        with costs.step("pick"):
            pick.run(ep, force)
    finally:
        costs.finish()


@app.command("snap")
def snap_cmd(episode: EpisodeDir, config: ConfigOpt = None, force: ForceOpt = False) -> None:
    """Snap boundaries to words, pad, rank → clips.json, and print the approval table."""
    ep = Episode.load(episode, config)
    snap.print_summary(ep, snap.run(ep, force))


@app.command("render")
def render_cmd(episode: EpisodeDir, config: ConfigOpt = None,
               only: Annotated[Optional[str], typer.Option(help="Comma-separated ranks, e.g. 1,3")] = None,
               ai_qa: Annotated[bool, typer.Option("--ai-qa")] = False) -> None:
    """Steps 4–7: reframe, captions, encode and QA every clip in clips.json."""
    _env_checks()
    ep = Episode.load(episode, config)
    try:
        ranks = [int(x) for x in only.split(",")] if only else None
    except ValueError:
        raise SystemExit(f"--only expects comma-separated clip numbers, e.g. 1,3 (got: {only})") from None
    costs.start(ep, "render" + (f" --only {only}" if only else ""))
    try:
        _render_and_qa(ep, ranks, ai_qa)
    finally:
        costs.finish()
        log(f"costs → {ep.dir / 'run.md'}")


@app.command("qa")
def qa_cmd(episode: EpisodeDir, config: ConfigOpt = None,
           ai_qa: Annotated[bool, typer.Option("--ai-qa")] = False) -> None:
    """Step 7: re-run QA checks and rewrite README.md."""
    require_tools()  # ffprobe checks + ffmpeg frame grabs
    ep = Episode.load(episode, config)
    if ai_qa:
        costs.start(ep, "qa --ai-qa")
    try:
        with costs.step("qa"):
            ok = qa.run(ep, ai=ai_qa)
    finally:
        costs.finish()
    if not ok:
        raise typer.Exit(1)


@app.command()
def doctor() -> None:
    """Check the environment: ffmpeg, libass, uv, claude, fonts, models. Exits 1 when a required tool is missing."""
    from .assets import font_path, yunet_path

    required = ("ffmpeg", "ffprobe", "uv", "uvx", "claude")
    for tool in required:
        console.print(f"{'✅' if shutil.which(tool) else '❌'} {tool}")
    logged_in = None
    if shutil.which("claude"):
        try:
            proc = subprocess.run(["claude", "auth", "status"], capture_output=True, text=True, timeout=30)
            logged_in = proc.returncode == 0 and bool(json.loads(proc.stdout).get("loggedIn"))
            console.print(f"{'✅' if logged_in else '❌'} claude logged in")
        except Exception:  # older CLI without `auth status`, timeout…
            console.print("ℹ️  claude login not verified (`claude auth status` unavailable)")
    if shutil.which("ffmpeg"):
        console.print(f"{'✅' if has_libass() else 'ℹ️ '} libass subtitles filter "
                      f"{'' if has_libass() else '(missing: Pillow overlay fallback is used)'}")
    console.print(f"{'✅' if os.environ.get('HF_TOKEN') else 'ℹ️ '} HF_TOKEN (diarization not used either way)")
    for label, check in (("font", lambda: font_path()), ("face model", yunet_path)):
        try:
            console.print(f"✅ {label} {check()}")
        except Exception as e:  # network errors etc.
            warn(f"{label}: {e}")
    require_tools(*required)  # exits with install hints if anything above was ❌
    if logged_in is False:
        raise SystemExit("claude is installed but not logged in — run `claude` once and sign in")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
