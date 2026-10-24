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
    """The pick step runs an agent CLI long after the slow steps; fail now if it's missing."""
    from .agent import require_agent

    require_agent(cfg)


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
    picker: Annotated[Optional[str], typer.Option(
        "--picker", help="Agent CLI that picks clips: claude (default) | codex | copilot | gemini | stub.")] = None,
    model: Annotated[Optional[str], typer.Option(
        "--model", help="Model for the picker agent (e.g. opus, gpt-5-codex, gemini-2.5-pro).")] = None,
    whisper_model: Annotated[Optional[str], typer.Option(help="faster-whisper model (default medium).")] = None,
    jobs: Annotated[Optional[int], typer.Option(help="Parallel renders (default cores/6).")] = None,
    drive_cookies: Annotated[Optional[str], typer.Option(
        "--drive-cookies", help="Browser whose Google login to use for a private Drive file (e.g. chrome).")] = None,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip the approval pause and render straight away.")] = False,
    ai_qa: Annotated[bool, typer.Option("--ai-qa", help="Ask the picker agent to review the QA contact sheet.")] = False,
) -> None:
    """Do everything: fetch → probe → transcribe → pick → snap → [approve] → render → qa."""
    _env_checks()
    cfg = load_config(config, _overrides(out=out, host=host, guest=guest, clips=clips, language=language, terms=terms,
                                         host_side=host_side, picker=picker, agent_model=model,
                                         whisper_model=whisper_model, jobs=jobs, drive_cookies=drive_cookies))
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
             picker: Annotated[Optional[str], typer.Option("--picker")] = None,
             model: Annotated[Optional[str], typer.Option("--model")] = None) -> None:
    """Step 3: the picker agent reads the whole transcript and picks clips → clips_raw.json."""
    ep = Episode.load(episode, config, _overrides(clips=clips, picker=picker, agent_model=model) or None)
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
def doctor(config: ConfigOpt = None) -> None:
    """Check the environment: ffmpeg, libass, uv, the picker agent CLI, fonts, models. Exits 1 when something
    required is missing."""
    from .agent import AGENTS, agent_bin, picker_name
    from .assets import font_path, yunet_path

    cfg = load_config(config)
    picker = picker_name(cfg)
    for tool in ("ffmpeg", "ffprobe", "uv", "uvx"):
        console.print(f"{'✅' if shutil.which(tool) else '❌'} {tool}")
    for a in AGENTS:
        b = agent_bin(cfg) if a == picker else a
        have = bool(shutil.which(b))
        mark = "✅" if have else ("❌" if a == picker else "ℹ️ ")
        console.print(f"{mark} {a}{f' ({b})' if b != a else ''}"
                      f"{' ← picker' if a == picker else ''}{'' if have else ' not installed'}")
    logged_in = None
    if picker == "claude" and shutil.which(agent_bin(cfg)):
        try:
            proc = subprocess.run([agent_bin(cfg), "auth", "status"], capture_output=True, text=True, timeout=30)
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
    required = ["ffmpeg", "ffprobe", "uv", "uvx"] + ([agent_bin(cfg)] if picker != "stub" else [])
    require_tools(*required)  # exits with install hints if anything above was ❌
    if logged_in is False:
        raise SystemExit("claude is installed but not logged in — run `claude` once and sign in")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
