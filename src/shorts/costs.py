"""Cost ledger: Claude spend per call and compute time per step, written to `_work/costs.json` and `run.md`.

One CLI invocation is one "run". The approval stop splits a full job into two runs (`run`, then `render`), so
`run.md` shows every run in the episode plus the episode total.
"""

from __future__ import annotations

import os
import platform
import resource
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from .cache import atomic_write_json, atomic_write_text, read_json
from .util import fmt_ts

_ep = None  # the Episode being costed in this process, set by start()
_run: dict[str, Any] | None = None


def _ledger_path(ep) -> Path:
    return ep.work / "costs.json"


def _cpu_s() -> float:
    """CPU seconds used so far by this process and every child it has waited for (ffmpeg, whisper, render workers)."""
    me, kids = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)
    return me.ru_utime + me.ru_stime + kids.ru_utime + kids.ru_stime


def start(ep, command: str) -> None:
    """Begin a run for this CLI invocation. Steps and Claude calls recorded afterwards belong to it."""
    global _ep, _run
    _ep = ep
    _run = {
        "id": f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}",
        "command": command,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "finished": None,
        "host": {"machine": platform.machine(), "system": platform.system(), "cpus": os.cpu_count()},
        "steps": [],
        "claude": [],
    }
    _save()


@contextmanager
def step(name: str) -> Iterator[None]:
    """Time one pipeline step (wall clock and CPU, children included). Recorded even if the step fails."""
    t0, c0 = time.time(), _cpu_s()
    status = "ok"
    try:
        yield
    except BaseException:
        status = "failed"
        raise
    finally:
        if _run is not None:
            _run["steps"].append({"step": name, "wall_s": round(time.time() - t0, 2),
                                  "cpu_s": round(_cpu_s() - c0, 2), "status": status})
            _save()


def record_claude(purpose: str, envelope: dict) -> None:
    """Record one `claude -p --output-format json` result envelope."""
    if _run is None:
        return
    usage = envelope.get("usage") or {}
    models = {
        m: {"input": u.get("inputTokens", 0), "output": u.get("outputTokens", 0),
            "cache_read": u.get("cacheReadInputTokens", 0), "cache_write": u.get("cacheCreationInputTokens", 0),
            "cost_usd": u.get("costUSD", 0.0)}
        for m, u in (envelope.get("modelUsage") or {}).items()
    }
    _run["claude"].append({
        "purpose": purpose,
        "models": models,
        "input_tokens": usage.get("input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
        "cache_read_tokens": usage.get("cache_read_input_tokens", 0),
        "cache_write_tokens": usage.get("cache_creation_input_tokens", 0),
        "cost_usd": envelope.get("total_cost_usd") or 0.0,
        "duration_s": round((envelope.get("duration_ms") or 0) / 1000, 1),
        "api_s": round((envelope.get("duration_api_ms") or 0) / 1000, 1),
        "turns": envelope.get("num_turns", 0),
        "ok": not envelope.get("is_error"),
        "session_id": envelope.get("session_id", ""),
    })
    _save()


def finish() -> None:
    if _run is not None and _run["finished"] is None:
        _run["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        _save()


def _save() -> None:
    path = _ledger_path(_ep)
    ledger = read_json(path) if path.exists() else {"runs": []}
    ledger["runs"] = [r for r in ledger["runs"] if r["id"] != _run["id"]] + [_run]
    atomic_write_json(path, ledger)
    write_report(_ep, ledger)


# ---- run.md -----------------------------------------------------------------------------------------------------

def _dur(s: float) -> str:
    s = float(s)
    if s < 60:
        return f"{s:.1f}s"
    m, sec = divmod(int(round(s)), 60)
    return f"{m // 60}h {m % 60:02d}m {sec:02d}s" if m >= 60 else f"{m}m {sec:02d}s"


def _usd(x: float) -> str:
    return f"${x:,.4f}" if x < 1 else f"${x:,.2f}"


def _n(x: int) -> str:
    return f"{x:,}"


def _episode_facts(ep) -> dict[str, Any]:
    facts: dict[str, Any] = {"source_s": None, "clips": [], "out_bytes": 0}
    probe_p = ep.work / "probe.json"
    if probe_p.exists():
        try:
            facts["source_s"] = float(read_json(probe_p)["format"]["duration"]) or None
        except (ValueError, TypeError, KeyError):
            pass
    clips_p = ep.work / "clips.json"
    if clips_p.exists():
        for c in read_json(clips_p).get("clips", []):
            mp4 = ep.dir / f"{c['slug']}.mp4"
            if mp4.exists():
                facts["clips"].append(c.get("duration", 0.0))
                facts["out_bytes"] += mp4.stat().st_size
    return facts


def _claude_rows(calls: list[dict]) -> list[str]:
    rows = ["| Purpose | Model | Input | Cache write | Cache read | Output | Turns | Time | Cost |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for c in calls:
        model = ", ".join(c["models"]) or "?"
        purpose = c["purpose"] + ("" if c["ok"] else " ⚠️ error")
        rows.append(f"| {purpose} | {model} | {_n(c['input_tokens'])} | {_n(c['cache_write_tokens'])} | "
                    f"{_n(c['cache_read_tokens'])} | {_n(c['output_tokens'])} | {c['turns']} | "
                    f"{_dur(c['duration_s'])} | {_usd(c['cost_usd'])} |")
    return rows


def write_report(ep, ledger: dict) -> None:
    runs = ledger["runs"]
    calls = [c for r in runs for c in r["claude"]]
    total_usd = sum(c["cost_usd"] for c in calls)
    total_wall = sum(s["wall_s"] for r in runs for s in r["steps"])
    total_cpu = sum(s["cpu_s"] for r in runs for s in r["steps"])
    facts = _episode_facts(ep)
    n_clips = len(facts["clips"])

    by_model: dict[str, dict[str, float]] = {}
    for c in calls:
        for m, u in c["models"].items():
            t = by_model.setdefault(m, {"calls": 0, "input": 0, "cache_write": 0, "cache_read": 0, "output": 0,
                                        "cost_usd": 0.0})
            t["calls"] += 1
            for k in ("input", "cache_write", "cache_read", "output", "cost_usd"):
                t[k] += u[k]

    lines = [f"# Run cost — {ep.dir.name}", "",
             f"Source: `{ep.source}`  ",
             f"Updated: {time.strftime('%Y-%m-%d %H:%M:%S')}", "",
             "## Summary", "",
             "| | |", "|---|---|",
             f"| Claude spend (all runs) | **{_usd(total_usd)}** over {len(calls)} call(s) |",
             f"| Pipeline time (all runs) | {_dur(total_wall)} wall · {_dur(total_cpu)} CPU |"]
    if facts["source_s"]:
        lines.append(f"| Source length | {fmt_ts(facts['source_s'])} "
                     f"({_usd(total_usd / (facts['source_s'] / 3600))} Claude per source hour) |")
    if n_clips:
        lines += [f"| Clips rendered | {n_clips} · {_dur(sum(facts['clips']))} of video · "
                  f"{facts['out_bytes'] / 1e6:,.1f} MB |",
                  f"| Per clip | {_usd(total_usd / n_clips)} Claude · {_dur(total_wall / n_clips)} wall |"]
    lines += ["",
              "> Claude costs are what `claude -p` reports, priced at API list rates. On a Claude subscription "
              "(Pro/Max/Team) the same tokens count toward your plan's usage limits instead of being billed. "
              "Transcription, face tracking and encoding run locally, so their cost is machine time only.", ""]

    if by_model:
        lines += ["## Claude by model", "",
                  "| Model | Calls | Input | Cache write | Cache read | Output | Cost |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for m, t in sorted(by_model.items()):
            lines.append(f"| {m} | {t['calls']} | {_n(t['input'])} | {_n(t['cache_write'])} | "
                         f"{_n(t['cache_read'])} | {_n(t['output'])} | {_usd(t['cost_usd'])} |")
        lines.append("")

    lines += ["## Runs", ""]
    for r in reversed(runs):
        usd = sum(c["cost_usd"] for c in r["claude"])
        wall = sum(s["wall_s"] for s in r["steps"])
        h = r["host"]
        state = f"finished {r['finished'][11:]}" if r["finished"] else "unfinished (stopped or still running)"
        lines += [f"### `shorts {r['command']}` — {r['started'].replace('T', ' ')}", "",
                  f"{_usd(usd)} Claude · {_dur(wall)} wall · {state} · "
                  f"{h['system']} {h['machine']}, {h['cpus']} CPUs", ""]
        if r["steps"]:
            lines += ["| Step | Wall | CPU | Status |", "|---|---:|---:|---|"]
            for s in r["steps"]:
                lines.append(f"| {s['step']} | {_dur(s['wall_s'])} | {_dur(s['cpu_s'])} | {s['status']} |")
            lines.append("")
        if r["claude"]:
            lines += _claude_rows(r["claude"]) + [""]
        else:
            lines += ["No Claude calls in this run (cached or stub picks, or AI QA off).", ""]

    atomic_write_text(ep.dir / "run.md", "\n".join(lines))
