"""Headless calls to a coding-agent CLI for the judgement steps (pick, AI QA).

The default agent is Claude Code; Codex, Copilot and Gemini work too. Choose with `picker` in shorts.toml,
`--picker`, or the SHORTS_PICKER env var ("stub" skips the LLM entirely); `picker_bin` overrides the executable.
Only Claude Code reports token costs back, so other agents are logged with time but no spend.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from . import costs
from .config import Config
from .util import require_tools

AGENTS = ("claude", "codex", "copilot", "gemini")


def picker_name(cfg: Config) -> str:
    name = os.environ.get("SHORTS_PICKER") or cfg.picker
    if name not in (*AGENTS, "stub"):
        raise SystemExit(f"unknown picker {name!r} — use one of: {', '.join(AGENTS)}, stub")
    return name


def agent_bin(cfg: Config) -> str:
    return cfg.picker_bin or picker_name(cfg)


def require_agent(cfg: Config) -> None:
    """Fail before the slow steps if the configured agent CLI is missing."""
    if picker_name(cfg) != "stub":
        require_tools(agent_bin(cfg))


def _command(name: str, bin: str, model: str | None, workdir: Path, prompt: str,
             last_msg: Path) -> tuple[list[str], str | None]:
    """argv + stdin text for one non-interactive call. Each CLI runs with the episode workdir as cwd,
    so file-reading tools can reach the contact sheet; the prompt goes on stdin where the CLI supports it."""
    model_args = ["--model", model] if model else []
    if name == "claude":
        return [bin, "-p", "--output-format", "json", "--allowedTools", "Read", "--add-dir", str(workdir),
                *model_args], prompt
    if name == "codex":
        return [bin, "exec", "--skip-git-repo-check", "--sandbox", "read-only", "--cd", str(workdir),
                "--output-last-message", str(last_msg), *model_args, "-"], prompt
    if name == "copilot":
        return [bin, "-p", prompt, *model_args], None
    if name == "gemini":
        return [bin, *model_args, "-p", prompt], None
    raise ValueError(name)


def run_agent(cfg: Config, prompt: str, workdir: Path, purpose: str, timeout_s: int | None = None) -> str:
    """One prompt in, the agent's reply text out. Raises RuntimeError with a clear reason on any failure."""
    name = picker_name(cfg)
    bin = agent_bin(cfg)
    if not shutil.which(bin):
        raise SystemExit(f"`{bin}` not found. Install it, choose another agent with --picker "
                         f"({', '.join(AGENTS)}), or set SHORTS_PICKER=stub.")
    timeout_s = timeout_s or cfg.agent_timeout_s
    t0 = time.time()
    with tempfile.TemporaryDirectory(prefix="shorts-agent-") as tmp:
        last_msg = Path(tmp) / "last_message.txt"
        cmd, stdin = _command(name, bin, cfg.agent_model, workdir, prompt, last_msg)
        try:
            proc = subprocess.run(cmd, input=stdin, capture_output=True, text=True, cwd=workdir, timeout=timeout_s)
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"{bin} gave no reply within {timeout_s}s "
                               "(raise agent_timeout_s in the config, or try a faster model)") from None
        if proc.returncode != 0:
            raise RuntimeError(f"{bin} failed ({proc.returncode}): {proc.stderr.strip()[-800:]}")
        if name == "claude":
            try:
                envelope = json.loads(proc.stdout)
            except json.JSONDecodeError:
                raise RuntimeError("claude -p didn't return JSON — is the claude CLI logged in? (run `claude` once) "
                                   f"stdout: {proc.stdout.strip()[:300]!r}") from None
            costs.record_claude(purpose, envelope)
            if envelope.get("is_error"):
                raise RuntimeError(f"claude -p returned an error: {envelope.get('result')}")
            return envelope.get("result", "")
        text = last_msg.read_text() if name == "codex" and last_msg.exists() else proc.stdout
    costs.record_agent(purpose, name, time.time() - t0)
    if not text.strip():
        raise RuntimeError(f"{bin} returned an empty reply (stderr: {proc.stderr.strip()[-300:]})")
    return text
