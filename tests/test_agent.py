"""Agent-CLI selection is agnostic: claude, codex, copilot and gemini all build a valid headless call."""

from pathlib import Path

import pytest

from shorts.agent import AGENTS, _command, agent_bin, picker_name
from shorts.config import load_config


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # don't pick up a developer's ./shorts.toml
    monkeypatch.delenv("SHORTS_PICKER", raising=False)
    return load_config()


def test_claude_is_the_default(cfg):
    assert picker_name(cfg) == "claude"
    assert agent_bin(cfg) == "claude"


def test_env_var_overrides_config(cfg, monkeypatch):
    monkeypatch.setenv("SHORTS_PICKER", "codex")
    assert picker_name(cfg) == "codex"
    monkeypatch.setenv("SHORTS_PICKER", "chatgpt")
    with pytest.raises(SystemExit, match="unknown picker"):
        picker_name(cfg)


def test_every_agent_builds_a_command(tmp_path):
    for name in AGENTS:
        cmd, stdin = _command(name, name, "some-model", tmp_path, "PROMPT", tmp_path / "last.txt")
        assert cmd[0] == name
        assert "some-model" in cmd
        # the prompt reaches the agent exactly once: on stdin or as an argument
        assert (stdin == "PROMPT") != ("PROMPT" in cmd)


def test_legacy_claude_keys_still_load(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = load_config(base={"claude_model": "opus", "claude_timeout_s": 7, "claude_bin": "/opt/claude"})
    assert cfg.agent_model == "opus"
    assert cfg.agent_timeout_s == 7
    assert cfg.picker_bin == "/opt/claude"
    # the old default value doesn't shadow another picker's executable
    cfg = load_config(base={"claude_bin": "claude", "picker": "codex"})
    assert cfg.picker_bin is None
