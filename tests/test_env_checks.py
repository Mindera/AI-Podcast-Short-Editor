"""The missing-dependency checks fail fast, name every missing tool, and include install hints."""

import pytest

from shorts import util
from shorts.transcribe import _require_speech


def test_require_tools_passes_when_present(monkeypatch):
    monkeypatch.setattr(util.shutil, "which", lambda t: f"/usr/bin/{t}")
    util.require_tools("ffmpeg", "claude")  # no exit


def test_require_tools_lists_every_missing_tool_with_hints(monkeypatch):
    monkeypatch.setattr(util.shutil, "which", lambda t: None)
    with pytest.raises(SystemExit) as e:
        util.require_tools("ffmpeg", "uvx", "claude")
    msg = str(e.value)
    assert "ffmpeg, uvx, claude" in msg
    assert "brew install ffmpeg" in msg
    assert "astral.sh/uv" in msg
    assert "@anthropic-ai/claude-code" in msg
    assert "shorts doctor" in msg


def test_require_tools_defaults_to_ffmpeg(monkeypatch):
    monkeypatch.setattr(util.shutil, "which", lambda t: None if t == "ffprobe" else f"/usr/bin/{t}")
    with pytest.raises(SystemExit, match="ffprobe"):
        util.require_tools()


def test_no_speech_transcript_is_a_clear_error():
    with pytest.raises(SystemExit, match="no speech"):
        _require_speech({"segments": []})
    assert _require_speech({"segments": [{"text": "hi"}]})["segments"]
