from pathlib import Path

import pytest

from shorts import costs
from shorts.cache import read_json


class FakeEp:
    def __init__(self, root: Path):
        self.dir = root
        self.source = "episode.mp4"
        self.work.mkdir(parents=True)

    @property
    def work(self) -> Path:
        return self.dir / "_work"


ENVELOPE = {
    "total_cost_usd": 0.42,
    "duration_ms": 61000,
    "duration_api_ms": 58000,
    "num_turns": 2,
    "is_error": False,
    "session_id": "abc",
    "usage": {"input_tokens": 10, "output_tokens": 3000, "cache_read_input_tokens": 5000,
              "cache_creation_input_tokens": 40000},
    "modelUsage": {"claude-sonnet-5-5": {"inputTokens": 10, "outputTokens": 3000, "cacheReadInputTokens": 5000,
                                         "cacheCreationInputTokens": 40000, "costUSD": 0.42}},
}


def test_runs_accumulate_into_ledger_and_report(tmp_path):
    ep = FakeEp(tmp_path / "ep")
    costs.start(ep, "run")
    with costs.step("pick"):
        costs.record_claude("pick", ENVELOPE)
    costs.finish()

    costs.start(ep, "render")
    with pytest.raises(RuntimeError):
        with costs.step("render"):
            raise RuntimeError("boom")
    costs.record_claude("AI QA", {**ENVELOPE, "total_cost_usd": 0.08,
                                  "modelUsage": {"claude-sonnet-5-5": {**ENVELOPE["modelUsage"]["claude-sonnet-5-5"],
                                                                       "costUSD": 0.08}}})
    costs.finish()

    ledger = read_json(ep.work / "costs.json")
    assert [r["command"] for r in ledger["runs"]] == ["run", "render"]
    assert ledger["runs"][1]["steps"][0]["status"] == "failed"

    md = (ep.dir / "run.md").read_text()
    assert "**$0.5000** over 2 call(s)" in md
    assert "| claude-sonnet-5-5 | 2 |" in md
    assert "| render |" in md and "failed" in md
    assert "40,000" in md


def test_record_without_run_is_noop(tmp_path):
    costs._run = None
    costs.record_claude("pick", ENVELOPE)
    with costs.step("x"):
        pass
    costs.finish()
