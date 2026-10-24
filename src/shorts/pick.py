"""Step 3: the judgement step. Headless `claude -p` reads the whole transcript and picks clips.

Set SHORTS_PICKER=stub to skip the LLM (evenly spaced clips, for smoke tests).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from . import costs
from .cache import atomic_write_json, file_sig, is_fresh, make_key, mark_done, read_json
from .episode import Episode
from .util import log, warn

PROMPT_VERSION = "pick-v1"


class Boundary(BaseModel):
    start_phrase: str
    start: float
    end_phrase: str
    end: float


class PickedClip(Boundary):
    hook_text: str
    why_it_works: str
    virality_score: float = Field(ge=0, le=10)
    title: str
    key_word: str = ""
    description: str
    hashtags: list[str]
    suggested_cover_timestamp: float
    speakers: list[tuple[float, float, Literal["host", "guest"]]]
    shorter_cut: Boundary | None = None


class Layout(BaseModel):
    description: str = ""
    host_side: Literal["left", "right", "unknown"] = "unknown"


class PickResult(BaseModel):
    layout: Layout = Field(default_factory=Layout)
    clips: list[PickedClip]
    runners_up: list[PickedClip] = Field(default_factory=list)
    corrections: dict[str, str] = Field(default_factory=dict)


SCHEMA_EXAMPLE = {
    "layout": {"description": "Wide two-shot plus a close-up camera per person; host sits left.", "host_side": "left"},
    "clips": [{
        "start_phrase": "first 4-8 words of the clip, verbatim",
        "start": 812.4,
        "end_phrase": "last 4-8 words of the clip, verbatim",
        "end": 851.9,
        "hook_text": "the opening line that hooks the viewer",
        "why_it_works": "one sentence",
        "virality_score": 8.5,
        "title": "Sentence case title up to 60 chars",
        "key_word": "one word from the title to highlight on the cover",
        "description": "Two or three lines\nfor the post caption.",
        "hashtags": ["#Mindera", "#SoftwareEngineering", "#Leadership", "#Podcast", "#AI"],
        "suggested_cover_timestamp": 815.0,
        "speakers": [[812.4, 820.1, "host"], [820.1, 851.9, "guest"]],
        "shorter_cut": None,
    }],
    "runners_up": ["same shape as clips, 3-4 items"],
    "corrections": {"misheard form exactly as in transcript": "Correct Spelling"},
}


def build_prompt(ep: Episode, transcript_txt: str, layout: dict) -> str:
    cfg, brand = ep.cfg, ep.cfg.brand
    mic_note = ""
    if "| mic " in transcript_txt:
        mic_note = ("\nSome lines carry `mic L` / `mic R`: the host and guest are on separate microphones, so the tag "
                    "shows who is talking. Work out which mic is the host from who asks the questions.")
    sheet = ep.work / "layout_sheet.jpg"
    layout_note = (
        f"\n\nCAMERA LAYOUT: use the Read tool to look at the contact sheet image `{sheet}` (10 frames sampled across "
        f"the episode, faces boxed in yellow). Face detections per frame (x/y as fractions of the frame): "
        f"{json.dumps(layout.get('frames', []))}. Describe the layout (wide two-shot, separate cameras, close-ups) "
        f"and, only if you can tell who is the host (e.g. the one who asks the questions in a close-up that matches "
        f"a question in the transcript), say which side the host sits on. Otherwise answer \"unknown\"."
        if sheet.exists() else ""
    )
    return f"""You are a senior short-form video editor for {brand.name}. Below is the FULL transcript of a podcast
episode (language: {cfg.language}). Host: {cfg.host}. Guest: {cfg.guest}.
Each line is `[start-end | m:ss] text`, times in seconds from the start of the episode.{mic_note}

TASK: pick the {cfg.clips} best moments to cut into vertical shorts (Reels / TikTok / YouTube Shorts), plus 3-4
runners-up. Read the whole transcript before choosing.

RULES
- Each clip must make sense on its own, without the earlier context of the conversation.
- Length: {cfg.min_clip_s:.0f}-{cfg.ideal_max_s:.0f} s is ideal, never over {cfg.max_clip_s:.0f} s. Start and end on a
  complete sentence or thought.
- Hook: the first 3 seconds must grab attention (a bold claim, a question, a funny line). Prefer stories, hot takes,
  practical advice, quotable lines and emotional moments.
- Skip: the cold-open teaser (it repeats later content), bios and intros, and anything that depends on earlier context.
- Clips must not overlap each other.
- Boundaries: `start_phrase` is the first 4-8 words of the clip and `end_phrase` the last 4-8 words, copied VERBATIM
  from the transcript (keep its misspellings). `start` / `end` are approximate seconds.
- If a clip is over {cfg.ideal_max_s:.0f} s, propose a shorter cut inside it in `shorter_cut` (same boundary fields);
  otherwise null.
- `speakers`: the turns `[start, end, "host"|"guest"]` in source seconds covering the whole clip, from who asks and
  who answers. This decides who is on screen, so be precise about turn changes.
- `title` at most 60 characters, sentence case, no emoji. `key_word`: one word of the title to highlight.
  `description`: 2-3 lines. `hashtags`: always "{brand.hashtags[0] if brand.hashtags else '#' + brand.name}" plus 4-7
  that fit the topic.
- `suggested_cover_timestamp`: a moment inside the clip (source seconds) where the speaker is mid-thought and expressive.
- Spelling: names and terms are often misheard, especially in the host's questions. Correct spellings: {", ".join(cfg.terms)}.
  Collect EVERY misheard form you notice into `corrections` (misheard text exactly as it appears → correct form).
- Order `clips` best first. `virality_score` is 0-10.{layout_note}

OUTPUT: only a JSON object, no prose, no code fences, with exactly this shape:
{json.dumps(SCHEMA_EXAMPLE, indent=2)}

TRANSCRIPT
{transcript_txt}"""


def _extract_json(text: str) -> dict:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    a, b = text.find("{"), text.rfind("}")
    if a < 0 or b < a:
        raise ValueError("no JSON object in the reply")
    return json.loads(text[a:b + 1])


def call_claude(ep: Episode, prompt: str, purpose: str = "pick") -> str:
    cfg = ep.cfg
    if not shutil.which(cfg.claude_bin):
        raise SystemExit(f"`{cfg.claude_bin}` not found. Install Claude Code, or set SHORTS_PICKER=stub.")
    cmd = [cfg.claude_bin, "-p", "--output-format", "json", "--allowedTools", "Read", "--add-dir", str(ep.work)]
    if cfg.claude_model:
        cmd += ["--model", cfg.claude_model]
    try:
        proc = subprocess.run(cmd, input=prompt, capture_output=True, text=True, cwd=ep.work,
                              timeout=cfg.claude_timeout_s)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"claude -p gave no reply within {cfg.claude_timeout_s}s "
                           "(raise claude_timeout_s in the config, or try --model sonnet)") from None
    if proc.returncode != 0:
        raise RuntimeError(f"claude -p failed ({proc.returncode}): {proc.stderr.strip()[-800:]}")
    try:
        envelope = json.loads(proc.stdout)
    except json.JSONDecodeError:
        raise RuntimeError("claude -p didn't return JSON — is the claude CLI logged in? (run `claude` once) "
                           f"stdout: {proc.stdout.strip()[:300]!r}") from None
    costs.record_claude(purpose, envelope)
    if envelope.get("is_error"):
        raise RuntimeError(f"claude -p returned an error: {envelope.get('result')}")
    return envelope.get("result", "")


def stub_pick(ep: Episode, transcript: dict) -> PickResult:
    """Deterministic picker for tests: evenly spaced ~25 s windows on segment boundaries."""
    segs = transcript["segments"]
    dur = transcript["duration"]
    clips = []
    for i in range(ep.cfg.clips):
        target = dur * (i + 0.5) / ep.cfg.clips
        first = min(segs, key=lambda s: abs(s["start"] - target))
        last = first
        for s in segs:
            if s["start"] >= first["start"] and s["end"] - first["start"] <= 30:
                last = s
        words_a, words_b = first["text"].split(), last["text"].split()
        clips.append(PickedClip(
            start_phrase=" ".join(words_a[:5]), start=first["start"],
            end_phrase=" ".join(words_b[-5:]), end=last["end"],
            hook_text=first["text"].strip(), why_it_works="stub picker", virality_score=10 - i,
            title=f"Stub clip {i + 1}", key_word="clip", description="Stub description.",
            hashtags=["#Mindera", "#Podcast"], suggested_cover_timestamp=first["start"] + 1,
            speakers=[(first["start"], last["end"], "host")],
        ))
    return PickResult(clips=clips)


def run(ep: Episode, force: bool = False) -> PickResult:
    work, cfg = ep.work, ep.cfg
    out = work / "clips_raw.json"
    txt_path = work / "transcript.txt"
    picker = os.environ.get("SHORTS_PICKER", "claude")
    key = make_key(PROMPT_VERSION, picker, file_sig(txt_path), cfg.host, cfg.guest, cfg.clips, cfg.terms,
                   cfg.claude_model, cfg.min_clip_s, cfg.ideal_max_s, cfg.max_clip_s, cfg.language, cfg.brand.hashtags)
    if not force and is_fresh(work, "pick", key, [out]):
        return PickResult.model_validate(read_json(out))

    if picker == "stub":
        result = stub_pick(ep, read_json(work / "transcript.json"))
    else:
        layout = read_json(work / "layout.json") if (work / "layout.json").exists() else {}
        prompt = build_prompt(ep, txt_path.read_text(), layout)
        (work / "pick_prompt.txt").write_text(prompt)
        log(f"pick: asking Claude (claude -p{' --model ' + cfg.claude_model if cfg.claude_model else ''}) "
            f"to read the whole transcript and choose {cfg.clips} clips…")
        reply = call_claude(ep, prompt)
        try:
            result = PickResult.model_validate(_extract_json(reply))
        except (ValueError, ValidationError) as e:
            warn(f"pick: reply didn't validate ({str(e)[:200]}); retrying once")
            reply = call_claude(ep, prompt + f"\n\nYour previous reply was invalid: {e}\nReturn ONLY the JSON object.",
                                "pick (retry)")
            result = PickResult.model_validate(_extract_json(reply))
        (work / "pick_reply.txt").write_text(reply)
    if len(result.clips) < cfg.clips:
        warn(f"pick: asked for {cfg.clips} clips, got {len(result.clips)}")
    atomic_write_json(out, result.model_dump())
    mark_done(work, "pick", key)
    return result
