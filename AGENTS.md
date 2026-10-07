# AGENTS.md — running and changing `shorts`

Instructions for AI coding assistants (Claude Code, Codex, Copilot agent mode, Cursor, Gemini CLI…) asked to cut
shorts from a podcast with this repo.

## Running it for someone

1. **Collect the inputs.** You need the **video** (Google Drive link, other URL, or local path). Ask only if missing.
   Everything else has defaults; map what the user says to flags:

   | They say | Flag |
   |---|---|
   | host / guest names | `--host "…" --guest "…"` |
   | "8 clips", "give me 5" | `--clips N` |
   | "in Portuguese" | `--lang pt` (`auto` to detect) |
   | names/terms to spell right | `--terms "A,B,C"` |
   | "host sits on the right" | `--host-side right` |
   | "save to …" | `--out …` |
   | "don't ask me, just render" | `--yes` |
   | "use Opus to pick" | `--model opus` |

2. **Check the environment:** `./shorts doctor`. Needs `ffmpeg`/`ffprobe`, `uv`, and the `claude` CLI (logged in).
   Never install system packages or use `sudo`; everything Python lives in the project's `uv` venv (Python 3.12).
   Missing libass is fine (captions fall back to Pillow overlays).

3. **Run it** in the background — transcription runs at roughly half real-time, so a 1-hour episode takes ~30 min:
   ```bash
   ./shorts "<url-or-path>" --host "…" --guest "…" --clips 8
   ```
   Without `--yes` and without a TTY, the run **stops after picking** and prints the ranked table. That's the
   approval gate: show the user the table (score, start, length, title, opening line), clips over 60 s with their
   suggested shorter cut, the runners-up and the caption corrections, then **wait for their "go"**. Then:
   ```bash
   ./shorts render "<episode-dir>"
   ```
   A healthy long run always has either an `ffmpeg` child process or new files appearing under `_work/`. If neither
   for several minutes, it's hung. ffmpeg calls have a built-in stall watchdog.

4. **Check the result.** Open `<episode-dir>/_qa/contact_sheet.jpg` and look at it: is the framed person the one
   speaking, are captions inside the safe zones and off faces, no flicker? Read `_qa/qa.json`. Fix (edit
   `_work/clips.json`, e.g. speaker turns or times) and `./shorts render <dir> --only N` before reporting.

5. **Report** the output folder, the `README.md` table, any QA warnings, and what you assumed.

## Rules

- Never modify or delete the source video. Never upload anything.
- Superseded outputs are moved to `_work/_superseded/<timestamp>/`, never deleted. Keep it that way.
- Every step caches on its inputs; re-running only redoes what changed. Use `--force` on a step to redo it.

## Changing the code

- Package: `src/shorts/`, one module per step (`probe`, `transcribe`, `pick`, `snap`, `reframe`, `captions`,
  `render`, `qa`), CLI in `cli.py`, settings in `config.py`.
- Tests: `uv run --group dev pytest`. The run-merging loop in `reframe.merge_short_runs` must stay provably
  terminating; `tests/test_merge.py` fuzzes it with random label sequences.
- Bump the version string in a step's cache key (e.g. `"captions-v1"`) when you change its output format, so old
  caches are invalidated.
- End-to-end without Claude: `SHORTS_PICKER=stub ./shorts <video> --clips 2 --yes`.
