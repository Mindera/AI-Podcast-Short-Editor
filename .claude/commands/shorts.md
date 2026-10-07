---
description: Cut branded vertical shorts from a podcast video (Drive link or path)
argument-hint: <video-url-or-path> [--host NAME] [--guest NAME] [--clips N] [...]
---
Follow AGENTS.md in this repo to cut shorts from: $ARGUMENTS

Run `./shorts $ARGUMENTS` in the background (no `--yes` unless I passed it), then show me the approval table,
the over-60s clips with suggested shorter cuts, the runners-up and the caption corrections, and wait for my "go"
before running `./shorts render <episode-dir>`. After rendering, look at `_qa/contact_sheet.jpg` yourself and fix
anything wrong before reporting.
