# Notes

Coding agent: Claude Code · Opus 5.5

The line above is read by `bench submit` and shown on the leaderboard next to each submission.
Keep exactly one such line, starting the line, and change its value to the coding agent that
builds this agent: a short name such as `Claude Code · Fable 5`, `Codex`, `Cursor` or `by hand`
(1-40 characters: letters, digits, spaces and `. _ + - / ( ) ·`).

- **Model:** Inkling 2 Mini, the leaderboard's fixed model, reached only through `llm.chat`.
- **How it was built:** Claude Code (Claude Opus 5.5), starting from the base solution.

## What this agent is

A closed loop around the model, with no task-specific content:

1. One call with the task, the API docs, the current `robot.observe()` state and a general
   system prompt (think briefly, one fenced Python block, print measured poses, verify with
   read functions, place objects by computing heights from extents, never drop from height).
2. Run the code. A reply with no code (e.g. all tokens spent thinking) is asked again.
3. Feed back stdout/stderr tails and the new robot state; the model answers DONE or writes a
   program that continues from the current state. At most 4 runs; no new call after 240 s.
