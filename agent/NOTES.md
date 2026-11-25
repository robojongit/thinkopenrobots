# Notes

Coding agent: reference harness (Claude Code port)

The line above is read by `bench submit` and shown on the leaderboard next to each submission.
Keep exactly one such line, starting the line, and change its value to the coding agent that
builds this agent: a short name such as `Claude Code · Fable 5`, `Codex`, `Cursor` or `by hand`
(1-40 characters: letters, digits, spaces and `. _ + - / ( ) ·`).

- **Model:** Inkling 2 Mini, the leaderboard's fixed model, reached only through `llm.chat`.
- **How it was built:** Claude Code (Claude Opus 5.5) ported the CaP-X harness; no tuning was
  done.

## What this agent is

The base solution: CaP-X's vanilla harness for the RoboSuite privileged configs, ported
unchanged into `solve()` in `agent.py`. One model call with CaP-X's system and user prompt
(temperature 1.0, max_tokens 20480), CaP-X's code extraction, one `robot.run` of the extracted
code, then return. No feedback to the model, no retries.

The docstring at the top of `agent.py` lists what was ported (with CaP-X file and line
references) and the few places where the leaderboard runner forces a difference.
