# Notes

Coding agent: Claude Code · Opus 5.5

The line above is read by `bench submit` and shown on the leaderboard next to each submission.
Keep exactly one such line, starting the line, and change its value to the coding agent that
builds this agent: a short name such as `Claude Code · Fable 5`, `Codex`, `Cursor` or `by hand`
(1-40 characters: letters, digits, spaces and `. _ + - / ( ) ·`).

- **Model:** Inkling 2 Mini, the leaderboard's fixed model, reached only through `llm.chat`.
- **How it was built:** Claude Code (Opus 5.5), starting from the CaP-X port in `base/`.

## What this agent is

A multi-turn loop in `agent.py`. The model replies with one fenced Python block per turn; the
block runs (fresh process, scene persists) and its stdout, stderr and the robot's gripper/joint
state go back to the model, which continues, repairs, verifies with read functions, and
replies DONE. Replies with no runnable code get a short corrective message. Calls, tokens and
wall clock are tracked locally so the episode ends before the 400 s limit. No task-specific
content: everything comes from `task.instruction` and `task.api_docs`.
