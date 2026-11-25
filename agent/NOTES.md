# Notes

- **Model:** Inkling 2 Mini, the leaderboard's fixed model, reached only through `llm.chat`.
- **Coding agent:** Claude Code (Claude Opus 5.5) ported the harness; no tuning was done.

## What this agent is

The base solution: CaP-X's vanilla harness for the RoboSuite privileged configs, ported
unchanged into `solve()` in `agent.py`. One model call with CaP-X's system and user prompt
(temperature 1.0, max_tokens 20480), CaP-X's code extraction, one `robot.run` of the extracted
code, then return. No feedback to the model, no retries.

The docstring at the top of `agent.py` lists what was ported (with CaP-X file and line
references) and the few places where the leaderboard runner forces a difference.
