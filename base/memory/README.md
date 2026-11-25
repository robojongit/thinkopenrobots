# Memory: the improvement loop

Notes for people and coding agents working on this `agent/` folder. Keep them short and true.

## The loop

1. **Measure first.** `uv run bench train` runs the training split with the current agent and
   writes `runs/<id>/` (add `--tasks cube_stack --seeds 5` for a quick check).
2. **Read failures.** `uv run bench view runs/<id> --failures` shows the model's replies,
   the code that ran, its output and its errors for each failed episode (`--full` for all of
   it). Look for a pattern before
   changing anything.
3. **Change one thing** in `agent/` (prompts, examples, control loop, sampling).
4. **Rerun on the same seeds** with `uv run bench train` (same `--tasks`/`--seeds`).
5. **Compare.** `uv run bench compare runs/<before> runs/<after>` gives a paired difference on
   shared seeds with a 95% interval. Keep the change only if the interval is above zero;
   otherwise revert it.
6. **Write it down** below, kept or not, so nobody tries the same thing twice.

Rules: edit only `agent/`; anything that moves the robot must be code the model wrote in this
episode (nothing you write runs with the robot API; general helper code goes in the prompts, for
the model to write into its own code); no task-specific scripted solutions; no new dependencies.

## Attempts

One entry per attempt: date, what changed, runs compared, result, kept or reverted.

<!-- 2026-01-01 · shorter system prompt · runs/A vs runs/B · +0.03 (-0.02 to +0.08) · reverted -->
