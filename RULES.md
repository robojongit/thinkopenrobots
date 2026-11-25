# Rules

The model is fixed and the code around it is yours to improve. This page explains what is scored,
how the final ranking is decided, and what is not allowed.

On this page: [Format](#format), [Scoring](#scoring), [Ranking](#ranking), [Rules](#rules-1),
[Episode budget](#episode-budget), [How to beat the base solution](#how-to-beat-the-base-solution).

## Format

Every entry uses the same model, Inkling 2 Mini. Your `agent/` folder decides how the model is
prompted. The model writes Python against a robot API, the code runs in simulation, and a task
checker decides success. There are 7 RoboSuite tasks from CaP-X, with poses and no images.

The base solution is vanilla Inkling 2 Mini with the vanilla CaP-X harness. It is the first entry
on the leaderboard and the starting point of every agent. The aim is to beat it.

- **Training:** 30 seeds for each of the 7 tasks (210 episodes). Fully public: seeds, scenes and
  the success checker. Run it as often as you like.
- **Public test:** 20 seeds for each of the 7 tasks (140 episodes). Every submission is scored on
  it, and only the overall score and its 95% interval are shown.
- **Private test:** 350 episodes, hidden until the end. Half are the known tasks on new seeds. The
  other half are held-out goal variants: new instructions on the same scenes and the same API. A
  harness that makes the model reason does well on them; one that memorises the 7 known answers
  does not.

## Scoring

- A score is the percentage of episodes where the task checker reports success. Each task has the
  same number of seeds, so hard tasks count as much as easy ones.
- Scores come with a 95% interval, because the model's sampling and the choice of seeds add
  noise. At 50% success the public interval is about 8 points either side, and the private one
  about 5.
- The leaderboard shows only the overall score, its interval and a coarse token bucket (under
  10k, 10–30k or over 30k tokens per episode). It never shows per-task test scores or any output
  from test episodes.
- The best of many public scores is inflated by luck. Public standings are a guide and are never
  used for the final result.

## Ranking

- **Explicit final picks.** In a 48-hour window before the finale, each account picks 2
  submissions with `bench pick`. With no picks, its 2 most recent submissions count, never its 2
  best public scores.
- **Two passes.** Every pick runs once on the private test. The top 10 run twice more, and their
  final score is the mean of 3 runs, because model sampling is not perfectly repeatable.
- **Ties are statistical.** Entries are compared on the shared private seeds. Entries whose 95%
  interval overlaps the leader's share first place, and the same rule forms the later rank bands.
- **Code review before results.** The top 10 agents are read before the reveal. Task-specific
  scripted solutions are disqualified.

The ranking uses the private test only. After the finale this site shows the private scores with
intervals, the rank bands and each account's chosen submissions.

## Rules

- **One GitHub account is one entrant.** Teams submit from one account. Using several accounts to
  get around limits is banned.
- **Any laptop works.** macOS and Linux can simulate locally. On Windows, `bench` runs natively
  and simulation uses the shared cloud machine (`--remote`); WSL2 is optional for local runs.
- **Up to 3 submissions per account per day,** with at most one waiting in the queue.
- **The robot only runs model-written code.** Agents may read poses directly, but anything that
  moves the robot must come from code Inkling wrote during the episode. `robot.run` accepts code
  that is an exact substring of the content of a model reply from this episode and runs it with
  the task's robot functions in scope; nothing participant-written runs with the robot API.
- **No task-specific scripted solutions,** in code or prompts. General helper code is welcome in
  your prompts, for the model to write into its own code.
- **No new dependencies.** The agent runs in a fixed image with Python, NumPy and SciPy.
- **Edit only `agent/`.** The harness, tasks and checkers are read-only.
- **Name your coding agent.** Each submission names the coding agent that built it, in
  `agent/NOTES.md`; it is shown on the leaderboard.
- **Code is open.** Submissions are public commits, and anyone may build on anyone's submitted
  agent, including submitting it as it is. The winning `agent/` folders become the next round's
  starting points.
- **Exploits are disqualified.** Reaching the checker, leaking test data or scripting solutions
  will disqualify an entry. Reporting an exploit is credited in the results.
- **Harness bugs** found during the competition are fixed and announced, and affected submissions
  are rescored for free.
- **The shared machine is best effort.** If it is down, submissions wait in the queue.

## Episode budget

Each episode has a fixed budget for model calls, total tokens (prompt and completion) and
wall-clock time, and no environment resets. The values are final: 10 calls, 66k tokens and
400 seconds. `bench train --help` shows them as the `--budget-*` defaults.

## How to beat the base solution

The same steps are in [`AGENTS.md`](AGENTS.md) in the repository, which a coding agent follows
when you paste the starter prompt.

### 1. Start from a measured baseline

Run `bench train --seeds 5` (35 episodes), then the full training split, before changing
anything. The unchanged base solution scores about 43% overall: cube lifting and cube stack are
near the top, cube restack and two-arm handover are at 0%, and nut assembly, spill wipe and
two-arm lift sit in between. If your numbers are far lower, check the `ended` counts in the
output for timeouts or model-server errors first. Five seeds per task is a smoke test, not a
measurement.

### 2. Read the failures

Use `bench view runs/<id> --failures --task <task>` and read several failed episodes of one task
before editing. For each, ask which of four things happened: the model returned no code (for
example it spent every token reasoning), the code crashed, the code ran cleanly but the task
still failed, or a budget limit ended the episode. Fix the largest group first. The base solution
makes one model call and never looks at the result, so most failures are the kind a second look
would catch.

### 3. Changes that tend to help

Make sure code arrives: keep the model's thinking short and ask again when a reply has no code.
Use more than one call: run the code, read its output and the robot's state, and ask the model to
repair its own code (the scene is not reset between runs). Check the result with the read
functions before returning. Spell out what the API docs leave unclear, such as units, frames and
what each function returns, in terms that hold for any task. Try sampling settings and a review
call before running. Keep every change general: the private test includes goal variants on the
same scenes, and a recipe for one task is a scripted solution, which is not allowed.

### 4. Check that a change is real

Rerun on the same seeds and the same budget, then run `bench compare runs/<before> runs/<after>`.
It pairs episodes by task and seed and reports the difference with a 95% interval. Keep a change
only when the verdict is a real gain, and revert it otherwise. For a borderline result, run both
versions again on more seeds with `--seed-list`. Write each attempt, kept or not, in
`agent/memory/`.

### 5. Pitfalls

Overfitting: gains that exist only on the training seeds, or that name a task's objects, will not
carry over to held-out goals. Public-score luck: the best of many public scores is inflated, so
do not submit every small tweak and do not pick finalists by public score alone. Budget: long
prompts, retries and extra calls spend tokens and seconds, and an episode still running at the
wall-clock limit fails even if the task was done. Noise: model sampling varies from run to run,
so a single run of a few episodes proves little. Comparisons across different budgets or models
are not valid.

Ready? [Copy the starter prompt](README.md#start-with-one-prompt), or browse the
[repository](https://github.com/robojongit/thinkopenrobots).
