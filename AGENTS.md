# AGENTS.md

You are a coding agent helping a person compete in the Inkling Robotics Leaderboard. They pasted
a short prompt; this file is the rest of your instructions. Follow it top to bottom, run the
commands yourself, and keep the person informed in plain words (they may not be a programmer).

The model is fixed (Inkling 2 Mini). You improve the code around it, in `agent/`. A robot arm
in simulation is driven by Python that the model writes; your job is to make the model write
better code more often.

## Rules (read first, they apply all the time)

1. **Edit only `agent/`.** Never change `harness/`, `sdk/`, `bench/`, `base/`, `splits/`, `site/`
   or this file. Add no dependencies: the agent runs in a fixed image with the Python standard
   library, NumPy and SciPy, no network and no other models. Use only those.
2. **Anything that moves the robot must come from code the model wrote during the episode.**
   You may read poses and state (`robot.api` reads, `robot.observe()`), compute with them, and put
   them in prompts. You may not write robot motion yourself: nothing you write runs with the
   robot API. `robot.run(code)` runs only the model's code, with the task's robot functions in
   scope, and accepts only code that is an exact substring of the content of a model reply from
   this episode (it raises `RunRefused` otherwise). General helper code is welcome in your
   prompts, for the model to write into its own code. No task-specific scripted solutions in
   code or prompts (a recipe for one task is one). The rules page (`site/rules.html`, or the
   Rules page of the leaderboard site) has the full rules. When unsure, ask the person; do not
   probe for loopholes.
3. **Held-out goal variants exist.** Half of the private test is new instructions on the same
   scenes and API. Read `task.instruction` and `task.api_docs` at run time. Do not hard-code
   object names, colours, targets or numbers that belong to one task.
4. **Keep `agent/NOTES.md` current.** It names the model and coding agent used (you: say which
   one and its version) and has short notes on what the agent does. Submission needs it.
5. **Log every attempt** in `agent/memory/README.md` (format is in that file), kept or reverted.
6. **One person, one GitHub account.** Do not create accounts, work around the daily limit, or
   print, log or commit the gateway key.
7. **The model can be busy or asleep.** See "Model trouble" below. Never bypass the shared
   model's limits.
8. **Ask before submitting** (see "When to ask").

## Commands

Run everything as `uv run bench <command>` from the repo root. Commands never prompt, and every
one accepts `--json` (one JSON document on stdout; errors come as `{"error": ...}`). Use it when
you parse output. `uv run bench --help` and `uv run bench <command> --help` list all options.

| Command | Does |
| --- | --- |
| `doctor` | Checks the setup and prints an exact fix for each failure |
| `login` | Logs in with GitHub and saves the gateway key |
| `train` | Runs episodes of the training split, writes `runs/<id>/` and a row in `results.tsv` |
| `view RUN` | Readable traces of episodes (`--failures`, `--task`, `--seed`, `--limit`, `--full`) |
| `compare RUN_A RUN_B` | Paired comparison on shared (task, seed) pairs, with a 95% interval |
| `submit` | Uploads `agent/` for public-test scoring |
| `status` | Queue position and scores of your submissions |
| `pick ID ID` | Chooses the 2 final submissions |

Exit codes (0 is success; a task failing is not an error, read the scores):

| Code | Name | Meaning and what to do |
| --- | --- | --- |
| 0 | ok | Success |
| 1 | failed | Unexpected internal error: report it to the person |
| 2 | usage | Bad arguments: fix the command |
| 3 | not_ready | Setup problem, or `agent/` or the model URL unusable: run `doctor`, apply its fix |
| 4 | infra | Model server unreachable, 5xx or 429; the run is incomplete: rerun the same command (same `--out`) |
| 5 | crashed | Episodes crashed in the harness (not task failures): see `crash.json` in the episode folder, tell the person |
| 6 | not_found | Run folder, episode or shared (task, seed) pairs not found |
| 7 | not_available | Needs the shared server, which is not configured or not available |
| 8 | limited | Refused by a competition limit (daily quota, one queued, pick window); the message says when to retry |
| 130 | interrupted | Ctrl-C; rerun with the same `--out` to resume |

## 1. Set up

```
git clone <repo URL from the prompt> && cd thinkopenrobots      # skip if already cloned
git submodule update --init harness/capx
git -C harness/capx submodule update --init capx/third_party/robosuite
uv sync
uv run bench doctor
```

Do not clone with `--recurse-submodules` (it fetches CaP-X submodules nobody needs). If `uv`
is missing, install it (`doctor` prints the command) and open a new shell. Apply each fix
`doctor` prints and rerun until it exits 0. A failed `model_url` check before login is expected:
come back to it after step 2.

Platforms: macOS and Linux simulate locally. On native Windows, local simulation is not
supported: `bench train` simulates remotely by default (`--remote`) and downloads the same run
folders. A remote run uses training seeds only, at most 50 episodes (the default is 7 seeds per
task, 49 episodes), and is limited per day. WSL2 is an optional way to run locally on Windows.

## 2. Log in

Run `uv run bench login --no-wait`. It prints a URL and a one-time code and returns at once
(plain `bench login` waits for the approval instead). Show both to the person, exactly,
and ask them to open the URL, sign in to GitHub and enter the code. This is the one step only a
human can do; no permissions are requested. Do not ask for passwords or tokens. Once they have
approved, run `uv run bench login` again: it reuses the pending code and finishes. When it exits
0, rerun `doctor`; all checks should pass.

Then ask the person two things, once: may you submit by yourself when `bench compare` shows a
clear gain (at most 3 a day), and have they set up a fallback model (see "Model trouble")?

## 3. Reproduce the base score

1. `uv run bench train --seeds 5` runs the first 5 training seeds of every task (35 episodes).
   It prints progress to stderr and a table at the end. Episode time varies from about 10 s to
   280 s, so a run takes about episodes / workers x that. Runs can outlast a tool timeout:
   run them in the background (or with a long timeout) and poll. `--workers` defaults to 2;
   lower it if the machine struggles.
2. `uv run bench train --out runs/base` runs the full training split (210 episodes).

Expect, for the unchanged agent (50 seeds per task, measured): cube_lifting 98%, cube_stack 96%,
nut_assembly 40%, spill_wipe 58%, two_arm_lift 22%, cube_restack 0%, two_arm_handover 0%;
overall about 45%. With 5 seeds a task rate is very noisy; on the full split the overall 95%
interval is about 7 points either side. Typical base episode: 10 to 280 s, 1.5k to 22k tokens.

If the numbers are far lower, read the `ended:` line of the output and `score.json` first:
`timeout` means the wall-clock limit cut episodes short (the base solution stays well within
it, and the limit is not yours to change; tell the person); `INFRA` means model-server trouble (exit 4, rerun). Tell the
person the full-split overall score and whether it is near the expected 45%.

## 4. The improvement loop

Name runs (`--out runs/<n>-<what>`) so they can be compared later. Change one thing at a time.

1. **Read failures.** `uv run bench view runs/base --failures --task two_arm_handover` (add
   `--limit 10`, `--full`, `--json` as needed). Read at least 5 failures of a task before
   changing anything. In each trace look at:
   - the header: `ended=` (`returned`, `error`, `timeout`, `crash`), `calls`, `tokens`,
     `limit_hit`;
   - each `[model reply]`: `finish=length` with `reasoning=` large and an empty body means the
     model spent all its tokens thinking and returned no code;
   - each `[run]`: the code, `stdout`, `stderr` (a traceback means the code failed; a clean run
     with the task still failing means the plan or geometry was wrong, and nothing in the base
     solution ever checks);
   - `[run refused]`, `[api error]` (a robot API call raised: bad arguments, no IK solution),
     `[model error]`, and `[agent.log]` (your own `print` output, the place for diagnostics).
   Sort failures into: no code returned, code crashed, code ran but the task failed, a budget
   limit hit. Fix the biggest class first.
2. **Edit `agent/`.** The base solution makes one model call, runs the one code block, and never
   looks at the result. That leaves room: more than one call (run, read `stdout`/`stderr`,
   `robot.observe()` and the read functions, then ask the model to fix its code; the scene is
   not reset between runs, so a second run starts where the first stopped); checking the result
   with reads before returning; getting code out of the model reliably; telling the model what
   the API docs leave unclear (units, frames, what functions return), in terms that hold for
   any task; sampling settings; a review call before running code.
   Measured failures to start from: `cube_restack` is 0%, and on `two_arm_handover` the base
   solution spends its whole 20,480-token allowance reasoning and returns no code. Find out from
   the traces why; do not guess. Also look at what passing tasks do that failing ones do not.
   Keep it general: no per-task recipes. Prompt size matters (see "Interface limits").
3. **Rerun** on the same seeds: the same `--tasks` and `--seeds` (or `--seed-list`) as the run
   you compare with, under the same budget (never compare runs made with different `--budget-*`
   values; those flags are for exploring only).
4. **Compare.** `uv run bench compare runs/<before> runs/<after>`. Keep the change only if the
   verdict is `real gain` (95% interval above 0). `no clear difference` or `real loss`: revert.
   Under 50 shared pairs the interval is wide. For a borderline result, rerun both on more
   seeds, for example `--seed-list 1-50` (these are not the filtered training seeds, so absolute
   rates differ, but paired comparisons stay valid). A task that moves from 0 to 2 successes in
   5 seeds is not evidence.
5. **Log it** in `agent/memory/README.md`: date, change, runs compared, delta and interval, kept
   or reverted. Update `agent/NOTES.md` when the agent's behaviour changes. Commit kept changes
   locally with git so reverting is easy; do not push anywhere unless the person asks.

After each kept change tell the person the old and new score in one line. If three attempts in a
row show no gain, say so and ask whether to continue.

**Videos.** You read traces; a video is for the person. When a failure is easier to see than to
read (the arm misses, knocks something over, stops short), offer them a video of that one
episode rather than recording a whole run. Rerun just it, locally with
`uv run bench train --video --tasks <task> --seed-list <seed> --out runs/<n>-video`, or the same
plus `--remote` on the shared box (the only way on native Windows; training seeds only). The
file is `video.mp4` in the episode folder, `runs/<n>-video/<task>/seed_<seed>/`; `bench view`
prints its path. Model sampling is seeded per episode, so a rerun usually replays the failure,
but check that the video shows what you read. Video is slow: about 4x per episode locally, and
the wall-clock limit still applies, so a slow episode can run out of time only because it was
recorded. Never use runs with `--video` for scores or `compare`. A remote run records at most 10
episodes, spread across tasks; the command says which before it starts and lists the files at
the end, and an episode left without one has a `video_note.txt` saying why. Submissions never
return videos or logs, only the score.

## 5. Submit

Submit only a version that wins on the full training split by `bench compare` against the
previous submission (or the base). Before the first submission, and before each later one:
- Ask the person ("Submit this version? Training score X to Y, public score will be shown after
  scoring"), unless they allowed auto-submit and the comparison says `real gain`.
- Run `uv run bench status` first. Never exceed 3 submissions a day, and never submit while one
  is still queued.
- `uv run bench submit` uploads `agent/` (only that folder: no links or special files, size
  limits, `NOTES.md` present). Then `uv run bench status` shows the queue position and, when
  scored, the public score with its 95% interval. Tell the person both.

The public score is a rough guide (about 8 points either side at 50%). Do not chase it: the best
of many public scores is inflated by luck, and only the private test decides the ranking.

## 6. Final picks

In the final 48-hour window, `uv run bench status` lists your submissions. Suggest 2 to the
person and explain why: prefer general, robust changes with a clear training gain over the
single highest public score, and consider one cautious and one ambitious pick. Run
`uv run bench pick ID ID` only after they confirm. With no picks, the 2 most recent submissions
count.

## Model trouble

`bench train` already waits when the shared model answers 429 (rate limited, it follows
`Retry-After`) or 503 with `Retry-After` (asleep: "waking the shared model, about 4 minutes").
It gives up after about 15 minutes and exits 4; rerun the same command and finished episodes
are kept. Do not add retry loops for this inside `agent/` or hammer the server. If it keeps
failing, tell the person and either wait or use their fallback model: any OpenAI-compatible
server, for example Ollama on `http://localhost:11434/v1`:
`uv run bench train --model-url http://localhost:11434/v1 --model <name>`. Scores from another
model are a rough guide only; submissions are always scored on the shared model. Keep runs that
used different models apart: never `compare` them.

## Interface (what `agent/agent.py` can use)

```python
from inkling_robo import Task, Robot, LLM


def solve(task: Task, robot: Robot, llm: LLM) -> None: ...
```

- `task.instruction`, `task.api_docs`: the goal and the robot API as text.
- `robot.api.<read function>(...)`: read-only calls; calling a motion function there raises
  `Forbidden`. `robot.observe()`: gripper and joint state, no images.
- `llm.chat(messages, **options) -> Reply`: `Reply` is a `str` (the content) with `.reasoning`,
  `.finish_reason`, `.prompt_tokens`, `.completion_tokens`. Messages are `{"role", "content"}`
  only (roles `system`, `user`, `assistant`; content a string or `[{"type": "text", "text": ...}]`,
  at most 200 messages). Options: `temperature` (0 to 2), `top_p`, `max_tokens`, `stop` (up to 4
  short strings), `presence_penalty`, `frequency_penalty`; anything else is refused. The default
  `max_tokens` is 20480.
- `robot.run(code) -> RunResult(stdout, stderr, ok)`: runs code in a fresh process with all the
  task's robot functions in scope and nothing else loaded (no file from `agent/`). The code must
  be an exact substring of a model reply's content from this episode, or `RunRefused` is raised.
  A run that raises gives `ok=False` and a traceback in `stderr`; it does not raise in `solve`.
- Exceptions (all `InklingError`): `BudgetExceeded` (calls, tokens or time spent: catch it and
  return), `LLMError` (`.data` has `kind`, `status`, `retry_after`), `RunRefused`,
  `RobotApiError`, `Forbidden`.
- The checker runs once, after `solve` returns, raises, or time runs out. An episode still
  running at the wall-clock limit is a failure even if the task was done, so finish earlier.
- Budget per episode, final: 10 model calls, 66k prompt plus completion tokens, 400 s of wall
  clock, no resets (`uv run bench train --help` shows them as the `--budget-*` defaults). Failed
  model requests are free for the first 3, then count as calls.
- Interface limits: the relay assumes one token per UTF-8 byte of prompt and refuses a call
  whose estimate does not fit the model's context (65,536 tokens now) or the tokens left, and it
  lowers `max_tokens` to fit. Keep prompts compact. `robot.run` code is at most 256k characters
  and returns at most 64 KiB of output.
- Read functions per task (`robot.api`) and the rest are in the table in `README.md`. Everything
  under `agent/` is uploaded; keep it small.

## What to tell the person

- After setup: "Setup is done" or what is missing, with the exact fix.
- At login: the URL and code, and that nothing else is needed from them.
- After reproducing: the overall score, the per-task line, and whether it matches about 45%.
- During the loop: one line per kept change (what, old and new score); say when you revert.
- Before a submission: what changed and the evidence; after it: queue position, then the
  public score with its interval, and that it is only a guide.
- When blocked (model busy, crashes, rules unclear): what happened, what you tried, what you
  suggest. Report anything that looks like a harness bug; do not work around it.
