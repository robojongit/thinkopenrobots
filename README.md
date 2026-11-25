# Inkling Robotics Leaderboard

An open leaderboard for robot harnesses built on Inkling 2 Mini. The model is fixed; you improve
the code around it. Your agent prompts the model to write Python that drives simulated robot
arms through 7 CaP-X RoboSuite tasks, and the robot only runs code the model wrote. Public test
scores guide you; a hidden private test decides the final ranking.

The base solution is vanilla Inkling 2 Mini with the vanilla CaP-X harness. It is the first
entry on the leaderboard and the starting point of every agent. The aim is to beat it.

## Start with one prompt

Paste this into Claude Code, the Claude desktop app with a folder connected, or any coding agent
with a terminal:

```
I want to compete in the Inkling Robotics Leaderboard.
Clone https://github.com/robojongit/thinkopenrobots, read its AGENTS.md
and follow it: set me up, reproduce the base score, then keep
improving agent/ and ask me before each submission.
```

The repository URL is a working name and may change before launch. The agent follows
[`AGENTS.md`](AGENTS.md), runs `bench` on your machine, shows you one GitHub login code to
approve (no permissions are requested), and asks before each submission. You can allow it to
submit by itself when `bench compare` shows a clear gain, up to 3 a day.

## Quick start by hand

You need Git and [uv](https://docs.astral.sh/uv/). macOS and Linux simulate locally; see
[Windows](#windows).

```
git clone https://github.com/robojongit/thinkopenrobots
cd thinkopenrobots
git submodule update --init harness/capx
git -C harness/capx submodule update --init capx/third_party/robosuite
uv sync
uv run bench doctor                 # checks the setup and prints an exact fix for each problem
uv run bench login                  # GitHub device login; saves the gateway key
uv run bench train --seeds 5        # 35 episodes: a quick check that the base score reproduces
uv run bench train                  # the full training split, 210 episodes
uv run bench view runs/<id> --failures
uv run bench compare runs/<before> runs/<after>
uv run bench submit                 # up to 3 a day
uv run bench status                 # queue position and public score
```

Do not clone with `--recurse-submodules`: it fetches CaP-X submodules you do not need.

- `bench train` runs episodes and writes `runs/<id>/` (one folder per episode with the
  transcript, the code that ran and its errors) and a row in `results.tsv`. Rerunning with the
  same `--out` resumes. `--tasks`, `--seeds N`, `--seed-list "1-50"`, `--workers`, and
  `--budget-*` choose what runs; `bench train --help` lists them all.
- `bench train --video` also writes `video.mp4` into each episode folder, at about 4x the time;
  use it for the few episodes you want to watch (`--tasks <task> --seed-list <seed>`). With
  `--remote`, at most 10 episodes of a run are recorded. Submissions never return videos.
- `bench view` prints readable traces; `--failures` shows only failed episodes.
- `bench compare` compares two runs on the episodes they share and gives a 95% interval and a
  verdict: `real gain`, `no clear difference` or `real loss`.
- Every command accepts `--json`, never prompts, and uses stable exit codes (`bench --help`
  lists them; 0 is success, and a failed task is not an error).

## The 7 tasks

Each task has a goal sentence in plain English, a robot API (the functions the model's code can
call), and a checker. Read functions can be used from `agent/agent.py` as well; motion functions
only run inside code the model wrote. Success rates are the base solution's, measured on 50
seeds per task (about 45% overall).

| Task | Goal | Read functions | Motion functions | Base solution |
| --- | --- | --- | --- | --- |
| `cube_lifting` | Pick up the red cube and lift it. | `get_object_pose`, `sample_grasp_pose` | `goto_pose`, `open_gripper`, `close_gripper` | 98% |
| `cube_stack` | Stack the red cube gently on the green cube, then release it. | `get_object_pose`, `sample_grasp_pose` | `goto_pose`, `open_gripper`, `close_gripper` | 96% |
| `cube_restack` | Place the red cube on the green cube and open the gripper; nothing dropped from a height. | `get_object_pose`, `sample_grasp_pose` | `goto_pose`, `open_gripper`, `close_gripper` | 0% |
| `nut_assembly` | Grasp the square nut and insert it onto the square peg. | `get_object_pose`, `sample_grasp_pose` | `goto_pose`, `goto_home_joint_position`, `open_gripper`, `close_gripper` | 40% |
| `spill_wipe` | Wipe up the brown spill. | `get_object_pose` | `goto_pose` | 58% |
| `two_arm_lift` | Each arm grasps one pot handle; lift the pot level. | `get_handle0_pos`, `get_handle1_pos`, `get_arm0_gripper_pose`, `get_arm1_gripper_pose` | `goto_pose_arm0`, `goto_pose_arm1`, `goto_pose_both`, `open_gripper_arm0`, `open_gripper_arm1`, `close_gripper_arm0`, `close_gripper_arm1` | 22% |
| `two_arm_handover` | Arm 0 lifts the hammer and hands it to arm 1, which grasps the handle. | `get_hammer_pose` | `goto_pose_arm0`, `goto_pose_arm1`, `open_gripper_arm0`, `open_gripper_arm1`, `close_gripper_arm0`, `close_gripper_arm1` | 0% |

Goals are paraphrased; the exact instruction and the API docs arrive as `task.instruction` and
`task.api_docs`. Only poses are available, no images. On `two_arm_handover` the base solution
spends its whole token allowance reasoning and returns no code.

## What you write

Everything under `agent/` is yours; it starts as a copy of `base/`. The entry point is one
function:

```python
from inkling_robo import LLM, Robot, Task


def solve(task: Task, robot: Robot, llm: LLM) -> None:
    messages = [
        {"role": "system", "content": "You generate Python code that solves the task."},
        {"role": "user", "content": f"{task.instruction}\nAPIs:\n{task.api_docs}"},
    ]
    reply = llm.chat(messages, temperature=1.0, max_tokens=20480)
    print(f"finish_reason={reply.finish_reason} tokens={reply.completion_tokens}")
    code = reply.split("```python\n", 1)[-1].split("```", 1)[0].strip()
    result = robot.run(code)  # code must come from a model reply of this episode
    print(f"ok={result.ok}")
```

- `task.instruction` and `task.api_docs`: the goal and the robot API, as text.
- `robot.api.<read function>(...)` and `robot.observe()`: read poses and gripper and joint state.
- `llm.chat(messages, **options)`: the one route to the model. It returns a `Reply` (a string
  with `.reasoning`, `.finish_reason`, `.prompt_tokens`, `.completion_tokens`). Allowed options:
  `temperature`, `top_p`, `max_tokens`, `stop`, `presence_penalty`, `frequency_penalty`.
- `robot.run(code)`: runs code in a fresh process with the task's robot functions in scope and
  returns `RunResult(stdout, stderr, ok)`. It accepts only code that is an exact substring of a
  model reply's content from this episode, and nothing you write is loaded there. The scene is
  not reset between runs.
- The episode ends when `solve` returns or the budget runs out. Only then does the checker run,
  once. An episode still running at the wall-clock limit fails.
- Your code runs with Python, NumPy and SciPy only, with no network and no other models.

`base/agent.py` is the reference: its docstring lists exactly what was ported from CaP-X.
`agent/memory/` holds the improvement loop and a log of attempts.

## Rules

The full rules are on the [rules page](site/rules.html) of the site. In short:

- Edit only `agent/`. The harness, tasks and checkers are read-only. No new dependencies.
- Anything that moves the robot must come from code the model wrote during the episode:
  `robot.run` accepts only an exact substring of a model reply from this episode, and nothing
  you write runs with the robot API. General helper code is welcome in your prompts, for the
  model to write into its own code. No task-specific scripted solutions, in code or prompts.
- One GitHub account is one entrant. Up to 3 submissions per account per day, one waiting in the
  queue at a time.
- Each submission names the coding agent that built it on one line of `agent/NOTES.md`, for
  example `Coding agent: Claude Code · Fable 5` (or `Coding agent: by hand`). It is shown on the
  leaderboard next to the submission.
- Code is open: submissions are public, and you may build on anyone's submitted agent, including
  submitting it as it is.
- Exploits are disqualified; reporting one is credited.

## Scoring and ranking

| Split | Episodes | What it is |
| --- | --- | --- |
| Training | 30 seeds x 7 tasks | Fully public. Run it as often as you like. |
| Public test | 20 seeds x 7 tasks | Every submission is scored on it; only the overall score and its 95% interval are shown. |
| Private test | 25 seeds x 7 tasks, plus 25 x 7 held-out variants | Hidden until the end. Decides the ranking. |

- A score is the mean of the 7 task success rates, so hard tasks count as much as easy ones. It
  comes with a 95% interval: about 8 points either side at 50% on the public test, about 5 on the
  private test.
- Half of the private test consists of held-out goal variants: new instructions on the same
  scenes and API. A harness that makes the model reason does well; one built around the 7 known
  answers does not.
- Each account picks 2 final submissions in a 48-hour window with `bench pick`; with no picks,
  its 2 most recent submissions count. Picks run on the private test; the top 10 run three times
  and are averaged.
- Entries whose 95% interval overlaps the leader's share first place, and the same rule forms
  later rank bands. The public score is never used for ranking.
- The top 10 agents are code-reviewed before the reveal.

## Budget

Each episode has a fixed budget, and no environment resets: 10 model calls, 66k tokens (prompt
plus completion) and 400 seconds of wall-clock time. These values are final; they are the
`--budget-*` defaults in `uv run bench train --help`. The base solution uses one call per
episode, 10 to 280 seconds and 1.5k to 22k tokens.

## The shared model, Mac and Ollama fallback

Episodes call a shared Inkling 2 Mini server. If it is rate limited (429) or asleep (503),
`bench train` waits and tells you; rerun the same command if it gives up (exit code 4). If you
would rather use your own machine, point `bench` at any OpenAI-compatible server, for example
Ollama on a Mac:

```
uv run bench train --model-url http://localhost:11434/v1 --model <model name>
```

Scores from another model are only a rough guide for development. Submissions are always scored
on the shared model.

## Windows

`bench` runs natively on Windows, but RoboSuite supports only macOS and Linux, so on native
Windows `bench train` simulates on the shared machine by default (`--remote`) and downloads the
same run folders. WSL2 is optional if you want local simulation.

## Getting help

Use GitHub issues for questions and discussion only. Submissions do not go through issues; they
go through `bench submit`. Harness bugs are fixed and announced, and affected submissions are
rescored for free.

## License

MIT, see [LICENSE](LICENSE).

## Development

The local simulator is CaP-X, pinned as the `harness/capx` submodule, with its RoboSuite
submodule (no other CaP-X submodule is needed). Python 3.10 to 3.12; CPU only.

```
git submodule update --init harness/capx
git -C harness/capx submodule update --init capx/third_party/robosuite
uv sync                     # dev tools + the default `sim` group (CaP-X, RoboSuite, MuJoCo)
uv run ruff check . && uv run ruff format --check . && uv run pytest   # fast suite, no simulator
uv run pytest -m sim        # real simulator: scene determinism, reference solutions, video
```

- Container mode (how scoring runs agents): `docker/agent/build.sh` builds the agent image, then
  `uv run pytest -m container` runs the isolation tests under every runtime Docker has of runc
  and runsc (gVisor; register it with `"runtimeArgs": ["--host-uds=open"]`), and
  `bench train --sandbox container [--runtime runc]` trains in containers.
- `uv sync --no-group sim` installs without the simulator; the project's own dependencies (all
  the participant SDK in `sdk/inkling_robo` uses) are msgpack and NumPy.
- `INKLING_CAPX_ROOT=/path/to/cap-x` uses another CaP-X checkout's code and configs (RoboSuite
  and the other packages still come from the synced environment).
- The CaP-X APIs call a PyRoKi IK server on 127.0.0.1:8116. The harness reuses one that is
  running or starts its own and stops it on exit; to share one between several processes, keep
  `uv run python -m harness.ik_server` running.
- Reference solutions through the harness, next to CaP-X's own launch path on the same seeds:
  `uv run python -m harness.tests.reference --seeds 1-10 --native`.
- Videos (`video=True`) render with MuJoCo's EGL backend; everything else runs with rendering
  off.
