"""Closed-loop agent: write code, run it, show the model what happened, let it fix or finish.

General, task-agnostic. Everything that moves the robot is code the model wrote in this episode;
this file only builds prompts, extracts code, reads state (``robot.observe()``) and decides
when to stop.

Loop:
1. One call with the task, the API docs, the current robot state and general guidance
   (keep thinking short, one fenced Python block, print what you measure, verify with reads).
2. Run the code. If the reply had no code (for example the model spent all its tokens
   thinking), ask again, insisting on a short plan and code.
3. Feed back stdout, stderr and the new robot state. The model either answers DONE or writes a
   new program that continues from the current state (the scene is not reset).
4. Stop on DONE, on budget, or when too little wall-clock time is left for another run.
"""

from __future__ import annotations

import json
import time

from inkling_robo import LLM, BudgetExceeded, InklingError, Robot, Task

TEMPERATURE = 0.7
FIRST_MAX_TOKENS = 14000
FOLLOW_MAX_TOKENS = 9000
WALL_LIMIT_S = 400.0
# Do not start another model call + run after this many seconds; an episode still running at
# the wall-clock limit fails even if the task was done.
LAST_CALL_AT_S = 240.0
MAX_ROUNDS = 4
OUT_TAIL = 3500
ERR_TAIL = 2000

SYSTEM_PROMPT = """You control a simulated robot by writing Python programs against the API below.
The API functions are already defined in the program's global scope (do not import them); import numpy/scipy yourself if needed. Only the standard library, numpy and scipy are available.

How to answer:
- Think briefly: a short plan of the steps and the key numbers, not a long deliberation. Do not try to settle every detail in your head; print values and check them in the code instead.
- Then reply with exactly ONE fenced code block: ```python ... ``` containing a complete program. No other code blocks.
- The program must print what it measures: positions read before and after each important step (e.g. object positions after grasping/lifting/placing), so the result can be verified. Print numbers rounded to 3 decimals.
- Use the read functions to check progress inside the program (e.g. after a grasp and lift, check that the object actually rose; after placing, check where it ended).

General robot facts:
- Positions are in meters, quaternions are (w, x, y, z). Read the docstrings for which frame and which point of an object a function returns.
- Approach grasp/place targets from above (use z_approach where available) and keep moves between waypoints clear of other objects: lift up before travelling sideways.
- To put an object on another one, compute the target height from positions and extents (sizes) of both objects; lower it close to resting contact before opening the gripper, never drop it from a height.
- Before moving an object, check whether something else is on top of it or in the way, using the positions you read.
- An arm/gripper motion function that raises an error (e.g. no IK solution) means the target pose is unreachable: pick a nearby pose or a different orientation.
- Keep programs efficient: a few waypoints, no long loops or sleeps. Time is limited.
"""

FEEDBACK_PROMPT = """The program ran. The scene was NOT reset: the robot and objects are where the program left them.
ok={ok}
stdout (tail):
{stdout}
stderr (tail):
{stderr}
Robot state now (observe()):
{state}

Check against the goal, using the printed values. If the goal is fully achieved and the final state satisfies it, reply with just the word DONE (no code).
Otherwise think briefly about what went wrong, then reply with ONE fenced ```python block with a new complete program that continues from the current state and finishes the task (API functions are in scope again; re-import numpy; re-read poses, do not assume earlier variables exist)."""

NO_CODE_PROMPT = """Your reply contained no code (you may have run out of tokens while thinking). Do not deliberate long: write a short plan of 3-6 lines, then ONE fenced ```python block with the complete program."""


def _fmt(value, nd: int = 3):
    if isinstance(value, float):
        return round(value, nd)
    if isinstance(value, (list, tuple)):
        return [_fmt(v, nd) for v in value]
    if isinstance(value, dict):
        return {k: _fmt(v, nd) for k, v in value.items()}
    try:  # numpy arrays or scalars, if any come through
        return _fmt(value.tolist(), nd)
    except AttributeError:
        return value


def robot_state(robot: Robot) -> str:
    try:
        return json.dumps(_fmt(robot.observe()))
    except Exception as e:  # noqa: BLE001 - state is only informative
        return f"(unavailable: {type(e).__name__})"


def extract_code(content: str) -> str:
    """The last fenced python block (or any fenced block); else bare code; else ''."""
    text = str(content)
    for fence in ("```python\n", "```py\n", "```Python\n", "```\n"):
        start = text.rfind(fence)
        if start != -1:
            body = text[start + len(fence) :]
            end = body.find("```")
            if end != -1:
                body = body[:end]
            return body.strip()
    stripped = text.strip()
    if stripped.upper().startswith("DONE"):
        return ""
    # Bare code: accept only if it looks like code.
    if any(k in stripped for k in ("(", "import ", "=")) and "\n" in stripped:
        return stripped
    return ""


def is_done(content: str) -> bool:
    text = str(content).strip()
    return "```" not in text and text.upper().rstrip(".!").endswith("DONE")


def tail(text: str, n: int) -> str:
    text = (text or "").strip()
    return text if len(text) <= n else "...\n" + text[-n:]


def solve(task: Task, robot: Robot, llm: LLM) -> None:
    t0 = time.monotonic()

    def elapsed() -> float:
        return time.monotonic() - t0

    user = (
        f"{task.instruction}\nAPIs:\n{task.api_docs}\n\n"
        f"Robot state now (observe()):\n{robot_state(robot)}"
    )
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
    max_tokens = FIRST_MAX_TOKENS
    runs = 0
    for round_no in range(MAX_ROUNDS + 2):
        if round_no > 0 and elapsed() > LAST_CALL_AT_S:
            print(f"stop: {elapsed():.0f}s elapsed")
            return
        try:
            reply = llm.chat(messages, temperature=TEMPERATURE, max_tokens=max_tokens)
        except BudgetExceeded:
            print("stop: budget exceeded")
            return
        except InklingError as e:
            print(f"stop: model error {e.code}")
            return
        print(
            f"call {round_no}: t={elapsed():.0f}s finish={reply.finish_reason} "
            f"prompt={reply.prompt_tokens} completion={reply.completion_tokens} "
            f"reasoning_chars={len(reply.reasoning or '')} content_chars={len(reply)}"
        )
        if runs > 0 and is_done(reply):
            print("model says DONE")
            return
        code = extract_code(reply)
        messages.append({"role": "assistant", "content": str(reply) or "(no answer)"})
        if not code:
            messages.append({"role": "user", "content": NO_CODE_PROMPT})
            max_tokens = FOLLOW_MAX_TOKENS
            continue
        if runs >= MAX_ROUNDS:
            return
        try:
            result = robot.run(code)
        except InklingError as e:
            print(f"run error: {e.code}")
            return
        runs += 1
        print(f"run {runs}: ok={result.ok} t={elapsed():.0f}s")
        if elapsed() > LAST_CALL_AT_S:
            print(f"stop: {elapsed():.0f}s elapsed")
            return
        messages.append(
            {
                "role": "user",
                "content": FEEDBACK_PROMPT.format(
                    ok=result.ok,
                    stdout=tail(result.stdout, OUT_TAIL) or "(empty)",
                    stderr=tail(result.stderr, ERR_TAIL) or "(empty)",
                    state=robot_state(robot),
                ),
            }
        )
        max_tokens = FOLLOW_MAX_TOKENS
