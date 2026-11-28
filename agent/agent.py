"""Multi-turn agent: the model writes code, sees what it printed, and continues.

Loop per episode (all task knowledge comes from ``task.instruction`` and ``task.api_docs``):

1. Ask for one fenced Python block. The model is told it works in turns, that each block runs
   in a fresh process (robot state persists, Python variables do not), to keep its thinking
   short, and to verify the goal with read functions before saying DONE.
2. Run the block, send back stdout, stderr and the robot's gripper/joint state, and ask for the
   next block (or DONE once the printed reads show the goal holds).
3. Replies without runnable code (tool-call syntax, empty content after a long think) get a
   short corrective message instead of a run.

Budgets are tracked locally (calls, tokens, wall clock) so the episode returns before the
400 s limit; ``max_tokens`` shrinks as time runs out.
"""

from __future__ import annotations

import re
import time

from inkling_robo import LLM, BudgetExceeded, InklingError, Robot, RunRefused, Task

MAX_CALLS = 10
TOKEN_BUDGET = 66_000
TIME_BUDGET = 400.0
TIME_MARGIN = 25.0  # seconds kept free at the end
TOKENS_PER_SECOND = 85.0  # measured completion speed is ~100 tok/s; stay conservative
MAX_TOKENS = 12_000
MIN_TOKENS = 1_500
TEMPERATURE = 1.0
OUTPUT_CAP = 3_000  # characters of stdout/stderr echoed back per stream

SYSTEM_PROMPT = """You control a simulated robot by writing Python, working in turns.

How each turn works:
- Reply with a short plan (a few sentences at most) and then exactly ONE code block fenced as ```python ... ```. It is executed right away and you get back its stdout, stderr and the robot's gripper/joint state.
- Every block runs in a FRESH Python process: variables, functions and imports from earlier blocks are gone, but the robot and the objects stay where they are (the scene is never reset). Re-read any pose you need, and import numpy yourself.
- The robot API functions are already defined in that process; call them directly as Python. Do not use any tool-call or JSON syntax.
- Always use the ```python fence, even if the task text says not to use fences.

How to work:
- Keep your thinking short and practical. Measure instead of speculating: if a quantity is unclear (a position, size, height, orientation, or what a function returns), print it. Pick a simple plan you can check, then fix it from the output.
- A good first block reads and prints every relevant pose/state. Then write the full solution in one block, printing key values (target poses, poses after each important step). Exits without an error do not mean success.
- After acting, verify the goal with the read functions and print the evidence. If it is not met, work out from the numbers why and try again from the current state.
- When the printed readings show the goal is fully achieved and the robot is in a stable final state, reply with only the word DONE and no code.
- You have a limited number of turns (about {calls}) and limited time, so do not waste turns."""

FENCE_RE = re.compile(r"```[ \t]*(?:python|py|Python)?[ \t]*\n(.*?)```", re.S)


def extract_code(content: str) -> str | None:
    """Return a runnable block that is an exact substring of ``content``, or None."""
    blocks = [b for b in FENCE_RE.findall(content) if b.strip()]
    if blocks:
        return max(blocks, key=len).strip()  # one is asked for; the longest is the program
    if "```" in content:  # an unclosed fence: take everything after the opening line
        tail = content.split("```", 1)[1]
        tail = tail.split("\n", 1)[1] if "\n" in tail else ""
        if tail.strip() and _compiles(tail.strip()):
            return tail.strip()
        return None
    bare = content.strip()
    if bare and "<|" not in bare and _compiles(bare) and _calls_something(bare):
        return bare
    return None


def _compiles(code: str) -> bool:
    try:
        compile(code, "<model>", "exec")
    except (SyntaxError, ValueError):
        return False
    return True


def _calls_something(code: str) -> bool:
    return "(" in code


def is_done(content: str) -> bool:
    text = content.strip().strip("*`. ").upper()
    return text == "DONE" or (len(text) < 200 and re.search(r"\bDONE\b", text) is not None)


def clip(text: str, cap: int = OUTPUT_CAP) -> str:
    text = text or ""
    if len(text) <= cap:
        return text
    half = cap // 2
    return f"{text[:half]}\n[... {len(text) - cap} characters cut ...]\n{text[-half:]}"


def describe_state(robot: Robot) -> str:
    try:
        obs = robot.observe()
    except InklingError as exc:
        return f"(unavailable: {exc})"
    parts = []
    for key, value in obs.items():
        try:
            import numpy as np

            arr = np.round(np.asarray(value, dtype=float).ravel(), 3)
            parts.append(f"{key}={arr.tolist()}")
        except (TypeError, ValueError):
            parts.append(f"{key}={value}")
    return clip("; ".join(parts), 800)


def solve(task: Task, robot: Robot, llm: LLM) -> None:
    start = time.monotonic()
    tokens_used = 0
    calls = 0
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT.format(calls=MAX_CALLS - 1)},
        {"role": "user", "content": f"{task.instruction}\nAPIs:\n{task.api_docs}"},
    ]

    while calls < MAX_CALLS:
        time_left = TIME_BUDGET - TIME_MARGIN - (time.monotonic() - start)
        prompt_est = sum(len(str(m["content"]).encode()) for m in messages) + 200
        tokens_left = TOKEN_BUDGET - tokens_used - prompt_est - 500
        max_tokens = int(min(MAX_TOKENS, tokens_left, time_left * TOKENS_PER_SECOND))
        if max_tokens < MIN_TOKENS:
            print(f"stopping: max_tokens would be {max_tokens} (time_left={time_left:.0f}s)")
            return
        try:
            reply = llm.chat(messages, temperature=TEMPERATURE, max_tokens=max_tokens)
        except BudgetExceeded as exc:
            print(f"budget exceeded: {exc}")
            return
        except InklingError as exc:
            print(f"model error: {exc}")
            return
        calls += 1
        tokens_used += (reply.prompt_tokens or 0) + (reply.completion_tokens or 0)
        content = str(reply)
        print(
            f"call {calls}: finish={reply.finish_reason} max_tokens={max_tokens} "
            f"completion={reply.completion_tokens} content_chars={len(content)} "
            f"t={time.monotonic() - start:.0f}s tokens={tokens_used}"
        )

        code = extract_code(content)
        if code is None:
            if is_done(content):
                print("model says DONE")
                return
            messages.append({"role": "assistant", "content": content or "(no reply)"})
            if not content.strip() and reply.finish_reason == "length":
                note = (
                    "You ran out of tokens while thinking and sent no code. Stop deliberating: "
                    "choose the simplest reasonable plan right now (you can measure and correct "
                    "after it runs), think only briefly, and reply with one ```python block."
                )
            else:
                note = (
                    "That reply had no runnable code. Write plain Python calling the API "
                    "functions directly (no tool-call or JSON syntax) inside one ```python "
                    "block, or reply DONE if the goal is verified."
                )
            messages.append({"role": "user", "content": note})
            continue

        messages.append({"role": "assistant", "content": content})
        try:
            result = robot.run(code)
        except RunRefused as exc:
            print(f"run refused: {exc}")
            return
        except InklingError as exc:
            print(f"run error: {exc}")
            return
        print(f"run {calls}: ok={result.ok} t={time.monotonic() - start:.0f}s")

        time_left = TIME_BUDGET - TIME_MARGIN - (time.monotonic() - start)
        feedback = (
            f"Code ran ({'no error' if result.ok else 'ERROR'}).\n"
            f"STDOUT:\n{clip(result.stdout) or '(empty)'}\n"
            f"STDERR:\n{clip(result.stderr) or '(empty)'}\n"
            f"Robot state: {describe_state(robot)}\n"
            f"Turns left: {MAX_CALLS - calls - 1}. Seconds left: about {max(0, int(time_left))}.\n"
            "If the printed readings show the goal is fully achieved, reply DONE. Otherwise "
            "reply with the next ```python block (fresh process; re-read poses)."
        )
        messages.append({"role": "user", "content": feedback})
