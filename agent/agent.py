"""The base solution: CaP-X's vanilla harness, ported unchanged into ``solve()``.

This reproduces what CaP-X's ``capx/envs/launch.py`` does for the 7 RoboSuite privileged configs
(``env_configs/<task>/*_privileged.yaml``). Paths below are in the CaP-X fork at
``harness/capx``.

What CaP-X does per trial for these configs:

1. One model call. The prompt is a system message and one user message whose text is
   ``f"{task_prompt}\\nAPIs:\\n" + "\\n".join(docs)`` (``capx/envs/tasks/base.py:115-121`` and
   ``:139-151``); ``task.instruction`` is that ``task_prompt`` and ``task.api_docs`` that joined
   ``docs``. No in-context examples beyond what the task prompt and the API docstrings contain.
   The query goes through ``_query_initial_code`` (``capx/envs/trial.py:469-498``, no ensemble)
   with ``temperature=1.0`` and ``max_tokens=2048 * 10`` (``capx/envs/launch.py:55-59``), sent
   as ``{model, temperature, max_tokens, messages}`` for a model not in CaP-X's named lists
   (``capx/llm/client.py:232-238``).
2. Code extraction with ``_extract_code`` (``capx/utils/launch_utils.py:165-189``): the text
   after the first "```python\\n" and before the last "```", stripped; with no fences, the whole
   reply. It always returns exactly one block.
3. One execution of that block (``capx/envs/trial.py:780-787``), then the trial ends. The
   multi-turn branch (``capx/envs/trial.py:804``) only runs when the YAML config sets
   ``multi_turn_prompt`` (read at ``capx/envs/runner.py:179`` and ``:259``); none of the 7
   privileged configs does, so nothing (stdout, stderr, errors) is ever fed back to the model.

Deviations, all forced by the leaderboard runner:

- The code runs through ``robot.run`` in a fresh process that has only the task's API functions
  in scope. CaP-X's exec namespace also held ``env``, ``APIS``, ``obs``, ``INPUTS`` and
  ``RESULT`` (``capx/envs/tasks/base.py:153-162``, ``:187-205``); the vanilla prompts never
  mention them, but model code that uses them fails here with a NameError. Persisting variables
  between blocks does not matter: there is only one block.
- A blank extracted block is not sent to ``robot.run`` (which refuses blank code); CaP-X would
  exec the empty string, which does nothing either.
- Model-server errors are not retried here. CaP-X retries 404/5xx answers every ~4 minutes
  (``capx/llm/client.py:246-258``) and reruns a whole trial on a 1000 s timeout
  (``capx/envs/runner.py:39-40``, ``:195-232``); here ``llm.chat`` raises, the episode ends,
  and ``bench train`` reruns episodes that failed because of the model server.
"""

from __future__ import annotations

from inkling_robo import LLM, Robot, Task

# capx/envs/tasks/base.py:115-117
SYSTEM_PROMPT = "You are a helpful assistant that generates Python code to directly solve the task."
# capx/envs/launch.py:55-59 (LaunchArgs defaults; the privileged configs do not override them)
TEMPERATURE = 1.0
MAX_TOKENS = 2048 * 10


def build_messages(instruction: str, api_docs: str) -> list[dict]:
    """CaP-X's ``_full_prompt`` (``capx/envs/tasks/base.py:118-121``, ``:139-151``)."""
    complete_prompt = f"{instruction}\nAPIs:\n{api_docs}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [{"type": "text", "text": complete_prompt}]},
    ]


def extract_code(content: str) -> list[str]:
    """CaP-X's ``_extract_code`` (``capx/utils/launch_utils.py:165-189``), unchanged.

    The result is a substring of ``content``, so ``robot.run`` accepts it.
    """
    fence_start = "```python\n"
    fence_end = "```"
    if fence_start in content:
        content = content[content.find(fence_start) + len(fence_start) :]
    if fence_end in content:
        content = content[: content.rfind(fence_end)]
    return [str(content).strip()]


def solve(task: Task, robot: Robot, llm: LLM) -> None:
    messages = build_messages(task.instruction, task.api_docs)
    reply = llm.chat(messages, temperature=TEMPERATURE, max_tokens=MAX_TOKENS)
    print(
        f"reply: finish_reason={reply.finish_reason} prompt_tokens={reply.prompt_tokens} "
        f"completion_tokens={reply.completion_tokens}"
    )
    for index, code in enumerate(extract_code(reply)):
        if not code:
            print(f"code block {index}: empty, nothing to run")
            continue
        result = robot.run(code)
        print(f"code block {index}: ok={result.ok}")
