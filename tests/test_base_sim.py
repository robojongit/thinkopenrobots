"""The base solution against CaP-X itself and on the real simulator (``pytest -m sim``).

- The messages ``base`` builds from ``Task`` equal the prompt CaP-X's trial loop sends, for all
  7 tasks (``capx/envs/trial.py:652-659``: ``obs["full_prompt"]`` is the env's ``_full_prompt``).
- ``base.extract_code`` gives the same output as CaP-X's ``_extract_code`` on the same replies.
- On the real simulator, a scripted model answering with a task's oracle code, in the bare-code
  format CaP-X's prompts ask for, goes through extraction and ``robot.run`` to success.
"""

from __future__ import annotations

import copy
import json

import pytest

from harness.tasks import TASKS
from tests.base_agent import load_agent, run_base

pytestmark = pytest.mark.sim

base = load_agent()


def reply_variants(code: str) -> list[str]:
    """Replies a model might give around ``code``: bare, fenced, with prose, several blocks,
    unterminated, a wrong fence tag, blank."""
    return [
        code,
        f"```python\n{code}```",
        f"Here is the plan.\n```python\n{code}\n```\nThis should work.",
        f"```python\n{code}\n```\nOr, alternatively:\n```python\nprint('second')\n```\n",
        f"```python\n{code}",
        f"```py\n{code}\n```",
        f"```\n{code}\n```",
        "",
        "```python\n```",
    ]


@pytest.mark.parametrize("task", sorted(TASKS))
def test_prompt_and_extraction_match_capx(task, ik_server):
    from capx.utils.launch_utils import _extract_code

    from harness.capx_env import CapxEnv
    from harness.tests.reference import oracle_code

    env = CapxEnv(task, 1)
    try:
        capx_prompt = env._on_env_thread(
            lambda: copy.deepcopy(env.env._get_observation()["full_prompt"])
        )
        assert base.build_messages(env.instruction, env.api_docs) == capx_prompt
        assert env.instruction == env.env._task_prompt
    finally:
        env.close()

    for reply in reply_variants(oracle_code(task)):
        assert base.extract_code(reply) == _extract_code(reply), reply


@pytest.mark.parametrize("task", ["cube_lifting", "two_arm_lift"])
def test_oracle_reply_succeeds_on_the_simulator(task, tmp_path, ik_server):
    from harness.tests.reference import oracle_code
    from tests.oracle_model import OracleModel

    with OracleModel([task]) as model:
        ep = run_base(task, 1, model.url, tmp_path / "out")  # the default, final budget

    assert model.answered == [task]
    (request,) = model.requests
    assert request["temperature"] == 1.0 and request["max_tokens"] == 20480
    (run,) = ep.json("runs.json")
    assert run["accepted"] and run["ok"], ep.log + json.dumps(run)
    assert (ep.out / run["code_file"]).read_text() == oracle_code(task).strip()
    assert ep.result["calls"] == 1 and ep.result["ended"] == "returned"
    assert ep.result["success"] is True
