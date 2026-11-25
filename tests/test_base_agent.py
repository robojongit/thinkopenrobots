"""The base solution (``base/``) and the participant starting point (``agent/``), no simulator.

Prompt construction and code extraction are checked against literal expectations here and
against CaP-X's own functions in ``test_base_sim.py``.
"""

from __future__ import annotations

import pytest

from bench.train import agent_fingerprint
from harness.llm_relay import Budget
from harness.tests.fake_env import FakeEnv
from harness.tests.fake_model import FakeModel
from tests.base_agent import AGENT_DIR, BASE_DIR, load_agent, run_base

base = load_agent()

LIFT_CODE = """
import numpy as np
# Grasp the cube from above, then lift it 10 cm.
pos, quat = sample_grasp_pose("red cube")
open_gripper()
goto_pose(pos, quat, z_approach=0.1)
close_gripper()
goto_pose(pos + np.array([0.0, 0.0, 0.1]), quat)
print("lifted")
"""


def code_files(folder):
    return {
        p.relative_to(folder).as_posix(): p.read_bytes()
        for p in sorted(folder.rglob("*.py"))
        if "__pycache__" not in p.parts
    }


def test_agent_code_is_a_copy_of_base():
    assert set(code_files(BASE_DIR)) == {"agent.py"}
    assert code_files(AGENT_DIR) == code_files(BASE_DIR)
    assert agent_fingerprint(AGENT_DIR) == agent_fingerprint(BASE_DIR)
    for name in ("NOTES.md", "memory/README.md"):
        assert (AGENT_DIR / name).is_file()


def test_messages_are_capx_full_prompt():
    assert base.build_messages("Goal: lift.", "goto_pose(pos)\n  Doc:\n    Move.") == [
        {
            "role": "system",
            "content": "You are a helpful assistant that generates Python code to directly solve"
            " the task.",
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Goal: lift.\nAPIs:\ngoto_pose(pos)\n  Doc:\n    Move."}
            ],
        },
    ]
    assert (base.TEMPERATURE, base.MAX_TOKENS) == (1.0, 20480)


@pytest.mark.parametrize(
    "reply, code",
    [
        ("x = 1\nprint(x)\n", "x = 1\nprint(x)"),
        ("Plan:\n```python\nx = 1\n```\nDone.", "x = 1"),
        (
            "```python\na = 1\n```\ntext\n```python\nb = 2\n```",
            "a = 1\n```\ntext\n```python\nb = 2",
        ),
        ("```python\nx = 1\n", "x = 1"),
        ("```py\nx = 1\n```", "```py\nx = 1"),
        ("```\nx = 1\n```", "```\nx = 1"),
        ("  \n", ""),
        ("", ""),
    ],
)
def test_extract_code(reply, code):
    assert base.extract_code(reply) == [code]
    assert code in reply  # robot.run accepts only substrings of a reply


def episode(tmp_path, reply: str, **kwargs):
    env = FakeEnv("cube_lifting", 3)
    with FakeModel([reply], prompt_tokens=1200, completion_tokens=300) as model:
        ep = run_base(
            "cube_lifting",
            3,
            model.url,
            tmp_path / "out",
            make_env=lambda t, s, r: env,
            budget=Budget(wall_s=60),
            **kwargs,
        )
    return ep, env, model


@pytest.mark.parametrize("agent_dir", [BASE_DIR, AGENT_DIR], ids=["base", "agent"])
@pytest.mark.parametrize("fenced", [False, True], ids=["bare", "fenced"])
def test_one_call_one_run_success(tmp_path, agent_dir, fenced):
    reply = f"Lifting.\n```python\n{LIFT_CODE}```\n" if fenced else LIFT_CODE
    ep, env, model = episode(tmp_path, reply, agent_dir=agent_dir)

    (request,) = model.requests
    assert request["messages"] == base.build_messages(env.instruction, env.api_docs)
    assert request["temperature"] == 1.0 and request["max_tokens"] == 20480
    (run,) = ep.json("runs.json")
    assert run["accepted"] and run["ok"] and run["stdout"] == "lifted\n"
    assert (ep.out / run["code_file"]).read_text() == LIFT_CODE.strip()
    assert [m[0] for m in env.motions] == ["open_gripper", "goto_pose", "close_gripper",
                                           "goto_pose"]  # fmt: skip
    assert ep.result["success"] is True and ep.result["ended"] == "returned"
    assert ep.result["calls"] == 1 and ep.result["tokens"] == 1500
    assert "code block 0: ok=True" in ep.log


def test_failing_code_gets_no_feedback_or_retry(tmp_path):
    ep, env, model = episode(tmp_path, "open_gripper()\nraise RuntimeError('boom')\n")
    assert len(model.requests) == 1
    (run,) = ep.json("runs.json")
    assert run["accepted"] and not run["ok"] and "RuntimeError: boom" in run["stderr"]
    assert [m[0] for m in env.motions] == ["open_gripper"]
    assert ep.result["success"] is False and ep.result["ended"] == "returned"


def test_blank_reply_runs_nothing(tmp_path):
    ep, env, model = episode(tmp_path, "```python\n\n```")
    assert len(model.requests) == 1 and ep.json("runs.json") == []
    assert env.motions == [] and env.checks == 1
    assert ep.result["success"] is False and ep.result["ended"] == "returned"
    assert "code block 0: empty" in ep.log


def test_model_server_error_ends_the_episode(tmp_path):
    ep, env, model = episode(tmp_path, lambda body: (503, {"error": "down"}))
    assert len(model.requests) == 1 and ep.json("runs.json") == []
    assert ep.result["success"] is False and ep.result["ended"] == "error"
    assert "LLMError" in ep.log
