"""End-to-end episodes: FakeEnv + fake model + tiny agents, plain-process sandbox."""

import json

import pytest

from harness.episode import run_episode
from harness.llm_relay import Budget, estimate_prompt_tokens
from harness.sandbox import ProcessSandbox
from harness.tests.conftest import CHAT_AND_RUN, LIFT_REPLY
from harness.tests.fake_env import FakeEnv
from harness.tests.fake_model import FakeModel


def test_successful_episode_and_run_folder(make_agent, episode):
    agent = make_agent(
        """
        import re
        from inkling_robo import Task, Robot, LLM

        def solve(task: Task, robot: Robot, llm: LLM) -> None:
            print("pose:", robot.api.get_object_pose("red cube"))
            print("obs:", sorted(robot.observe()))
            reply = llm.chat(
                [{"role": "user", "content": task.instruction + "\\n" + task.api_docs}],
                temperature=1.0,
                max_tokens=2048,
            )
            code = re.search(r"```python\\n(.*?)```", reply, re.S).group(1)
            result = robot.run(code)
            print("run:", result.ok, repr(result.stdout), repr(result.stderr))
        """
    )
    with FakeModel([LIFT_REPLY], prompt_tokens=120, completion_tokens=80) as model:
        ep = episode(agent, model)

    assert ep.result == {
        "task": "cube_lifting",
        "seed": 3,
        "success": True,
        "tokens": 200,
        "calls": 1,
        "seconds": ep.result["seconds"],
        "ended": "returned",
        "crash": None,
    }
    assert 0 < ep.result["seconds"] < 30
    assert ep.env.checks == 1 and ep.env.closed
    assert [m[0] for m in ep.env.motions] == ["open_gripper", "goto_pose", "close_gripper",
                                              "goto_pose"]  # fmt: skip
    assert "run: True 'lifted ndarray\\n' ''" in ep.log
    assert "pose: (array([" in ep.log  # tuples of arrays, as in CaP-X
    assert "obs: ['robot_cartesian_pos', 'robot_joint_pos']" in ep.log

    out = ep.out
    names = {p.name for p in out.iterdir()}
    assert names == {"result.json", "usage.json", "transcript.jsonl", "llm.jsonl", "runs.json",
                     "errors.json", "agent.log", "code"}  # fmt: skip
    record = ep.json("result.json")
    assert "seed" not in record and "3" not in json.dumps(record["task"])
    assert record["success"] and record["task_completed"] and record["limit_hit"] is None
    assert ep.json("usage.json") == {
        "calls": 1,
        "prompt_tokens": 120,
        "completion_tokens": 80,
        "tokens": 200,
        "seconds": ep.result["seconds"],
        "budget": {"calls": 10, "tokens": 66000, "wall_s": 30},
        "limit_hit": None,
        "upstream_errors": 0,
        "upstream_failures": [],
    }
    (run,) = ep.json("runs.json")
    assert run["accepted"] and run["ok"] and run["code_file"] == "code/run_000.py"
    assert (out / "code" / "run_000.py").read_text() in LIFT_REPLY
    (llm,) = ep.jsonl("llm.jsonl")
    assert llm["reply"]["content"] == LIFT_REPLY and llm["request"]["max_tokens"] == 2048
    events = ep.jsonl("transcript.jsonl")
    kinds = [(e["kind"], e.get("source"), e.get("name")) for e in events]
    assert ("llm", None, None) in kinds
    assert ("call", "agent", "get_object_pose") in kinds
    assert ("call", "runner", "goto_pose") in kinds
    assert all(e["source"] == "runner" for e in events if e.get("name") in ep.env.spec.motions)
    assert ep.json("errors.json") == []
    assert model.requests[0]["model"] == "inkling-2-mini"


def test_model_code_errors_come_back_as_stderr(make_agent, episode):
    reply = "```python\nprint('start')\nget_object_pose('blue cube')\n```"
    with FakeModel([reply]) as model:
        ep = episode(make_agent(CHAT_AND_RUN), model)
    assert ep.result["success"] is False and ep.result["ended"] == "returned"
    assert "run: False 'start\\n'" in ep.log
    assert 'File "<string>", line 2' in ep.log
    assert "ValueError: Invalid object name: blue cube" in ep.log
    (run,) = ep.json("runs.json")
    assert run["ok"] is False and run["exit_status"] == 1


def test_a_skills_file_in_the_agent_folder_never_runs(make_agent, episode, tmp_path):
    # Participant code never runs in the code runner: only the model's code, with the task's
    # API functions in scope. A skills.py that would have been loaded there before is ignored.
    marker = tmp_path / "skills-ran"
    agent = make_agent(CHAT_AND_RUN)
    (agent / "skills.py").write_text(
        f"open({str(marker)!r}, 'w').close()\ndef lift(name, height):\n    return 'lifted'\n"
    )
    code = "import sys\nprint('lift' in globals(), 'skills' in globals(), 'skills' in sys.modules)"
    with FakeModel([f"```python\n{code}\n```"]) as model:
        ep = episode(agent, model)
    assert "run: True 'False False False\\n' ''" in ep.log
    assert not marker.exists()


def test_agent_exception_ends_the_episode_as_error(make_agent, episode):
    agent = make_agent(
        """
        def solve(task, robot, llm):
            raise RuntimeError("agent bug")
        """
    )
    with FakeModel(["x"]) as model:
        ep = episode(agent, model)
    assert ep.result["ended"] == "error" and ep.result["success"] is False
    assert "RuntimeError: agent bug" in ep.log
    assert ep.env.checks == 1


def test_two_arm_task_split(make_agent, episode):
    agent = make_agent(
        """
        from inkling_robo import Forbidden

        def solve(task, robot, llm):
            print("hammer:", robot.api.get_hammer_pose()[0].shape)
            print("dir:", dir(robot.api))
            try:
                robot.api.goto_pose_arm0([0, 0, 0], [0, 0, 1, 0])
            except Forbidden:
                print("arm0 forbidden")
            reply = llm.chat([{"role": "user", "content": "go"}])
            print(robot.run(reply.split("```python\\n")[1].split("```")[0]))
        """
    )
    reply = "```python\ngoto_pose_arm1([0.6, 0, 0.2], [0, 0, 1, 0])\n```"
    with FakeModel([reply]) as model:
        ep = episode(agent, model, task="two_arm_handover")
    assert "hammer: (3,)" in ep.log
    assert "dir: ['get_hammer_pose']" in ep.log
    assert "arm0 forbidden" in ep.log
    assert [m[0] for m in ep.env.motions] == ["goto_pose_arm1"]


def test_budget_is_reported(make_agent, episode):
    with FakeModel([LIFT_REPLY]) as model:
        ep = episode(make_agent(CHAT_AND_RUN), model, budget=Budget(calls=1, tokens=500, wall_s=20))
    assert ep.json("usage.json")["budget"] == {"calls": 1, "tokens": 500, "wall_s": 20}
    bound = estimate_prompt_tokens(model.requests[0]["messages"])
    assert model.requests[0]["max_tokens"] == 500 - bound


class BrokenChecker(FakeEnv):
    def task_completed(self) -> bool:
        super().task_completed()
        raise MemoryError("checker blew up")


class BrokenClose(FakeEnv):
    def close(self) -> None:
        raise RuntimeError("close failed")


class BrokenSandbox(ProcessSandbox):
    def start_agent(self, agent_dir, socket_path):
        raise OSError("cannot start the agent")


@pytest.mark.parametrize(
    "env_class, sandbox, reason",
    [
        (BrokenChecker, ProcessSandbox, "MemoryError: checker blew up"),
        (BrokenClose, ProcessSandbox, "RuntimeError: close failed"),
        (FakeEnv, BrokenSandbox, "OSError: cannot start the agent"),
    ],
)
def test_harness_crash_in_an_episode_is_a_failure_with_a_reason(
    make_agent, tmp_path, env_class, sandbox, reason
):
    # A crash must not be an exception a caller could answer with a re-run (crash-to-reroll):
    # the episode is failed, with the reason recorded.
    env = env_class("cube_lifting", 3)
    with FakeModel([LIFT_REPLY]) as model:
        result = run_episode(
            "cube_lifting",
            3,
            make_agent(CHAT_AND_RUN),
            model.url,
            Budget(wall_s=30),
            tmp_path / "out",
            make_env=lambda t, s, r: env,
            sandbox=sandbox,
        )
    assert result["ended"] == "crash" and result["success"] is False
    assert result["crash"] == reason
    record = json.loads((tmp_path / "out" / "result.json").read_text())
    assert record["crash"] == reason and record["success"] is False
    assert (tmp_path / "out" / "transcript.jsonl").exists()


def test_env_build_failure_still_raises(make_agent, tmp_path):
    def no_env(task, seed, render):
        raise RuntimeError("no simulator")

    with pytest.raises(RuntimeError, match="no simulator"):
        run_episode("cube_lifting", 1, make_agent("def solve(t, r, l): pass"), "http://x",
                    Budget(), tmp_path / "out", make_env=no_env)  # fmt: skip
