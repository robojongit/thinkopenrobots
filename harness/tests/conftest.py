import json
import textwrap
from pathlib import Path

import pytest

from harness.episode import run_episode
from harness.llm_relay import Budget
from harness.tests.fake_env import FakeEnv

LIFT_REPLY = """I will lift the cube.
```python
import numpy as np
pos, quat = sample_grasp_pose("red cube")
open_gripper()
goto_pose(pos, quat, z_approach=0.1)
close_gripper()
goto_pose(pos + np.array([0.0, 0.0, 0.1]), quat)
print("lifted", type(pos).__name__)
```
"""

CHAT_AND_RUN = """
import re

def solve(task, robot, llm):
    reply = llm.chat(
        [{"role": "user", "content": task.instruction + "\\n" + task.api_docs}],
        temperature=1.0,
        max_tokens=2048,
    )
    code = re.search(r"```python\\n(.*?)```", reply, re.S).group(1)
    result = robot.run(code)
    print("run:", result.ok, repr(result.stdout), repr(result.stderr))
"""


@pytest.fixture
def make_agent(tmp_path):
    def make(source: str, name: str = "agent") -> Path:
        d = tmp_path / name
        d.mkdir()
        (d / "agent.py").write_text(textwrap.dedent(source))
        return d

    return make


class Episode:
    def __init__(self, result: dict, env: FakeEnv, out: Path) -> None:
        self.result = result
        self.env = env
        self.out = out

    def json(self, name: str):
        return json.loads((self.out / name).read_text())

    def jsonl(self, name: str) -> list:
        return [json.loads(line) for line in (self.out / name).read_text().splitlines()]

    @property
    def log(self) -> str:
        return (self.out / "agent.log").read_text()


@pytest.fixture
def episode(tmp_path):
    def run(
        agent_dir: Path,
        model,
        budget: Budget | None = None,
        task: str = "cube_lifting",
        seed: int = 3,
    ) -> Episode:
        env = FakeEnv(task, seed)
        out = tmp_path / "out"
        result = run_episode(
            task,
            seed,
            agent_dir,
            model.url,
            budget or Budget(wall_s=30),
            out,
            make_env=lambda t, s, r: env,
        )
        return Episode(result, env, out)

    return run


@pytest.fixture(scope="session")
def ik_server():
    """The real simulator's prerequisites: skip if CaP-X is not installed; reuse a running
    PyRoKi IK server or start one for the session (subprocesses reuse it)."""
    try:
        from harness.capx_env import import_capx

        import_capx()
        import robosuite  # noqa: F401
    except Exception as e:  # missing submodule or sim group
        pytest.skip(f"simulator not installed: {e}")
    from harness.ik_server import ensure_ik_server, stop_ik_server

    ensure_ik_server()
    yield
    stop_ik_server()
