"""Defences that need the real simulator (``pytest -m sim``): bad motion arguments never reach
CaP-X, and a MuJoCo blow-up (which silently resets the scene) fails the episode."""

from __future__ import annotations

import json
import time

import numpy as np
import pytest

from harness.capx_env import CapxEnv
from harness.tests.reference import run_reference

pytestmark = pytest.mark.sim


def test_nan_motion_is_refused_before_capx(tmp_path, ik_server):
    code = "goto_pose([float('nan'), 0.0, 0.3], [0.0, 0.0, 1.0, 0.0])\n"
    t0 = time.monotonic()
    result = run_reference("cube_lifting", 1, tmp_path, code=code)
    (run,) = json.loads((tmp_path / "episode" / "runs.json").read_text())
    assert "ValueError: motion arguments must be finite" in run["stderr"]
    assert run["seconds"] < 5  # no IK retries under the env lock
    assert result["success"] is False and time.monotonic() - t0 < 120


def test_mujoco_auto_reset_fails_the_episode(tmp_path, ik_server, monkeypatch):
    monkeypatch.chdir(tmp_path)  # MuJoCo writes MUJOCO_LOG.TXT into the working directory
    env = CapxEnv("cube_lifting", 1)
    try:

        def blow_up():
            env.env.low_level_env.robosuite_env.sim.data._data.qvel[:] = np.nan

        env._on_env_thread(blow_up)
        with pytest.raises(RuntimeError, match="unstable"):
            env.functions["open_gripper"]()
        with pytest.raises(RuntimeError, match="unstable"):  # and every call after
            env.functions["get_object_pose"]("red cube")
        assert env.unstable and env.task_completed() is False
    finally:
        env.close()


def test_checker_waits_for_the_scene_to_settle(ik_server):
    # A cube dropped from 20 cm is "lifted" at the instant of the drop; after the settle steps
    # it lies on the table again.
    env = CapxEnv("cube_lifting", 1)
    try:
        rs = env.env.low_level_env.robosuite_env

        def drop_cube() -> bool:
            joint = rs.cube.joints[0]
            pose = rs.sim.data.get_joint_qpos(joint).copy()
            pose[2] = rs.model.mujoco_arena.table_offset[2] + 0.2
            rs.sim.data.set_joint_qpos(joint, pose)
            rs.sim.forward()
            return bool(rs._check_success())

        assert env._on_env_thread(drop_cube) is True  # what the old single check saw
        assert env.task_completed() is False
    finally:
        env.close()


def test_checker_computes_the_reward_first_like_capx(ik_server, monkeypatch):
    # CaP-X's step() calls compute_reward() before task_completed(); for spill_wipe the reward
    # is what records the wiped markers. (robosuite also does it on every control step, so on
    # the 70 reference episodes the verdict was the same either way.)
    env = CapxEnv("spill_wipe", 1)
    try:
        order = []
        reward, check = env.env.compute_reward, env.env.low_level_env.task_completed
        monkeypatch.setattr(env.env, "compute_reward", lambda: order.append("reward") or reward())
        monkeypatch.setattr(
            env.env.low_level_env, "task_completed", lambda: order.append("check") or check()
        )
        assert env.task_completed() is False
        assert order == ["reward", "check"]
    finally:
        env.close()
