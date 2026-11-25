"""A fake environment implementing the ``sim_server.EnvAdapter`` interface for any task.

Physics is one cube and point grippers: ``goto*`` moves a gripper (and the cube if held),
``close*`` grabs the cube if the gripper is within 2 cm, ``open*`` drops it. The task counts as
completed when the cube is held more than 4 cm above its start.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from harness.tasks import TASKS

QUAT_DOWN = np.array([0.0, 0.0, 1.0, 0.0])


class FakeEnv:
    def __init__(self, task: str = "cube_lifting", seed: int = 0, render: bool = False) -> None:
        self.spec = TASKS[task]
        rng = np.random.default_rng(seed)
        self.cube = np.array([0.5 + rng.uniform(-0.05, 0.05), rng.uniform(-0.05, 0.05), 0.025])
        self.start_z = float(self.cube[2])
        self.gripper = np.array([0.3, 0.0, 0.4])
        self.holding = False
        self.motions: list[tuple[str, tuple, dict]] = []
        self.checks = 0
        self.closed = False
        self.instruction = f"[fake {task}] Pick up the red cube and lift it."
        self.functions: dict[str, Any] = {}
        for name in self.spec.reads:
            self.functions[name] = self._read
        for name in self.spec.motions:
            self.functions[name] = self._motion_fn(name)
        if "get_object_pose" in self.spec.reads:
            self.functions["get_object_pose"] = self.get_object_pose
        self.api_docs = "\n".join(f"{n}(...)" for n in sorted(self.functions))

    def get_object_pose(
        self, object_name: str, return_bbox_extent: bool = False
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        if "red" not in object_name or "cube" not in object_name:
            raise ValueError(f"Invalid object name: {object_name}")
        extent = np.array([0.05, 0.05, 0.05]) if return_bbox_extent else None
        return self.cube.astype(np.float32), QUAT_DOWN.copy(), extent

    def _read(self, *args: Any, **kwargs: Any) -> tuple[np.ndarray, np.ndarray]:
        return self.cube.astype(np.float32), QUAT_DOWN.copy()

    def _motion_fn(self, name: str) -> Any:
        def fn(*args: Any, **kwargs: Any) -> None:
            self.motions.append((name, args, kwargs))
            if name.startswith("goto_pose") and args:
                self.gripper = np.asarray(args[0], dtype=np.float64).reshape(3)
                if self.holding:
                    self.cube = self.gripper.copy()
            elif name.startswith("close_gripper"):
                self.holding = bool(np.linalg.norm(self.gripper - self.cube) < 0.02)
            elif name.startswith("open_gripper"):
                self.holding = False
                self.cube[2] = self.start_z

        return fn

    def observe(self) -> dict[str, Any]:
        cart = np.concatenate([self.gripper, QUAT_DOWN, [0.0 if self.holding else 1.0]])
        return {"robot_joint_pos": np.zeros(8), "robot_cartesian_pos": cart}

    def task_completed(self) -> bool:
        self.checks += 1
        return self.holding and self.cube[2] > self.start_z + 0.04

    def write_video(self, path: Path) -> bool:
        return False

    def close(self) -> None:
        self.closed = True
