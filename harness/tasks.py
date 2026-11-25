"""The 7 tasks: CaP-X privileged config, and which robot API functions are reads vs motion.

This is the single declarative source for the per-task API split. It was read off each
privileged API class's ``functions()`` in CaP-X (``capx/integrations/franka/*_privileged.py``):
a function is a *read* if it only looks at observations or sim data, *motion* if it moves an
arm or a gripper (``move_to_joints_blocking*``, ``_set_gripper*`` + ``_step_once``).
``harness/tests/test_tasks.py`` re-checks this against a CaP-X checkout when one is available,
and ``capx_env`` refuses to start if the live API's function set differs.

Reads are served on the agent socket and the runner socket; motion only on the runner socket.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskSpec:
    config: str
    """Privileged YAML config, relative to the CaP-X root."""
    api: str
    """The CaP-X API registered name the config lists under ``apis``."""
    reads: frozenset[str]
    motions: frozenset[str]
    observe_keys: tuple[str, ...]
    """Observation entries returned by ``observe()``: joint and gripper state only."""

    @property
    def functions(self) -> frozenset[str]:
        return self.reads | self.motions


_ONE_ARM_OBS = ("robot_joint_pos", "robot_cartesian_pos")
_TWO_ARM_OBS = (
    "robot0_joint_pos",
    "robot1_joint_pos",
    "robot0_gripper_qpos",
    "robot1_gripper_qpos",
    "robot0_cartesian_pos",
    "robot1_cartesian_pos",
)
_CUBE_READS = frozenset({"get_object_pose", "sample_grasp_pose"})
_CUBE_MOTIONS = frozenset({"goto_pose", "open_gripper", "close_gripper"})
_TWO_ARM_MOTIONS = frozenset(
    {
        "goto_pose_arm0",
        "goto_pose_arm1",
        "open_gripper_arm0",
        "open_gripper_arm1",
        "close_gripper_arm0",
        "close_gripper_arm1",
    }
)

TASKS: dict[str, TaskSpec] = {
    "cube_lifting": TaskSpec(
        config="env_configs/cube_lifting/franka_robosuite_cube_lifting_privileged.yaml",
        api="FrankaControlPrivilegedApi",
        reads=_CUBE_READS,
        motions=_CUBE_MOTIONS,
        observe_keys=_ONE_ARM_OBS,
    ),
    "cube_stack": TaskSpec(
        config="env_configs/cube_stack/franka_robosuite_cube_stack_privileged.yaml",
        api="FrankaControlPrivilegedApi",
        reads=_CUBE_READS,
        motions=_CUBE_MOTIONS,
        observe_keys=_ONE_ARM_OBS,
    ),
    "cube_restack": TaskSpec(
        config="env_configs/cube_restack/franka_robosuite_cube_restack_privileged.yaml",
        api="FrankaControlPrivilegedApi",
        reads=_CUBE_READS,
        motions=_CUBE_MOTIONS,
        observe_keys=_ONE_ARM_OBS,
    ),
    "nut_assembly": TaskSpec(
        config="env_configs/nut_assembly/franka_robosuite_nut_assembly_privileged.yaml",
        api="FrankaControlNutAssemblyPrivilegedApi",
        reads=frozenset({"get_object_pose", "sample_grasp_pose"}),
        motions=frozenset(
            {"goto_pose", "goto_home_joint_position", "open_gripper", "close_gripper"}
        ),
        observe_keys=_ONE_ARM_OBS,
    ),
    "spill_wipe": TaskSpec(
        config="env_configs/spill_wipe/franka_robosuite_spill_wipe_privileged.yaml",
        api="FrankaControlSpillWipePrivilegedApi",
        reads=frozenset({"get_object_pose"}),
        motions=frozenset({"goto_pose"}),
        observe_keys=_ONE_ARM_OBS,
    ),
    "two_arm_lift": TaskSpec(
        config="env_configs/two_arm_lift/franka_robosuite_two_arm_lift_privileged.yaml",
        api="FrankaTwoArmLiftPrivilegedApi",
        reads=frozenset(
            {"get_handle0_pos", "get_handle1_pos", "get_arm0_gripper_pose", "get_arm1_gripper_pose"}
        ),
        motions=_TWO_ARM_MOTIONS | {"goto_pose_both"},
        observe_keys=_TWO_ARM_OBS,
    ),
    "two_arm_handover": TaskSpec(
        config="env_configs/two_arm_handover/two_arm_handover_privileged.yaml",
        api="FrankaHandoverPrivilegedApi",
        reads=frozenset({"get_hammer_pose"}),
        motions=_TWO_ARM_MOTIONS,
        observe_keys=_TWO_ARM_OBS,
    ),
}
