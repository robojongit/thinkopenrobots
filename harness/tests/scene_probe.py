"""Build CaP-X scenes and print a JSON report about them (used by ``test_determinism.py``).

    python -m harness.tests.scene_probe TASK SEED [SEED ...] [--reuse]

For each seed, in order and in this one process: build a fresh ``CapxEnv`` (or, with
``--reuse``, build one env and ``reset(seed=...)`` it, as CaP-X's own trial loop does) and
record a digest of the initial scene, the stray random generators reachable from the env, and
whether calling every read-only API function and ``observe()`` changed any random generator's
state or the simulation state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from typing import Any

import numpy as np

from harness.seeding import find_stray_generators
from harness.tasks import TASKS

_CUBE = ("red cube", "green cube")
_NUT = (
    "extruded handle of the brown square nut",
    "white hollow center of the brown square nut",
    "square block",
)

# Read-only calls made per task: (function, args, kwargs). Covers every read function.
READ_CALLS: dict[str, list[tuple[str, tuple, dict]]] = {
    "cube_lifting": [
        ("get_object_pose", ("red cube",), {"return_bbox_extent": True}),
        ("sample_grasp_pose", ("red cube",), {}),
    ],
    "cube_stack": [
        *[("get_object_pose", (c,), {"return_bbox_extent": True}) for c in _CUBE],
        *[("sample_grasp_pose", (c,), {}) for c in _CUBE],
    ],
    "cube_restack": [
        *[("get_object_pose", (c,), {"return_bbox_extent": True}) for c in _CUBE],
        *[("sample_grasp_pose", (c,), {}) for c in _CUBE],
    ],
    "nut_assembly": [
        *[("get_object_pose", (n,), {}) for n in _NUT],
        ("sample_grasp_pose", (_NUT[0],), {}),
        ("sample_grasp_pose", (_NUT[2],), {}),
    ],
    "spill_wipe": [("get_object_pose", ("brown spill",), {"return_bbox_extent": True})],
    "two_arm_lift": [
        ("get_handle0_pos", (), {}),
        ("get_handle1_pos", (), {}),
        ("get_arm0_gripper_pose", (), {}),
        ("get_arm1_gripper_pose", (), {}),
    ],
    "two_arm_handover": [("get_hammer_pose", (), {})],
}

# MuJoCo arrays that a scene's random choices can touch (model: object sizes and fixed
# placements such as the spill; data: every joint, free-body and actuator state).
_MODEL_FIELDS = ("body_pos", "body_quat", "geom_size", "geom_pos", "geom_quat", "site_pos")
_DATA_FIELDS = ("qpos", "qvel", "act", "mocap_pos", "mocap_quat")


def _arrays(value: Any) -> list[np.ndarray]:
    if value is None:
        return []
    if isinstance(value, (tuple, list)):
        return [a for v in value for a in _arrays(v)]
    if isinstance(value, dict):
        return [a for k in sorted(value) for a in _arrays(value[k])]
    return [np.asarray(value)]


def _digest(arrays: list[np.ndarray]) -> str:
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(f"{a.dtype.str}{a.shape}".encode())
        h.update(a.tobytes())
    return h.hexdigest()


def _sim_arrays(robosuite_env: Any) -> list[np.ndarray]:
    sim = robosuite_env.sim
    return [np.array(getattr(sim.model, f)) for f in _MODEL_FIELDS] + [
        np.array(getattr(sim.data, f)) for f in _DATA_FIELDS
    ]


def _rng_states(robosuite_env: Any) -> str:
    return json.dumps(
        [
            robosuite_env.rng.bit_generator.state,
            [x.tolist() if isinstance(x, np.ndarray) else x for x in np.random.get_state()],
            random.getstate(),
        ],
        default=str,
    )


def probe(env: Any, task: str) -> dict[str, Any]:
    """Report on ``env`` (a ``CapxEnv``) right after its reset."""
    robosuite_env = env.env.low_level_env.robosuite_env
    sim_before = _sim_arrays(robosuite_env)
    rng_before = _rng_states(robosuite_env)
    reads = [env.functions[name](*args, **kwargs) for name, args, kwargs in READ_CALLS[task]]
    observed = env.observe()
    reads_again = [env.functions[name](*a, **kw) for name, a, kw in READ_CALLS[task]]
    sim_digest = _digest(sim_before)
    return {
        "scene": _digest(sim_before + _arrays(reads) + _arrays(observed)),
        "sim": sim_digest,
        "reads": [np.round(a.astype(float), 4).tolist() for a in _arrays(reads)],
        "reads_change_rng": _rng_states(robosuite_env) != rng_before,
        "reads_change_sim": _digest(_sim_arrays(robosuite_env)) != sim_digest,
        "reads_repeatable": _digest(_arrays(reads)) == _digest(_arrays(reads_again)),
        "stray_generators": find_stray_generators(env.env, robosuite_env.rng),
        "covered_reads": sorted({name for name, _, _ in READ_CALLS[task]}),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=sorted(TASKS))
    parser.add_argument("seeds", type=int, nargs="+")
    parser.add_argument("--reuse", action="store_true", help="one env, reset per seed")
    args = parser.parse_args(argv)

    from harness.capx_env import CapxEnv

    reports = []
    if args.reuse:
        env = CapxEnv(args.task, args.seeds[0])
        for i, seed in enumerate(args.seeds):
            if i:  # CaP-X's own reset(seed) only, as in its trial loop
                env.env.reset(seed=seed)
            reports.append({"seed": seed, **probe(env, args.task)})
        env.close()
    else:
        for seed in args.seeds:
            env = CapxEnv(args.task, seed)
            reports.append({"seed": seed, **probe(env, args.task)})
            env.close()
    print("SCENE_PROBE " + json.dumps(reports))
    return 0


if __name__ == "__main__":
    sys.exit(main())
