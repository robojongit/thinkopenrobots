"""Scenes are a function of the seed alone (real simulator; ``pytest -m sim``).

For each task, in fresh processes (``scene_probe``): the scene for seed S is bit-identical
whether it is the first scene the process builds, comes after other seeds, or comes from
resetting one env with CaP-X's own ``reset(seed=...)``; different seeds give different scenes;
no random generator other than ``robosuite_env.rng`` is reachable from the env; and read-only
API calls and ``observe()`` change neither any random generator's state nor the simulation.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from harness.tasks import TASKS
from harness.tests.scene_probe import READ_CALLS

ROOT = Path(__file__).resolve().parents[2]
SEED, OTHER = 101, (202, 303)


def probe(task: str, seeds: list[int], *, reuse: bool = False) -> list[dict]:
    cmd = [sys.executable, "-m", "harness.tests.scene_probe", task, *map(str, seeds)]
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    proc = subprocess.run(
        cmd + (["--reuse"] if reuse else []),
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("SCENE_PROBE ")]
    assert proc.returncode == 0 and lines, proc.stdout[-3000:] + proc.stderr[-3000:]
    return json.loads(lines[0].removeprefix("SCENE_PROBE "))


def test_every_read_function_is_probed():
    for task, spec in TASKS.items():
        assert {name for name, _, _ in READ_CALLS[task]} == set(spec.reads), task


@pytest.mark.sim
@pytest.mark.parametrize("task", sorted(TASKS))
def test_scene_depends_only_on_the_seed(task, ik_server):
    alone = probe(task, [SEED])
    after_others = probe(task, [*OTHER, SEED])
    reused = probe(task, [*OTHER, SEED], reuse=True)

    # Same seed: same scene across processes, whatever ran before, fresh env or reset.
    assert alone[0]["scene"] == after_others[-1]["scene"] == reused[-1]["scene"]
    assert [r["scene"] for r in after_others] == [r["scene"] for r in reused]
    # Different seeds: different scenes, and different object poses.
    assert len({r["scene"] for r in after_others}) == 3
    assert len({json.dumps(r["reads"]) for r in after_others}) == 3

    for r in alone + after_others + reused:
        assert r["stray_generators"] == [], r["stray_generators"]
        assert not r["reads_change_rng"], f"seed {r['seed']}: a read advanced an RNG"
        assert not r["reads_change_sim"], f"seed {r['seed']}: a read changed the sim state"
        assert r["reads_repeatable"]
