"""CaP-X's reference solutions run through the harness on the real simulator (``pytest -m sim``).

Seed 1 is solved by every task's oracle, through the harness and through CaP-X's own launch
path. For success rates over more seeds, and next to CaP-X's launch path, use
``python -m harness.tests.reference --seeds 1-10 --native``.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from harness.tasks import TASKS
from harness.tests.reference import oracle_code, run_reference

pytestmark = pytest.mark.sim


@pytest.mark.parametrize("task", sorted(TASKS))
def test_reference_solution_solves_seed_1(task, tmp_path, ik_server):
    result = run_reference(task, 1, tmp_path)
    episode = tmp_path / "episode"
    (run,) = json.loads((episode / "runs.json").read_text())
    assert run["accepted"] and run["ok"], (episode / "agent.log").read_text()
    assert (episode / "code" / "run_000.py").read_text().strip() == oracle_code(task).strip()
    assert result["ended"] == "returned" and result["calls"] == 1
    assert result["success"] is True
    assert not (episode / "video.mp4").exists()


def test_video_is_a_playable_mp4_of_the_scene(tmp_path, ik_server):
    import imageio

    result = run_reference("cube_lifting", 3, tmp_path, video=True)
    assert result["success"] is True
    reader = imageio.get_reader(tmp_path / "episode" / "video.mp4")
    frames = [reader.get_data(i) for i in range(reader.count_frames())]
    assert len(frames) > 20 and frames[0].shape == (512, 512, 3)
    # The scene is mostly a white table; a frame rendered off the GL context's thread is noise.
    assert all(f.mean() > 150 for f in frames)
    assert np.abs(frames[0].astype(int) - frames[-1].astype(int)).mean() > 1  # the arm moved


def test_rendering_is_off_without_video(ik_server):
    from harness.capx_env import CapxEnv

    env = CapxEnv("cube_lifting", 1)
    try:
        robosuite_env = env.env.low_level_env.robosuite_env
        assert not robosuite_env.has_offscreen_renderer and not robosuite_env.use_camera_obs
        assert env.write_video(None) is False  # type: ignore[arg-type]
    finally:
        env.close()
