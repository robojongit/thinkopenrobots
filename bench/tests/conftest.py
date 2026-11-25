import json
from collections.abc import Callable
from pathlib import Path

import pytest

from harness.tasks import TASKS
from harness.tests.conftest import make_agent  # noqa: F401  (fixture reused as is)

RunOutcomes = dict[str, dict[int, bool]]


@pytest.fixture
def write_run(tmp_path: Path) -> Callable[..., Path]:
    """Build a run folder from {task: {seed: success}} without running any episode."""

    def write(name: str, outcomes: RunOutcomes) -> Path:
        run = tmp_path / name
        for task, seeds in outcomes.items():
            assert task in TASKS
            for seed, success in seeds.items():
                folder = run / task / f"seed_{seed}"
                folder.mkdir(parents=True)
                record = {
                    "task": task,
                    "success": success,
                    "tokens": 1000 + seed,
                    "calls": 3,
                    "seconds": 10.0 + seed,
                    "ended": "returned",
                }
                (folder / "result.json").write_text(json.dumps(record))
        return run

    return write
