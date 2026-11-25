"""The run folder: ``runs/<id>/<task>/seed_<seed>/`` holds one episode as `run_episode` writes it.

The seed is part of the path because ``result.json`` leaves it out. Two marker files record
episodes that did not produce a score: ``infra.json`` (model server trouble) and ``crash.json``
(the harness raised). An episode is finished when it has ``result.json`` and no ``infra.json``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from harness.tasks import TASKS

SCORED, INFRA, CRASH, INCOMPLETE = "scored", "infra", "crash", "incomplete"


def episode_dir(run: Path, task: str, seed: int) -> Path:
    return run / task / f"seed_{seed}"


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


@dataclass(frozen=True)
class Episode:
    task: str
    seed: int
    path: Path

    @property
    def result(self) -> dict[str, Any] | None:
        return read_json(self.path / "result.json")

    @property
    def status(self) -> str:
        if (self.path / "infra.json").exists():
            return INFRA
        if (self.path / "crash.json").exists():
            return CRASH
        return SCORED if (self.path / "result.json").exists() else INCOMPLETE

    @property
    def success(self) -> bool:
        return bool((self.result or {}).get("success"))


def list_episodes(run: Path) -> list[Episode]:
    """Every episode folder in the run, in task-table then seed order."""
    found = []
    for task in TASKS:
        for path in (run / task).glob("seed_*") if (run / task).is_dir() else []:
            suffix = path.name.removeprefix("seed_")
            if path.is_dir() and suffix.lstrip("-").isdigit():
                found.append(Episode(task, int(suffix), path))
    return sorted(found, key=lambda e: (list(TASKS).index(e.task), e.seed))


def scored_results(run: Path) -> list[dict[str, Any]]:
    """Result records of the finished episodes, each with its ``seed`` filled in."""
    return [{**e.result, "seed": e.seed} for e in list_episodes(run) if e.status == SCORED]
