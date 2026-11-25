"""Seeds to run: the published splits (``splits/<name>.json`` at the repo root), or an explicit
list."""

from __future__ import annotations

import json
from pathlib import Path

from harness.tasks import TASKS

SPLITS_DIR = Path(__file__).resolve().parent.parent / "splits"


def train_seeds(
    tasks: list[str] | None = None, per_task: int | None = None
) -> dict[str, list[int]]:
    """Training seeds per task (the first ``per_task`` of each), in task-table order."""
    seeds = json.loads((SPLITS_DIR / "train.json").read_text())["seeds"]
    chosen = tasks or list(TASKS)
    return {t: seeds[t][:per_task] for t in TASKS if t in chosen}


def parse_seed_list(text: str) -> list[int]:
    """Seeds from "1-50" or "3,7,10-12" (ranges inclusive), in order, without repeats.
    Raises ValueError on anything else."""
    seeds: list[int] = []
    for part in text.split(","):
        low, dash, high = part.strip().partition("-")
        first, last = int(low), int(high if dash else low)
        if first < 0 or last < first:
            raise ValueError(f"bad seed range {part.strip()!r}")
        seeds.extend(s for s in range(first, last + 1) if s not in seeds)
    return seeds


def listed_seeds(tasks: list[str] | None, seeds: list[int]) -> dict[str, list[int]]:
    """The same explicit seeds for every chosen task, in task-table order."""
    chosen = tasks or list(TASKS)
    return {t: list(seeds) for t in TASKS if t in chosen}
