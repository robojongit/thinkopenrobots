"""Score maths: overall = mean of per-task success rates, with 95% intervals."""

from __future__ import annotations

import math
import statistics
from typing import Any

Z95 = 1.96


def wilson(successes: int, n: int) -> tuple[float, float]:
    """95% Wilson score interval for a success rate."""
    if n == 0:
        return 0.0, 1.0
    p = successes / n
    denom = 1 + Z95**2 / n
    centre = (p + Z95**2 / (2 * n)) / denom
    half = Z95 * math.sqrt(p * (1 - p) / n + Z95**2 / (4 * n**2)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def overall_interval(per_task: dict[str, dict[str, Any]]) -> tuple[float, float]:
    """95% interval for the mean of per-task rates (normal approximation, tasks independent).

    The variance of each rate uses the Agresti-Coull adjusted proportion, so a task at 0% or
    100% does not claim zero uncertainty.
    """
    if not per_task:
        return 0.0, 1.0
    mean = statistics.fmean(t["rate"] for t in per_task.values())
    var = 0.0
    for t in per_task.values():
        p = (t["successes"] + 2) / (t["n"] + 4)
        var += p * (1 - p) / (t["n"] + 4)
    half = Z95 * math.sqrt(var) / len(per_task)
    return max(0.0, mean - half), min(1.0, mean + half)


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "p90": 0.0, "max": 0.0}
    ordered = sorted(values)
    p90 = ordered[min(len(ordered) - 1, math.ceil(0.9 * len(ordered)) - 1)]
    return {
        "mean": round(statistics.fmean(values), 3),
        "median": round(statistics.median(values), 3),
        "p90": round(p90, 3),
        "max": round(ordered[-1], 3),
    }


def compute_score(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Score from result records (``task``, ``success``, ``tokens``, ``seconds``, ``ended``).

    Tasks with no finished episode are left out of the overall mean.
    """
    per_task: dict[str, dict[str, Any]] = {}
    for task in dict.fromkeys(r["task"] for r in results):
        rows = [r for r in results if r["task"] == task]
        wins = sum(bool(r["success"]) for r in rows)
        low, high = wilson(wins, len(rows))
        per_task[task] = {
            "rate": round(wins / len(rows), 4),
            "successes": wins,
            "n": len(rows),
            "ci95": [round(low, 4), round(high, 4)],
        }
    overall = statistics.fmean(t["rate"] for t in per_task.values()) if per_task else 0.0
    low, high = overall_interval(per_task)
    ended: dict[str, int] = {}
    for r in results:
        ended[r["ended"]] = ended.get(r["ended"], 0) + 1
    return {
        "overall": round(overall, 4),
        "ci95": [round(low, 4), round(high, 4)],
        "per_task": per_task,
        "episodes": len(results),
        "tokens": summarize([r["tokens"] for r in results]),
        "seconds": summarize([r["seconds"] for r in results]),
        "ended": ended,
    }
