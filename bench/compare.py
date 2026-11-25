"""`bench compare`: paired comparison of two runs on the (task, seed) pairs they share."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .exitcodes import CliError, ExitCode
from .runs import scored_results

BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 0
MIN_PAIRS = 50


def outcomes(run: Path) -> dict[tuple[str, int], bool]:
    return {(r["task"], r["seed"]): bool(r["success"]) for r in scored_results(run)}


def paired_bootstrap(diffs: dict[str, np.ndarray]) -> tuple[float, float]:
    """95% interval for the mean over tasks of the mean per-task difference.

    Pairs are resampled with replacement within each task, so every replicate keeps the task mix
    the overall score is defined on. The RNG seed is fixed: the same data gives the same interval.
    """
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    replicates = np.zeros(BOOTSTRAP_RESAMPLES)
    for d in diffs.values():
        replicates += d[rng.integers(0, len(d), (BOOTSTRAP_RESAMPLES, len(d)))].mean(axis=1)
    low, high = np.percentile(replicates / len(diffs), [2.5, 97.5])
    return float(low), float(high)


def verdict(low: float, high: float) -> str:
    return "real gain" if low > 0 else "real loss" if high < 0 else "no clear difference"


def compare_runs(run_a: Path, run_b: Path) -> dict[str, Any]:
    a, b = outcomes(run_a), outcomes(run_b)
    shared = sorted(a.keys() & b.keys())
    if not shared:
        raise CliError(ExitCode.NOT_FOUND, "the runs share no scored (task, seed) pairs")

    per_task: dict[str, dict[str, Any]] = {}
    diffs: dict[str, np.ndarray] = {}
    for task in dict.fromkeys(t for t, _ in shared):
        pairs = [(a[t, s], b[t, s]) for t, s in shared if t == task]
        diffs[task] = np.array([int(y) - int(x) for x, y in pairs], dtype=float)
        rate_a = sum(x for x, _ in pairs) / len(pairs)
        rate_b = sum(y for _, y in pairs) / len(pairs)
        per_task[task] = {
            "n": len(pairs),
            "a": round(rate_a, 4),
            "b": round(rate_b, 4),
            "delta": round(rate_b - rate_a, 4),
            "gained": sum(y and not x for x, y in pairs),
            "lost": sum(x and not y for x, y in pairs),
        }
    low, high = paired_bootstrap(diffs)
    mean_a = float(np.mean([t["a"] for t in per_task.values()]))
    mean_b = float(np.mean([t["b"] for t in per_task.values()]))

    warnings = []
    if len(shared) < MIN_PAIRS:
        warnings.append(
            f"only {len(shared)} shared (task, seed) pairs (under {MIN_PAIRS}); "
            "the interval is wide and a verdict is weak evidence"
        )
    notes = []
    unpaired = len(a) + len(b) - 2 * len(shared)
    if unpaired:
        notes.append(
            f"compared on the {len(shared)} (task, seed) pairs both runs share; "
            f"{unpaired} episodes exist in only one run and were not used"
        )
    return {
        "a": str(run_a),
        "b": str(run_b),
        "pairs": len(shared),
        "overall": {
            "a": round(mean_a, 4),
            "b": round(mean_b, 4),
            "delta": round(mean_b - mean_a, 4),
            "ci95": [round(low, 4), round(high, 4)],
        },
        "flips": {
            "gained": sum(t["gained"] for t in per_task.values()),
            "lost": sum(t["lost"] for t in per_task.values()),
        },
        "per_task": per_task,
        "verdict": verdict(low, high),
        "warnings": warnings,
        "notes": notes,
    }


def format_comparison(c: dict[str, Any]) -> str:
    o = c["overall"]
    lines = [f"compare {c['a']} (a) -> {c['b']} (b): {c['pairs']} shared (task, seed) pairs"]
    lines += [f"note: {n}" for n in c["notes"]]
    lines += [f"WARNING: {w}" for w in c["warnings"]]
    lines.append(f"{'task':<18}{'n':>4}{'a':>8}{'b':>8}{'delta':>8}{'gained':>8}{'lost':>6}")
    for task, t in c["per_task"].items():
        lines.append(
            f"{task:<18}{t['n']:>4}{t['a']:>8.3f}{t['b']:>8.3f}{t['delta']:>+8.3f}"
            f"{t['gained']:>8}{t['lost']:>6}"
        )
    lines.append(
        f"overall: a {o['a']:.3f}, b {o['b']:.3f}, delta {o['delta']:+.3f} "
        f"(95% interval {o['ci95'][0]:+.3f} to {o['ci95'][1]:+.3f}); "
        f"flips: {c['flips']['gained']} fail->success, {c['flips']['lost']} success->fail"
    )
    lines.append(f"verdict: {c['verdict']}")
    return "\n".join(lines)
