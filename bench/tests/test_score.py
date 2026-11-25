import pytest

from bench.score import compute_score, summarize, wilson


def rows(task: str, wins: int, n: int, ended: str = "returned") -> list[dict]:
    return [
        {"task": task, "success": i < wins, "tokens": 100 * (i + 1), "seconds": 1.5, "ended": ended}
        for i in range(n)
    ]


def test_overall_is_the_mean_of_per_task_rates_not_of_episodes():
    score = compute_score(rows("cube_lifting", 10, 10) + rows("cube_stack", 0, 30))
    assert score["overall"] == 0.5  # pooled episodes would give 0.25
    assert score["per_task"]["cube_lifting"]["rate"] == 1.0
    assert score["per_task"]["cube_stack"]["successes"] == 0
    assert score["episodes"] == 40


def test_interval_brackets_the_score_and_shrinks_with_more_episodes():
    small = compute_score(rows("cube_lifting", 5, 10) + rows("cube_stack", 5, 10))
    large = compute_score(rows("cube_lifting", 50, 100) + rows("cube_stack", 50, 100))
    for score in (small, large):
        low, high = score["ci95"]
        assert low < score["overall"] == 0.5 < high
    assert large["ci95"][1] - large["ci95"][0] < small["ci95"][1] - small["ci95"][0]


def test_all_wins_still_has_uncertainty_but_stays_in_range():
    score = compute_score(rows("cube_lifting", 5, 5))
    low, high = score["ci95"]
    assert 0.5 < low < 1.0 and high == 1.0
    assert 0.0 <= score["per_task"]["cube_lifting"]["ci95"][0] < 1.0


def test_wilson_known_value():
    low, high = wilson(5, 10)
    assert low == pytest.approx(0.2366, abs=1e-3) and high == pytest.approx(0.7634, abs=1e-3)
    assert wilson(0, 0) == (0.0, 1.0)


def test_summaries_and_ended_counts():
    score = compute_score(rows("cube_lifting", 2, 4) + rows("cube_stack", 1, 2, ended="timeout"))
    assert score["ended"] == {"returned": 4, "timeout": 2}
    assert score["tokens"] == summarize([100, 200, 300, 400, 100, 200])
    assert score["tokens"]["max"] == 400 and score["seconds"]["mean"] == 1.5
    assert summarize([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])["p90"] == 9


def test_empty_score_is_defined():
    score = compute_score([])
    assert score["overall"] == 0.0 and score["episodes"] == 0 and score["per_task"] == {}
