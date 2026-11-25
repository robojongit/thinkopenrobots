import json

import pytest

from bench.cli import main
from bench.compare import compare_runs, format_comparison
from bench.exitcodes import CliError, ExitCode

TASKS = ["cube_lifting", "cube_stack", "nut_assembly"]


def outcomes(rule) -> dict:
    return {
        task: {seed: rule(task_i, seed) for seed in range(40)} for task_i, task in enumerate(TASKS)
    }


def test_clear_gain(write_run):
    a = write_run("a", outcomes(lambda t, s: s < 12))  # 30% everywhere
    b = write_run("b", outcomes(lambda t, s: s < 28))  # same wins plus 16 more each: 70%
    c = compare_runs(a, b)
    assert c["pairs"] == 120 and c["verdict"] == "real gain"
    assert c["overall"]["delta"] == pytest.approx(0.4)
    low, high = c["overall"]["ci95"]
    assert 0 < low < 0.4 < high
    assert c["flips"] == {"gained": 48, "lost": 0}
    assert c["per_task"]["cube_stack"] == {
        "n": 40, "a": 0.3, "b": 0.7, "delta": 0.4, "gained": 16, "lost": 0,
    }  # fmt: skip
    assert c["warnings"] == []


def test_clear_loss_is_the_mirror_image(write_run):
    a = write_run("a", outcomes(lambda t, s: s < 28))
    b = write_run("b", outcomes(lambda t, s: s < 12))
    c = compare_runs(a, b)
    assert c["verdict"] == "real loss" and c["overall"]["ci95"][1] < 0
    assert c["flips"] == {"gained": 0, "lost": 48}


def test_no_difference_with_balanced_flips(write_run):
    # b wins the odd seeds a loses and loses the even seeds a wins: same score, many flips.
    a = write_run("a", outcomes(lambda t, s: s % 2 == 0))
    b = write_run("b", outcomes(lambda t, s: s % 2 == 1))
    c = compare_runs(a, b)
    assert c["verdict"] == "no clear difference"
    assert c["overall"]["delta"] == 0
    assert c["overall"]["ci95"][0] < 0 < c["overall"]["ci95"][1]
    assert c["flips"] == {"gained": 60, "lost": 60}


def test_identical_runs_have_a_zero_width_interval(write_run):
    a = write_run("a", outcomes(lambda t, s: s < 20))
    c = compare_runs(a, write_run("b", outcomes(lambda t, s: s < 20)))
    assert c["overall"]["ci95"] == [0.0, 0.0] and c["verdict"] == "no clear difference"


def test_is_reproducible(write_run):
    a = write_run("a", outcomes(lambda t, s: (s * 7 + t) % 3 == 0))
    b = write_run("b", outcomes(lambda t, s: (s * 5 + t) % 3 == 0))
    assert compare_runs(a, b) == compare_runs(a, b)


def test_only_shared_pairs_count_and_few_pairs_warn(write_run):
    a = write_run("a", {"cube_lifting": {1: True, 2: False, 3: True}})
    b = write_run("b", {"cube_lifting": {2: True, 3: True, 4: True}, "cube_stack": {1: True}})
    c = compare_runs(a, b)
    assert c["pairs"] == 2 and list(c["per_task"]) == ["cube_lifting"]
    assert any("only 2 shared" in w for w in c["warnings"])
    assert any("no partner" in w for w in c["warnings"])
    assert "WARNING" in format_comparison(c)


def test_no_shared_pairs_is_not_found(write_run):
    a = write_run("a", {"cube_lifting": {1: True}})
    b = write_run("b", {"cube_lifting": {2: True}})
    with pytest.raises(CliError) as e:
        compare_runs(a, b)
    assert e.value.code == ExitCode.NOT_FOUND


def test_cli_text_and_json(write_run, capsys):
    a = write_run("a", outcomes(lambda t, s: s < 12))
    b = write_run("b", outcomes(lambda t, s: s < 28))
    assert main(["compare", str(a), str(b)]) == 0
    out = capsys.readouterr().out
    assert "verdict: real gain" in out and "cube_stack" in out
    assert main(["compare", str(a), str(b), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["verdict"] == "real gain"
    assert main(["compare", str(a), str(a) + "-missing"]) == ExitCode.NOT_FOUND
