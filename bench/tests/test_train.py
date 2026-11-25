import dataclasses
import json
import time
from pathlib import Path

import pytest

from bench.exitcodes import CliError, ExitCode
from bench.train import TrainOptions, exit_code, run_train
from harness.llm_relay import Budget
from harness.tests.conftest import CHAT_AND_RUN, LIFT_REPLY
from harness.tests.fake_env import FakeEnv
from harness.tests.fake_model import FakeModel

SEEDS = {"cube_lifting": [11, 12], "cube_stack": [21]}
NO_SOLVE = "def solve(task, robot, llm):\n    pass\n"


def boom_env(task: str, seed: int, render: bool = False):
    raise RuntimeError("simulator exploded")


def options(tmp_path: Path, agent: Path, url: str, **kwargs) -> TrainOptions:
    defaults = {
        "seeds": SEEDS,
        "out": tmp_path / "runs" / "r1",
        "workers": 2,
        "default_wait_s": 0.01,
        "max_wait_s": 0.1,
        "results_tsv": tmp_path / "results.tsv",
    }
    return TrainOptions(
        agent_dir=agent, model_url=url, model="m", api_key=None, **{**defaults, **kwargs}
    )


def server_error(status: int):
    return lambda body: (status, {"error": "no"})


def test_train_writes_the_run_folder_score_and_results_row(make_agent, tmp_path):
    agent = make_agent(CHAT_AND_RUN)
    with FakeModel([LIFT_REPLY], prompt_tokens=100, completion_tokens=50) as model:
        report = run_train(options(tmp_path, agent, model.url), make_env=FakeEnv)

    out = tmp_path / "runs" / "r1"
    for task, seeds in SEEDS.items():
        for seed in seeds:
            folder = out / task / f"seed_{seed}"
            assert json.loads((folder / "result.json").read_text())["success"] is True
            assert (folder / "code" / "run_000.py").exists()
    assert (out / "run.json").exists()
    score = json.loads((out / "score.json").read_text())
    assert score["overall"] == 1.0 and score["episodes"] == 3 == score["planned"]
    assert score["complete"] and score["infra_failed"] == 0 and score["crashed"] == 0
    assert score["per_task"]["cube_lifting"]["n"] == 2
    assert score["tokens"]["mean"] == 150 and score["ended"] == {"returned": 3}
    assert report["overall"] == 1.0 and exit_code(report) == ExitCode.OK
    header, row = (tmp_path / "results.tsv").read_text().splitlines()
    assert header.split("\t")[:3] == ["run_id", "finished", "overall"]
    assert row.split("\t")[0] == "r1" and row.split("\t")[2] == "1.0"


def test_task_failure_is_a_scored_failure_not_an_infra_problem(make_agent, tmp_path):
    agent = make_agent(NO_SOLVE)
    with FakeModel([LIFT_REPLY]) as model:
        report = run_train(
            options(tmp_path, agent, model.url, seeds={"cube_lifting": [1]}), FakeEnv
        )
    assert report["overall"] == 0.0 and report["episodes"] == 1 and report["infra_failed"] == 0
    assert exit_code(report) == ExitCode.OK


def reasoning_only(body):
    """A reply that spent every completion token on reasoning and has no content."""
    message = {"role": "assistant", "content": None, "reasoning": "hmm " * 50}
    usage = {"prompt_tokens": 1400, "completion_tokens": body["max_tokens"]}
    return 200, {"choices": [{"message": message, "finish_reason": "length"}], "usage": usage}


def test_reply_without_content_is_a_clean_task_failure(tmp_path):
    base = Path(__file__).resolve().parents[2] / "base"
    with FakeModel([LIFT_REPLY, reasoning_only]) as model:  # the first answer is the preflight
        report = run_train(options(tmp_path, base, model.url, seeds={"cube_lifting": [1]}), FakeEnv)
    assert report["episodes"] == 1 and report["overall"] == 0.0
    assert report["infra_failed"] == report["crashed"] == 0 and report["ended"] == {"returned": 1}
    assert len(model.requests) == 2 and exit_code(report) == ExitCode.OK


def test_reply_cut_off_by_the_episode_time_limit_is_a_task_failure_not_infra(tmp_path):
    def slow(body):
        time.sleep(3)
        return reasoning_only(body)

    base = Path(__file__).resolve().parents[2] / "base"
    opts = options(tmp_path, base, "", seeds={"cube_lifting": [1]}, budget=Budget(wall_s=1))
    with FakeModel([LIFT_REPLY, slow]) as model:
        report = run_train(dataclasses.replace(opts, model_url=model.url), FakeEnv)
    assert report["episodes"] == 1 and report["overall"] == 0.0 and report["infra_failed"] == 0
    assert report["ended"] == {"timeout": 1} and len(model.requests) == 2  # no retry


def test_client_error_from_the_model_counts_as_task_failure(make_agent, tmp_path):
    agent = make_agent(CHAT_AND_RUN)
    with FakeModel([LIFT_REPLY, server_error(400)]) as model:  # the first answer is the preflight
        report = run_train(
            options(tmp_path, agent, model.url, seeds={"cube_lifting": [1]}, workers=1), FakeEnv
        )
    assert report["episodes"] == 1 and report["overall"] == 0.0 and report["infra_failed"] == 0


def test_resume_skips_finished_episodes(make_agent, tmp_path):
    agent = make_agent(CHAT_AND_RUN)
    opts = options(tmp_path, agent, "unused")
    with FakeModel([LIFT_REPLY]) as model:
        opts = options(tmp_path, agent, model.url)
        run_train(opts, FakeEnv)
        kept = opts.out / "cube_lifting" / "seed_11"
        (kept / "marker").write_text("untouched")
        redo = opts.out / "cube_stack" / "seed_21"
        (redo / "result.json").unlink()
        before = len(model.requests)
        report = run_train(opts, FakeEnv)
    assert (kept / "marker").exists()  # finished episodes are not rerun, not even cleaned
    assert (redo / "result.json").exists()
    assert len(model.requests) - before == 2  # one preflight probe and the one missing episode
    assert report["episodes"] == 3 and report["complete"]
    assert len((tmp_path / "results.tsv").read_text().splitlines()) == 2  # row replaced, not added


def test_resume_with_a_changed_agent_is_refused(make_agent, tmp_path):
    agent = make_agent(CHAT_AND_RUN)
    with FakeModel([LIFT_REPLY]) as model:
        opts = options(tmp_path, agent, model.url, seeds={"cube_lifting": [1]})
        run_train(opts, FakeEnv)
        (agent / "agent.py").write_text(CHAT_AND_RUN + "\n# changed\n")
        with pytest.raises(CliError) as e:
            run_train(opts, FakeEnv)
    assert e.value.code == ExitCode.USAGE and "new --out" in e.value.message


def test_model_server_hiccup_is_retried_and_not_counted_as_a_failure(make_agent, tmp_path):
    agent = make_agent(CHAT_AND_RUN)
    # answers in order: preflight ok, the episode's call fails with 503, retry probe ok, retry ok
    with FakeModel([LIFT_REPLY, server_error(503), LIFT_REPLY]) as model:
        report = run_train(
            options(tmp_path, agent, model.url, seeds={"cube_lifting": [1]}, workers=1), FakeEnv
        )
    assert report["overall"] == 1.0 and report["infra_failed"] == 0
    assert len(model.requests) == 4
    assert not (tmp_path / "runs/r1/cube_lifting/seed_1/infra.json").exists()


def test_persistent_server_trouble_is_reported_as_infra_and_finished_on_rerun(make_agent, tmp_path):
    agent = make_agent(CHAT_AND_RUN)
    with FakeModel([LIFT_REPLY, server_error(503)]) as model:
        opts = options(
            tmp_path, agent, model.url, seeds={"cube_lifting": [1, 2]}, workers=1, retries=0
        )
        report = run_train(opts, FakeEnv)
        assert report["episodes"] == 0 and report["infra_failed"] == 1 and not report["complete"]
        assert report["overall"] == 0.0  # nothing was scored: the failure did not become a 0
        assert "HTTP 503" in report["infra"][0]["detail"]
        assert exit_code(report) == ExitCode.INFRA
        assert (opts.out / "cube_lifting/seed_1/infra.json").exists()
    with FakeModel([LIFT_REPLY]) as healthy:
        report = run_train(
            options(tmp_path, agent, healthy.url, seeds={"cube_lifting": [1, 2]}, retries=0),
            FakeEnv,
        )
    assert report["complete"] and report["overall"] == 1.0 and report["infra_failed"] == 0
    assert not (opts.out / "cube_lifting/seed_1/infra.json").exists()


def test_unreachable_model_server_stops_before_any_episode(make_agent, tmp_path):
    agent = make_agent(CHAT_AND_RUN)
    opts = options(tmp_path, agent, "http://127.0.0.1:9/v1")
    with pytest.raises(CliError) as e:
        run_train(opts, FakeEnv)
    assert e.value.code == ExitCode.INFRA
    assert not (opts.out / "cube_lifting").exists()


def test_rejected_credentials_are_a_setup_problem_not_infra(make_agent, tmp_path):
    agent = make_agent(CHAT_AND_RUN)
    with FakeModel([server_error(401)]) as model:
        with pytest.raises(CliError) as e:
            run_train(options(tmp_path, agent, model.url), FakeEnv)
    assert e.value.code == ExitCode.NOT_READY


def test_harness_crash_is_reported_apart_from_task_failure(make_agent, tmp_path):
    agent = make_agent(CHAT_AND_RUN)
    with FakeModel([LIFT_REPLY]) as model:
        report = run_train(
            options(tmp_path, agent, model.url, seeds={"cube_lifting": [1]}), boom_env
        )
    assert report["crashed"] == 1 and report["episodes"] == 0 and report["overall"] == 0.0
    assert report["crash"][0]["error"] == "RuntimeError: simulator exploded"
    assert exit_code(report) == ExitCode.CRASHED
    assert (tmp_path / "runs/r1/cube_lifting/seed_1/crash.json").exists()
