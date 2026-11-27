import dataclasses
import json
import os
import sys
import time
import warnings
from pathlib import Path

import pytest

from bench.exitcodes import CliError, ExitCode
from bench.train import SIM_LOG, TrainOptions, exit_code, format_estimate, run_train
from harness.llm_relay import Budget
from harness.tests.conftest import CHAT_AND_RUN, LIFT_REPLY
from harness.tests.fake_env import FakeEnv
from harness.tests.fake_model import FakeModel

SEEDS = {"cube_lifting": [11, 12], "cube_stack": [21]}
NO_SOLVE = "def solve(task, robot, llm):\n    pass\n"


def boom_env(task: str, seed: int, render: bool = False):
    raise RuntimeError("simulator exploded")


def noisy_env(task: str, seed: int, render: bool = False):
    """An env that is as chatty as the real simulator: import-time prints, library warnings, a
    C-level write to stderr."""
    print("LIBERO not installed!")
    print("[robosuite WARNING] No private macro file found!", file=sys.stderr)
    warnings.warn("a library warning", stacklevel=1)
    os.write(2, b"native noise on fd 2\n")
    return FakeEnv(task, seed, render)


def options(tmp_path: Path, agent: Path, url: str, **kwargs) -> TrainOptions:
    defaults = {
        "seeds": SEEDS,
        "out": tmp_path / "runs" / "r1",
        "workers": 2,
        "default_wait_s": 0.01,
        "max_wait_s": 0.1,
        "results_tsv": tmp_path / "results.tsv",
        "api_key": None,
    }
    return TrainOptions(agent_dir=agent, model_url=url, model="m", **{**defaults, **kwargs})


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


NOISE = ("LIBERO not installed!", "[robosuite WARNING]", "native noise on fd 2", "library warning")


def test_quiet_by_default_noise_goes_to_sim_log(make_agent, tmp_path, capfd):
    agent = make_agent(CHAT_AND_RUN)
    with FakeModel([LIFT_REPLY]) as model:
        opts = options(tmp_path, agent, model.url, seeds={"cube_lifting": [1, 2]})
        report = run_train(opts, noisy_env)
    shown = capfd.readouterr()
    assert report["overall"] == 1.0
    assert not any(n in shown.out + shown.err for n in NOISE)
    assert "[1/2] cube_lifting" in shown.err and "[2/2] cube_lifting" in shown.err
    log = (opts.out / SIM_LOG).read_text()
    assert all(n in log for n in NOISE)


def test_verbose_shows_everything_and_writes_no_log(make_agent, tmp_path, capfd):
    agent = make_agent(CHAT_AND_RUN)
    with FakeModel([LIFT_REPLY]) as model:
        opts = options(tmp_path, agent, model.url, seeds={"cube_lifting": [1]}, verbose=True)
        run_train(opts, noisy_env)
    shown = capfd.readouterr()
    assert all(n in shown.out + shown.err for n in NOISE)
    assert not (opts.out / SIM_LOG).exists()


def test_quiet_run_still_shows_why_an_episode_crashed(make_agent, tmp_path, capfd):
    agent = make_agent(CHAT_AND_RUN)
    with FakeModel([LIFT_REPLY]) as model:
        opts = options(tmp_path, agent, model.url, seeds={"cube_lifting": [1]})
        report = run_train(opts, boom_env)
    err = capfd.readouterr().err
    assert report["crashed"] == 1
    assert "[1/1] cube_lifting seed 1: RuntimeError: simulator exploded" in err
    assert f"{opts.out / SIM_LOG}" in err  # where to look for the simulator's own output


def test_start_of_run_line_gives_episodes_workers_and_a_duration_range(make_agent, tmp_path, capfd):
    agent = make_agent(CHAT_AND_RUN)
    with FakeModel([LIFT_REPLY]) as model:
        run_train(options(tmp_path, agent, model.url, seeds={"cube_lifting": [1, 2]}), FakeEnv)
    first = capfd.readouterr().err.splitlines()[0]
    assert first.startswith("planned: 2 episodes at 2 workers, roughly 10 seconds to 5 minutes")


def test_estimate_figures():
    # 35 episodes at 2 workers took about 18 minutes; the 210 of the training split 1.5-2 hours
    assert format_estimate(35, 2).startswith(
        "planned: 35 episodes at 2 workers, roughly 12 to 23 minutes"
    )
    assert "roughly 74 minutes to 2.3 hours" in format_estimate(210, 2)
    assert "1 episode at 1 worker" in format_estimate(1, 1)
    assert "(5 already scored)" in format_estimate(30, 2, done=5)


def allowance_used(body):
    error = {
        "code": "model_allowance_used",
        "message": "daily model allowance used",
        "tokens_used": 5_018_043,
        "tokens_per_day": 5_000_000,
        "resets_at": "2026-11-28T00:00:00Z",
    }
    return 429, {"error": error}, {"Retry-After": "52238"}


@pytest.mark.parametrize("retries", [2, 0])
def test_a_used_up_allowance_stops_the_run_at_once_and_it_resumes_later(
    make_agent, tmp_path, retries
):
    agent = make_agent(CHAT_AND_RUN)
    # preflight ok, episode 1 ok, then the allowance is used
    with FakeModel([LIFT_REPLY, LIFT_REPLY, allowance_used]) as model:
        seeds = {"cube_lifting": [1, 2, 3]}
        opts = options(tmp_path, agent, model.url, seeds=seeds, workers=1, retries=retries)
        with pytest.raises(CliError) as e:
            run_train(opts, FakeEnv)
    assert e.value.code == ExitCode.LIMITED == 8
    assert e.value.message.startswith("1/3 episodes scored in ")
    assert "daily model allowance used: 5,018,043 of 5,000,000 tokens, resets at 00:00 UTC" in (
        e.value.message
    )
    assert len(model.requests) == 4  # episode 2's call and one probe: no waits, no retries
    assert (opts.out / "cube_lifting/seed_2/infra.json").exists()
    assert not (opts.out / "cube_lifting/seed_3").exists()  # not started
    assert json.loads((opts.out / "score.json").read_text())["episodes"] == 1
    with FakeModel([LIFT_REPLY]) as healthy:
        report = run_train(dataclasses.replace(opts, model_url=healthy.url), FakeEnv)
    assert report["complete"] and report["episodes"] == 3 and exit_code(report) == ExitCode.OK


def test_a_used_up_allowance_stops_before_any_episode(make_agent, tmp_path):
    agent = make_agent(CHAT_AND_RUN)
    with FakeModel([allowance_used]) as model:
        opts = options(tmp_path, agent, model.url)
        with pytest.raises(CliError) as e:
            run_train(opts, FakeEnv)
    assert e.value.code == ExitCode.LIMITED and len(model.requests) == 1
    assert "resets at 00:00 UTC" in e.value.message and not (opts.out / "cube_lifting").exists()


def test_two_runs_past_the_concurrent_limit_wait_instead_of_failing(make_agent, tmp_path):
    error = {"code": "too_many_concurrent", "message": "at most 4 model requests at a time"}
    busy = lambda body: (429, {"error": error}, {"Retry-After": "1"})  # noqa: E731
    agent = make_agent(CHAT_AND_RUN)
    with FakeModel([LIFT_REPLY, busy, LIFT_REPLY]) as model:
        report = run_train(
            options(tmp_path, agent, model.url, seeds={"cube_lifting": [1]}, workers=1), FakeEnv
        )
    assert report["overall"] == 1.0 and report["infra_failed"] == 0 and len(model.requests) == 3
    llm = (tmp_path / "runs/r1/cube_lifting/seed_1/llm.jsonl").read_text()
    assert '"rate_limited_s": 1.0' in llm


@pytest.fixture
def gateway():
    from bench.tests.fake_gateway import FakeGateway

    with FakeGateway() as gw:
        gw.state.keys["ink_" + "k" * 43] = "alice"
        yield gw


def via_gateway(tmp_path, agent, gw, **kwargs):
    return options(
        tmp_path,
        agent,
        gw.url + "/v1",
        api_key="ink_" + "k" * 43,
        gateway_url=gw.url,
        seeds={"cube_lifting": [1]},
        workers=1,
        **kwargs,
    )


def test_train_shows_the_allowance_and_warns_when_the_plan_may_not_fit(
    make_agent, tmp_path, gateway, capsys
):
    gateway.state.tokens_per_day, gateway.state.tokens_used = 50_000, 1_000
    run_train(via_gateway(tmp_path, make_agent(CHAT_AND_RUN), gateway), FakeEnv)
    err = capsys.readouterr().err
    assert "model tokens today: 1,000 of 50,000 used, resets at " in err
    assert "this run needs at most 66,000 tokens (1 x the 66,000-token episode budget" in err
    assert "warning: that may be more than the 49,000 tokens left today (about 0 episodes)" in err


def test_train_estimates_from_this_agents_last_run(make_agent, tmp_path, gateway, capsys):
    from bench.train import agent_fingerprint

    agent = make_agent(CHAT_AND_RUN)
    sha = agent_fingerprint(agent)
    (tmp_path / "results.tsv").write_text(
        "run_id\tagent_sha\ttokens_mean\nold\tffffffffffff\t9\nr0\t" + sha + "\t26500.0\n"
    )
    gateway.state.old_gateway = True  # only tokens_today and tokens_per_day: still read
    run_train(via_gateway(tmp_path, agent, gateway), FakeEnv)
    err = capsys.readouterr().err
    assert "model tokens today: 0 of 30,000,000 used" in err
    assert "this run needs about 26,500 tokens (1 x 26,500, the mean of r0)" in err
    assert "warning" not in err


def test_train_with_the_allowance_used_stops_before_asking_the_model(make_agent, tmp_path, gateway):
    gateway.state.tokens_used = gateway.state.tokens_per_day
    with pytest.raises(CliError) as e:
        run_train(via_gateway(tmp_path, make_agent(CHAT_AND_RUN), gateway), FakeEnv)
    assert e.value.code == ExitCode.LIMITED and gateway.state.chat_requests == 0
    assert "daily model allowance used: 30,000,000 of 30,000,000 tokens" in e.value.message
