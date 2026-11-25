import json

import pytest

from bench import cli, doctor, train
from bench.cli import main
from bench.exitcodes import MEANING, ExitCode
from harness.llm_relay import Budget
from harness.tasks import TASKS
from harness.tests.conftest import CHAT_AND_RUN, LIFT_REPLY
from harness.tests.fake_env import FakeEnv
from harness.tests.fake_model import FakeModel


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("INKLING_CONFIG", str(tmp_path / "none.json"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    for name in (
        "INKLING_MODEL_URL",
        "INKLING_GATEWAY_HOST",
        "INKLING_GATEWAY_URL",
        "INKLING_MODEL",
        "INKLING_KEY_PATH",
    ):
        monkeypatch.delenv(name, raising=False)
    agent = tmp_path / "agent"
    agent.mkdir()
    (agent / "agent.py").write_text(CHAT_AND_RUN)
    (agent / "NOTES.md").write_text("notes\n")
    return tmp_path


def run_cli(capsys, *argv: str) -> tuple[int, str, str]:
    code = main(list(argv))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_exit_code_table_is_complete_stable_and_in_help(capsys):
    assert {c.name: int(c) for c in ExitCode} == {
        "OK": 0, "FAILED": 1, "USAGE": 2, "NOT_READY": 3, "INFRA": 4,
        "CRASHED": 5, "NOT_FOUND": 6, "NOT_AVAILABLE": 7, "LIMITED": 8, "INTERRUPTED": 130,
    }  # fmt: skip
    assert set(MEANING) == set(ExitCode)
    with pytest.raises(SystemExit) as e:
        main(["--help"])
    assert e.value.code == 0
    out = capsys.readouterr().out
    assert "exit codes:" in out
    assert all(f"{int(c):>3}  {c.name.lower()}" in out for c in ExitCode)


@pytest.mark.parametrize(
    "argv", [["login"], ["submit"], ["status"], ["pick", "a", "b"], ["train", "--remote"]]
)
def test_server_side_commands_need_a_configured_gateway(capsys, workdir, argv):
    code, out, err = run_cli(capsys, *argv, "--json")
    assert code == ExitCode.NOT_AVAILABLE == 7
    assert "no gateway configured" in err
    assert json.loads(out)["error"]["name"] == "not_available"


def test_usage_errors_go_to_stderr_and_are_json_with_json(capsys):
    code, out, err = run_cli(capsys, "train", "--tasks", "nope", "--json")
    assert code == ExitCode.USAGE and out and "unknown tasks" in err
    assert json.loads(out)["error"]["name"] == "usage"
    code, out, err = run_cli(capsys, "frobnicate")
    assert code == ExitCode.USAGE and out == "" and "invalid choice" in err
    assert run_cli(capsys, "train", "--seeds", "0")[0] == ExitCode.USAGE
    assert run_cli(capsys, "train", "--workers", "0")[0] == ExitCode.USAGE
    assert run_cli(capsys, "view")[0] == ExitCode.USAGE


def test_train_needs_a_model_url_an_agent_and_a_simulator(capsys, workdir, monkeypatch):
    code, _, err = run_cli(capsys, "train")
    assert code == ExitCode.NOT_READY and "no model URL" in err
    url = ["--model-url", "http://127.0.0.1:9/v1"]
    (workdir / "agent" / "agent.py").unlink()
    code, _, err = run_cli(capsys, "train", *url)
    assert code == ExitCode.NOT_READY and "agent.py" in err and "git checkout" in err
    (workdir / "agent" / "agent.py").write_text(CHAT_AND_RUN)
    monkeypatch.setattr(doctor, "is_native_windows", lambda: False)
    monkeypatch.setattr(doctor.importlib.util, "find_spec", lambda name: None)
    code, _, err = run_cli(capsys, "train", *url)
    assert code == ExitCode.NOT_READY and "not importable" in err and "uv sync" in err


def test_native_windows_defaults_to_remote(capsys, workdir, monkeypatch):
    monkeypatch.setattr(doctor, "is_native_windows", lambda: True)
    code, _, err = run_cli(capsys, "train")
    assert "native Windows: simulating on the shared box (--remote)" in err
    assert code == ExitCode.NOT_AVAILABLE and "no gateway configured" in err


def test_train_end_to_end_through_the_cli(capsys, workdir, monkeypatch):
    monkeypatch.setattr(doctor, "check_simulator", lambda: doctor.Check("simulator", doctor.OK, ""))
    real = train.run_train
    monkeypatch.setattr(cli.train, "run_train", lambda opts: real(opts, make_env=FakeEnv))
    with FakeModel([LIFT_REPLY]) as model:
        argv = ["train", "--tasks", "cube_lifting,cube_stack", "--seeds", "2", "--workers", "1"]
        argv += ["--model-url", model.url, "--out", "runs/mine", "--json"]
        code, out, _ = run_cli(capsys, *argv)
        assert code == 0
        report = json.loads(out)
        assert report["overall"] == 1.0 and report["episodes"] == 4 and report["complete"]
        assert report["per_task"].keys() == {"cube_lifting", "cube_stack"}
        assert (workdir / "runs/mine/score.json").exists() and (workdir / "results.tsv").exists()
        before = len(model.requests)
        code, out, _ = run_cli(capsys, *argv[:-1])  # resume as text: nothing left to run
        assert code == 0 and "4/4 episodes scored" in out and len(model.requests) == before


def test_train_in_containers_checks_docker_first_and_passes_the_sandbox_on(
    capsys, workdir, monkeypatch
):
    monkeypatch.setattr(doctor, "check_simulator", lambda: doctor.Check("simulator", doctor.OK, ""))
    url = ["--model-url", "http://127.0.0.1:9/v1"]

    def no_runtime(runtime, image):
        raise RuntimeError(f"Docker has no runtime {runtime!r}")

    monkeypatch.setattr(cli, "check_docker", no_runtime)
    code, _, err = run_cli(capsys, "train", "--sandbox", "container", *url)
    assert (
        code == ExitCode.NOT_READY and "--sandbox container: Docker has no runtime 'runsc'" in err
    )

    seen = []
    monkeypatch.setattr(cli, "check_docker", lambda runtime, image: seen.append(runtime))
    monkeypatch.setattr(cli.train, "run_train", lambda opts: seen.append(opts) or 1 / 0)
    with pytest.raises(ZeroDivisionError):
        main(["train", "--sandbox", "container", "--runtime", "runc", *url])
    assert seen[0] == "runc" and (seen[1].sandbox, seen[1].runtime) == ("container", "runc")


def test_train_seed_list_and_budget_options(capsys, workdir, monkeypatch):
    monkeypatch.setattr(doctor, "check_simulator", lambda: doctor.Check("simulator", doctor.OK, ""))
    seen = []
    monkeypatch.setattr(cli.train, "run_train", lambda opts: seen.append(opts) or 1 / 0)
    url = ["--model-url", "http://127.0.0.1:9/v1"]
    with pytest.raises(ZeroDivisionError):
        main(["train", "--tasks", "cube_stack,cube_lifting", "--seed-list", "1-3,7,2", *url])
    assert seen[-1].seeds == {"cube_lifting": [1, 2, 3, 7], "cube_stack": [1, 2, 3, 7]}
    assert seen[-1].budget == Budget()  # the harness's default, from one place
    with pytest.raises(ZeroDivisionError):
        argv = ["--budget-tokens", "50000", "--budget-wall-s", "840", "--budget-calls", "5"]
        main(["train", "--seed-list", "5", *argv, *url])
    assert seen[-1].budget == Budget(calls=5, tokens=50_000, wall_s=840.0)
    assert set(seen[-1].seeds) == set(TASKS) and seen[-1].seeds["cube_stack"] == [5]
    for bad in (["--seed-list", "3-1"], ["--seed-list", "x"], ["--seed-list", "1", "--seeds", "2"],
                ["--budget-tokens", "0"]):  # fmt: skip
        assert run_cli(capsys, "train", *bad, *url)[0] == ExitCode.USAGE
