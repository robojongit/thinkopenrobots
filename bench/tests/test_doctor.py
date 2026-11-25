import json
import sys

import pytest

from bench import doctor
from bench.cli import main
from bench.doctor import FAIL, OK, SKIP, WARN
from bench.exitcodes import ExitCode
from bench.tests.scripted_server import ScriptedServer


@pytest.fixture
def agent_dir(tmp_path):
    d = tmp_path / "agent"
    d.mkdir()
    (d / "agent.py").write_text("def solve(task, robot, llm):\n    pass\n")
    (d / "NOTES.md").write_text("# Notes\n\nCoding agent: Codex\n")
    return d


def test_agent_ok(agent_dir):
    assert doctor.check_agent(agent_dir).status == OK


def test_agent_problems_each_come_with_a_fix(agent_dir):
    (agent_dir / "NOTES.md").unlink()
    check = doctor.check_agent(agent_dir)
    assert check.status == FAIL and "NOTES.md" in check.detail and "NOTES.md" in check.fix

    (agent_dir / "agent.py").write_text("def other(): pass\n")
    check = doctor.check_agent(agent_dir)
    assert check.status == FAIL and "solve" in check.detail and check.fix

    (agent_dir / "agent.py").write_text("def solve(:\n")
    check = doctor.check_agent(agent_dir)
    assert check.status == FAIL and "syntax error at line 1" in check.detail and check.fix

    (agent_dir / "agent.py").unlink()
    check = doctor.check_agent(agent_dir)
    assert check.status == FAIL and "missing" in check.detail and "git checkout" in check.fix


def test_coding_agent_line(agent_dir):
    check = doctor.check_coding_agent(agent_dir)
    assert check.status == OK and "Codex" in check.detail
    (agent_dir / "NOTES.md").write_text("# Notes\n- **Coding agent:** Codex\n")
    check = doctor.check_coding_agent(agent_dir)
    assert check.status == FAIL and 'no "Coding agent:" line' in check.detail
    assert '"Coding agent: Claude Code · Fable 5"' in check.fix
    (agent_dir / "NOTES.md").write_text("Coding agent: <b>x</b>\n")
    assert doctor.check_coding_agent(agent_dir).status == FAIL
    (agent_dir / "NOTES.md").unlink()
    assert doctor.check_coding_agent(agent_dir).status == SKIP


def test_python_and_uv(monkeypatch):
    assert doctor.check_python().status == OK
    monkeypatch.setattr(sys, "version_info", (3, 9, 1, "final", 0))
    check = doctor.check_python()
    assert check.status == FAIL and "uv python install" in check.fix

    monkeypatch.setattr(doctor.shutil, "which", lambda name: None)
    assert "install.sh" in doctor.check_uv().fix
    monkeypatch.setattr(doctor, "is_native_windows", lambda: True)
    assert "install.ps1" in doctor.check_uv().fix


def test_native_windows_says_local_simulation_is_unsupported(monkeypatch):
    monkeypatch.setattr(doctor, "is_native_windows", lambda: True)
    sim, platform = doctor.check_simulator(), doctor.check_platform()
    assert sim.status == SKIP and "remote is the default" in sim.detail
    assert platform.status == WARN and "unsupported" in platform.detail


def test_simulator_missing_names_what_is_missing(monkeypatch):
    monkeypatch.setattr(doctor, "is_native_windows", lambda: False)
    monkeypatch.setattr(doctor.importlib.util, "find_spec", lambda name: None)
    monkeypatch.setattr(doctor, "capx_root", lambda: doctor.Path("/nonexistent"))
    check = doctor.check_simulator()
    assert check.status == FAIL and "robosuite" in check.detail and "capx" in check.detail
    assert check.fix == doctor.SIM_INSTALL and check.fix.endswith("&& uv sync")
    assert "capx/third_party/robosuite" in check.fix


def test_model_url_checks(monkeypatch):
    with ScriptedServer([(200, {})]) as ok:
        assert doctor.check_model_url(ok.url).status == OK
        assert ok.requests == [("GET", "/v1/models")]
    with ScriptedServer([(404, {})]) as no_models:
        assert doctor.check_model_url(no_models.url).status == OK  # it answered
    with ScriptedServer([(503, {"Retry-After": "240"})]) as waking:
        assert doctor.check_model_url(waking.url).status == WARN
    with ScriptedServer([(500, {})]) as broken:
        assert doctor.check_model_url(broken.url).status == FAIL
    down = doctor.check_model_url("http://127.0.0.1:9/v1")
    assert down.status == FAIL and "unreachable" in down.detail and "ollama serve" in down.fix
    none = doctor.check_model_url(None)
    assert none.status == FAIL and "INKLING_MODEL_URL" in none.fix


def test_model_url_through_the_gateway_says_reachable_and_that_the_model_may_sleep():
    with ScriptedServer([(404, {})]) as gateway:
        check = doctor.check_model_url(gateway.url, via_gateway=True)
    assert check.status == OK and "gateway" in check.detail and "reachable" in check.detail
    assert "may be asleep" in check.detail and "5 minutes" in check.detail
    assert "HTTP 404" not in check.detail
    with ScriptedServer([(404, {})]) as own:  # the person's own model: nothing about sleeping
        assert "asleep" not in doctor.check_model_url(own.url).detail


def test_cli_exit_codes_and_json(agent_dir, monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("INKLING_CONFIG", str(tmp_path / "none.json"))
    monkeypatch.delenv("INKLING_MODEL_URL", raising=False)
    monkeypatch.setattr(doctor, "check_simulator", lambda: doctor.Check("simulator", OK, "fake"))
    with ScriptedServer([(200, {})]) as server:
        argv = ["doctor", "--agent", str(agent_dir), "--model-url", server.url, "--json"]
        assert main(argv) == ExitCode.OK
        data = json.loads(capsys.readouterr().out)
        names = {c["name"] for c in data["checks"]}
        assert data["ok"] and names >= {"python", "uv", "agent", "coding_agent"}
        (agent_dir / "NOTES.md").write_text("# Notes\n")
        assert main(argv) == ExitCode.NOT_READY
        data = json.loads(capsys.readouterr().out)
        assert data["failed"] == ["coding_agent"]
        (agent_dir / "NOTES.md").unlink()
        assert main(argv) == ExitCode.NOT_READY
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["failed"] == ["agent"] and "NOTES.md" in data["checks"][3]["fix"]
        assert "failed checks: agent" in captured.err
    assert main(["doctor", "--agent", str(agent_dir)]) == ExitCode.NOT_READY
    assert "fix:" in capsys.readouterr().out
