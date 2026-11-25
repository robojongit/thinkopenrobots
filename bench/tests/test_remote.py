"""`bench login`, `submit`, `status`, `pick` and `train --remote` against a gateway.

Every test runs against `fake_gateway.FakeGateway`. When ``INKLING_OPS_DIR`` points at a
checkout of the private ops repo (with ``uv sync`` done there), the tests marked ``both`` also
run against the real server, started as ``python -m gateway dev`` (real app and queue, fakes for
GitHub, the GPU box and vLLM).
"""

from __future__ import annotations

import json
import os
import re
import stat
import subprocess
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from bench import doctor, infra, remote
from bench.archive import ArchiveProblem, build, collect
from bench.cli import main
from bench.config import load_config
from bench.exitcodes import ExitCode
from bench.gateway import Gateway, GatewayError
from bench.tests.fake_gateway import FakeGateway
from bench.tests.scripted_server import ScriptedServer
from harness.tests.conftest import CHAT_AND_RUN

REPO = Path(__file__).resolve().parents[2]
OPS = os.environ.get("INKLING_OPS_DIR")
LABEL = "Claude Code · Fable 5"
LABEL_LINE = f"Coding agent:  {LABEL} "


class RealGateway:
    """The ops repo's server in dev mode, as a subprocess."""

    def __init__(self, data_dir: Path) -> None:
        env = {k: v for k, v in os.environ.items() if k not in ("VIRTUAL_ENV", "PYTHONPATH")}
        env["INKLING_TRAIN_SPLIT"] = str(REPO / "splits" / "train.json")
        cmd = [
            "uv",
            "run",
            "--directory",
            str(OPS),
            "--frozen",
            "python",
            "-m",
            "gateway",
            "dev",
            "--port",
            "0",
            "--data-dir",
            str(data_dir),
            "--pick-window-open",
        ]
        self.proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env
        )
        self.log: list[str] = []
        deadline = time.time() + 120
        self.url = ""
        while time.time() < deadline:
            line = self.proc.stdout.readline()
            if not line:
                break
            self.log.append(line)
            if m := re.match(r"listening on (http://\S+)", line):
                self.url = m[1]
                break
        if not self.url:
            self.proc.kill()
            raise RuntimeError("ops dev server did not start:\n" + "".join(self.log))
        threading.Thread(target=lambda: self.log.extend(self.proc.stdout), daemon=True).start()

    def approve(self, user_code: str, login: str) -> None:
        body = json.dumps({"user_code": user_code, "login": login}).encode()
        req = urllib.request.Request(
            self.url + "/dev/approve",
            body,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=10).read()

    def score_all(self, key: str) -> None:
        gw = Gateway(self.url, key)
        for _ in range(200):
            subs = gw.submissions()["submissions"]
            if all(s["state"] not in ("queued", "running") for s in subs):
                return
            time.sleep(0.1)
        raise AssertionError("submissions not scored")

    def close(self) -> None:
        self.proc.terminate()
        self.proc.wait(10)


class FakeBackend:
    def __init__(self) -> None:
        self.fake = FakeGateway().__enter__()
        self.url = self.fake.url
        self.fake.state.window_open = True

    def approve(self, user_code: str, login: str) -> None:
        req = urllib.request.Request(
            self.url + "/dev/approve",
            json.dumps({"user_code": user_code, "login": login}).encode(),
            method="POST",
        )
        urllib.request.urlopen(req, timeout=10).read()

    def score_all(self, key: str) -> None:
        self.fake.score_all()

    def close(self) -> None:
        self.fake.__exit__()


BACKENDS = ["fake"] + (["real"] if OPS else [])


@pytest.fixture(params=BACKENDS)
def server(request, tmp_path):
    backend = RealGateway(tmp_path / "ops-data") if request.param == "real" else FakeBackend()
    yield backend
    backend.close()


@pytest.fixture
def fake(tmp_path):
    backend = FakeBackend()
    yield backend
    backend.close()


@pytest.fixture
def home(tmp_path, monkeypatch):
    """A working directory with agent/, a config dir in tmp, and fast polling."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("INKLING_CONFIG", str(tmp_path / "none.json"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    for name in (
        "INKLING_MODEL_URL",
        "INKLING_GATEWAY_HOST",
        "INKLING_GATEWAY_URL",
        "INKLING_KEY_PATH",
        "INKLING_MODEL",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(remote, "POLL_S", 0.1)
    monkeypatch.setattr(remote, "sleep", lambda s: time.sleep(min(s, 0.1)))
    agent = tmp_path / "agent"
    agent.mkdir()
    (agent / "agent.py").write_text(CHAT_AND_RUN)
    (agent / "NOTES.md").write_text(f"# Notes\n\n{LABEL_LINE}\n\nModel: Inkling 2 Mini.\n")
    (agent / "memory").mkdir()
    (agent / "memory" / "README.md").write_text("notes\n")
    (agent / "__pycache__").mkdir()
    (agent / "__pycache__" / "agent.cpython-311.pyc").write_bytes(b"\0junk")
    return tmp_path


def run_cli(capsys, *argv):
    code = main(list(argv))
    out = capsys.readouterr()
    return code, out.out, out.err


def login(capsys, server, monkeypatch, name="alice"):
    monkeypatch.setenv("INKLING_GATEWAY_URL", server.url)
    code, out, err = run_cli(capsys, "login", "--no-wait", "--json")
    assert code == 0, err
    user_code = json.loads(out)["user_code"]
    assert user_code in err and "github.com/login/device" in err
    server.approve(user_code, name)
    code, out, err = run_cli(capsys, "login", "--json")
    assert code == 0, err
    return json.loads(out)


# --- doctor: the login check ---------------------------------------------------------------------
@pytest.fixture
def healthy_setup(monkeypatch):
    """Everything but the login and model checks is fine, so the exit code says about those."""
    monkeypatch.setattr(doctor, "check_simulator", lambda: doctor.Check("simulator", "ok", "fake"))
    monkeypatch.setattr(doctor, "check_uv", lambda: doctor.Check("uv", "ok", "fake"))


def doctor_checks(capsys, *extra):
    code, out, err = run_cli(capsys, "doctor", "--json", *extra)
    data = json.loads(out)
    return code, {c["name"]: c for c in data["checks"]}, out + err


def test_doctor_login_check_logged_in_and_never_prints_the_key(
    capsys, home, server, monkeypatch, healthy_setup
):
    login(capsys, server, monkeypatch)
    key = (home / "config" / "inkling-robo" / "key").read_text().strip()
    code, checks, shown = doctor_checks(capsys)
    assert code == 0
    assert checks["login"]["status"] == "ok" and checks["login"]["detail"] == "logged in as alice"
    assert checks["model_url"]["status"] == "ok" and "reachable" in checks["model_url"]["detail"]
    assert "may be asleep" in checks["model_url"]["detail"]
    assert key not in shown


def test_doctor_login_check_without_a_key_is_a_todo_with_the_command(
    capsys, home, server, monkeypatch
):
    monkeypatch.setenv("INKLING_GATEWAY_URL", server.url)
    code, checks, _ = doctor_checks(capsys)
    login_check = checks["login"]
    assert login_check["status"] == "todo" and login_check["fix"] == "uv run bench login --no-wait"
    assert "login" not in json.loads(capsys.readouterr().out or "{}").get("failed", [])
    code_text, text, _ = run_cli(capsys, "doctor")
    assert "[todo] login: not logged in" in text and "fix: uv run bench login --no-wait" in text


def test_doctor_login_check_with_a_rejected_key_is_a_todo(
    capsys, home, server, monkeypatch, healthy_setup
):
    monkeypatch.setenv("INKLING_GATEWAY_URL", server.url)
    key_path = home / "config" / "inkling-robo" / "key"
    key_path.parent.mkdir(parents=True)
    key_path.write_text("ink_not_a_real_key\n")
    code, checks, shown = doctor_checks(capsys)
    assert code == 0
    assert checks["login"]["status"] == "todo" and "rejected" in checks["login"]["detail"]
    assert checks["login"]["fix"] == "uv run bench login --no-wait"
    assert "ink_not_a_real_key" not in shown


def test_doctor_login_does_not_fail_doctor_with_your_own_model(
    capsys, home, fake, monkeypatch, healthy_setup
):
    monkeypatch.setenv("INKLING_GATEWAY_URL", fake.url)
    with ScriptedServer([(200, {})]) as own:
        argv = ("--model-url", own.url)
        code, checks, _ = doctor_checks(capsys, *argv)
        assert code == 0
        assert checks["login"]["status"] == "skip" and "own model" in checks["login"]["detail"]
        assert "model_url" in checks and checks["model_url"]["status"] == "ok"
        assert "asleep" not in checks["model_url"]["detail"]
        login(capsys, fake, monkeypatch)  # logged in: still reported, model_url unchanged
        code, checks, _ = doctor_checks(capsys, *argv)
        assert code == 0 and checks["login"]["status"] == "ok"
        key_path = home / "config" / "inkling-robo" / "key"
        key_path.write_text("ink_stale\n")
        code, checks, _ = doctor_checks(capsys, *argv)
        assert code == 0
        assert checks["login"]["status"] == "skip" and "rejected" in checks["login"]["detail"]


def test_doctor_login_check_when_the_gateway_is_down_only_warns(
    capsys, home, monkeypatch, healthy_setup
):
    monkeypatch.setenv("INKLING_GATEWAY_URL", "http://127.0.0.1:9")
    key_path = home / "config" / "inkling-robo" / "key"
    key_path.parent.mkdir(parents=True)
    key_path.write_text("ink_whatever\n")
    _, checks, _ = doctor_checks(capsys)  # the gateway being down fails model_url, not login
    assert checks["login"]["status"] == "warn" and "unreachable" in checks["login"]["detail"]


# --- login ---------------------------------------------------------------------------------------
def test_login_saves_the_key_private_and_never_prints_it(capsys, home, server, monkeypatch):
    result = login(capsys, server, monkeypatch)
    assert result["status"] == "ok" and result["account"] == "alice"
    key_path = Path(result["key_path"])
    key = key_path.read_text().strip()
    assert key.startswith("ink_") and key_path == home / "config" / "inkling-robo" / "key"
    if os.name == "posix":
        assert stat.S_IMODE(key_path.stat().st_mode) == 0o600
    assert key not in json.dumps(result)
    assert not (home / "config" / "inkling-robo" / "login.json").exists()
    code, out, err = run_cli(capsys, "status")
    assert code == 0 and "no submissions yet" in out and key not in out + err


def test_login_waits_and_the_pending_code_is_reused(capsys, home, fake, monkeypatch):
    monkeypatch.setenv("INKLING_GATEWAY_URL", fake.url)
    run_cli(capsys, "login", "--no-wait")
    first = re.search(r"code (\d{4}-\d{4})", capsys.readouterr().err or "") or None
    code, out, err = run_cli(capsys, "login", "--no-wait", "--json")
    user_code = json.loads(out)["user_code"]
    assert first is None or first[1] == user_code
    assert len(fake.fake.state.logins) == 1  # the same flow, not a new code
    threading.Timer(0.3, fake.approve, (user_code, "bob")).start()
    code, out, err = run_cli(capsys, "login")
    assert code == 0 and "logged in as bob" in out


def test_commands_need_a_login_first(capsys, home, fake, monkeypatch):
    monkeypatch.setenv("INKLING_GATEWAY_URL", fake.url)
    for argv in (["submit"], ["status"], ["pick"], ["train", "--remote"]):
        code, _, err = run_cli(capsys, *argv)
        assert code == ExitCode.NOT_READY and "bench login" in err, argv


def test_plain_http_only_for_this_machine(capsys, home, monkeypatch):
    monkeypatch.setenv("INKLING_GATEWAY_URL", "http://gateway.example.org")
    code, _, err = run_cli(capsys, "status")
    assert code == ExitCode.NOT_READY and "https" in err


# --- submit, status, pick -----------------------------------------------------------------------
def test_submit_status_and_pick(capsys, home, server, monkeypatch):
    key = Path(login(capsys, server, monkeypatch)["key_path"]).read_text().strip()
    code, out, err = run_cli(capsys, "submit", "--json")
    assert code == 0, err
    sub = json.loads(out)
    assert sub["state"] == "queued" and re.match(r"^s\d{4,}$", sub["id"])
    code, out, err = run_cli(capsys, "submit")
    if code == 0:  # the real server may already have scored the first one
        server.score_all(key)
        code, out, err = run_cli(capsys, "submit")
    assert code == ExitCode.LIMITED == 8 and "queue" in err
    server.score_all(key)
    code, out, _ = run_cli(capsys, "status")
    assert (
        code == 0
        and sub["id"] in out
        and "(95%" in out
        and "submissions in the last 24 hours" in out
    )
    code, out, _ = run_cli(capsys, "status", sub["id"], "--json")
    one = json.loads(out)
    assert one["state"] == "done" and 0 <= one["ci_low"] <= one["score"] <= one["ci_high"] <= 100
    assert not {"per_task", "seeds", "log", "logs"} & set(one)
    code, out, _ = run_cli(capsys, "status", "s9999")
    assert code == ExitCode.NOT_FOUND
    code, out, _ = run_cli(capsys, "pick", sub["id"], "--json")
    assert code == 0 and json.loads(out)["picks"] == [sub["id"]]
    assert json.loads(out)["source"] == "explicit"
    code, out, _ = run_cli(capsys, "pick")
    assert code == 0 and sub["id"] in out
    assert run_cli(capsys, "pick", "a", "b", "c")[0] == ExitCode.USAGE


def test_quota_and_window_errors_map_to_limited(capsys, home, fake, monkeypatch):
    login(capsys, fake, monkeypatch)
    for _ in range(3):
        assert run_cli(capsys, "submit")[0] == 0
        fake.fake.score_all()
    code, _, err = run_cli(capsys, "submit", "--json")
    assert code == ExitCode.LIMITED and "3 submissions" in err
    fake.fake.state.window_open = False
    sid = fake.fake.state.jobs[0]["id"]
    code, _, err = run_cli(capsys, "pick", sid)
    assert code == ExitCode.LIMITED and "not open" in err


def test_submit_checks_agent_before_uploading(capsys, home, fake, monkeypatch):
    login(capsys, fake, monkeypatch)
    agent = home / "agent"
    (agent / "run.sh").write_text("echo hi")
    if os.name == "posix":
        (agent / "link.py").symlink_to("/etc/passwd")
    (agent / "NOTES.md").unlink()
    code, _, err = run_cli(capsys, "submit")
    assert code == ExitCode.NOT_READY
    assert "run.sh: file type not allowed" in err and "NOTES.md is missing" in err
    if os.name == "posix":
        assert "link.py is a symbolic link" in err
    assert fake.fake.state.uploads == []


# --- coding agent ------------------------------------------------------------------------------
def public_leaderboard(url: str) -> dict:
    with urllib.request.urlopen(url + "/v1/leaderboard", timeout=10) as r:
        return json.loads(r.read())


def test_coding_agent_goes_from_notes_to_status_and_leaderboard(capsys, home, server, monkeypatch):
    key = Path(login(capsys, server, monkeypatch, name="carol")["key_path"]).read_text().strip()
    code, out, err = run_cli(capsys, "submit")
    assert code == 0, err
    assert f"coding agent: {LABEL}" in err and f"coding agent: {LABEL}" in out
    sid = re.search(r"submitted (s\d+)", out)[1]
    server.score_all(key)
    code, out, _ = run_cli(capsys, "status", sid, "--json")
    assert code == 0 and json.loads(out)["coding_agent"] == LABEL
    code, out, _ = run_cli(capsys, "status", sid)
    assert f"coding agent: {LABEL}" in out
    code, out, _ = run_cli(capsys, "status", "--json")
    assert [s["coding_agent"] for s in json.loads(out)["submissions"]] == [LABEL]
    for _ in range(100):  # the record commit, then the leaderboard, follow the score
        entries = [e for e in public_leaderboard(server.url)["entries"] if e["id"] == sid]
        if entries:
            break
        time.sleep(0.1)
    assert entries and entries[0]["account"] == "carol" and entries[0]["coding_agent"] == LABEL


HOSTILE_NOTES = [
    ("# Notes\nModel: Inkling 2 Mini.\n", 'no "Coding agent:" line'),
    ("- **Coding agent:** Codex\n", 'no "Coding agent:" line'),
    ("coding agent: Codex\n", 'no "Coding agent:" line'),
    ("Coding agent: Codex\nCoding agent: Cursor\n", "2 lines starting with"),
    ("Coding agent:   \n", "is empty"),
    ("Coding agent: <img src=x onerror=alert(1)>\n", "not allowed"),
    ("Coding agent: Codex & friends\n", "not allowed"),
    ('Coding agent: "Codex"\n', "not allowed"),
    ("Coding agent: " + "x" * 41 + "\n", "41 characters"),
    ("Coding agent: Codex\x1b[31m\n", "not allowed"),
    ("Coding agent: Codex\rCoding agent: x\n", "not allowed"),
    ("Coding agent: Cod\u0435x\n", "not allowed"),  # Cyrillic e
]


@pytest.mark.parametrize("notes, message", HOSTILE_NOTES)
def test_submit_refuses_a_bad_coding_agent_line(capsys, home, fake, monkeypatch, notes, message):
    login(capsys, fake, monkeypatch)
    (home / "agent" / "NOTES.md").write_text(notes, newline="")
    code, _, err = run_cli(capsys, "submit")
    assert code == ExitCode.NOT_READY and message in err
    assert "Coding agent: Claude Code · Fable 5" in err  # the exact line to add
    assert fake.fake.state.uploads == []


@pytest.mark.parametrize("notes, message", HOSTILE_NOTES[:1] + HOSTILE_NOTES[3:6])
def test_server_refuses_what_bench_refuses(capsys, home, server, monkeypatch, notes, message):
    key = Path(login(capsys, server, monkeypatch)["key_path"]).read_text().strip()
    (home / "agent" / "NOTES.md").write_text(notes, newline="")
    with pytest.raises(GatewayError) as e:
        Gateway(server.url, key).submit(build(home / "agent").data)
    assert e.value.code == "invalid_archive" and message in e.value.message
    assert "Coding agent: Claude Code · Fable 5" in e.value.message


def test_remote_training_needs_no_coding_agent_line(capsys, home, server, monkeypatch):
    login(capsys, server, monkeypatch)
    (home / "agent" / "NOTES.md").write_text("Model: Inkling 2 Mini.\n")
    argv = ["train", "--remote", "--tasks", "cube_lifting", "--seeds", "1", "--out", "runs/n"]
    code, out, err = run_cli(capsys, *argv)
    assert code == 0, err


def test_archive_is_deterministic_and_skips_caches(home):
    a, b = build(home / "agent"), build(home / "agent")
    assert a.data == b.data and a.files == 3
    assert set(collect(home / "agent")) == {
        "agent/agent.py",
        "agent/NOTES.md",
        "agent/memory/README.md",
    }
    (home / "agent" / "big.md").write_bytes(b"x" * (5 * 1024 * 1024 + 1))
    with pytest.raises(ArchiveProblem, match="limit is 5242880"):
        collect(home / "agent")


# --- remote training --------------------------------------------------------------------------
def test_train_remote_downloads_and_scores_like_a_local_run(capsys, home, server, monkeypatch):
    login(capsys, server, monkeypatch)
    argv = [
        "train",
        "--remote",
        "--tasks",
        "cube_lifting,cube_stack",
        "--seeds",
        "2",
        "--out",
        "runs/r1",
        "--json",
    ]
    code, out, err = run_cli(capsys, *argv)
    assert code == 0, err
    report = json.loads(out)
    assert report["episodes"] == 4 and report["planned"] == 4 and report["complete"]
    assert set(report["per_task"]) == {"cube_lifting", "cube_stack"}
    run = home / "runs" / "r1"
    assert (run / "score.json").exists() and (home / "results.tsv").exists()
    assert json.loads((run / "remote.json").read_text())["downloaded"] is True
    assert json.loads((run / "run.json").read_text())["where"] == "remote"
    seeds = json.loads((REPO / "splits" / "train.json").read_text())["seeds"]
    first = seeds["cube_lifting"][0]
    assert (run / "cube_lifting" / f"seed_{first}" / "result.json").exists()
    assert not (run / ".incoming").exists() and not (run / ".results.tar.gz").exists()
    code, out, _ = run_cli(capsys, "view", "runs/r1", "--json")
    assert code == 0
    code, out, _ = run_cli(capsys, "compare", "runs/r1", "runs/r1", "--json")
    assert code == 0
    # Resuming a finished run sends nothing new.
    code, out, err = run_cli(capsys, *argv)
    assert code == 0 and "queued" not in err


def test_train_remote_rules(capsys, home, fake, monkeypatch):
    login(capsys, fake, monkeypatch)
    code, _, err = run_cli(capsys, "train", "--remote", "--seeds", "8")
    assert code == ExitCode.USAGE and "at most 50" in err
    code, _, err = run_cli(capsys, "train", "--remote", "--model-url", "http://x/v1")
    assert code == ExitCode.USAGE and "--model-url" in err
    code, _, err = run_cli(capsys, "train", "--remote", "--budget-calls", "3")
    assert code == ExitCode.USAGE
    code, _, err = run_cli(
        capsys, "train", "--remote", "--seed-list", "1-3", "--tasks", "cube_lifting"
    )
    assert code == ExitCode.USAGE and "training seeds only" in err
    assert fake.fake.state.jobs == []


def test_native_windows_trains_remotely(capsys, home, fake, monkeypatch):
    login(capsys, fake, monkeypatch)
    monkeypatch.setattr(doctor, "is_native_windows", lambda: True)
    code, out, err = run_cli(
        capsys, "train", "--tasks", "cube_lifting", "--seeds", "1", "--out", "runs/w"
    )
    assert code == 0, err
    assert "native Windows" in err and "1/1 episodes scored" in out
    (job,) = fake.fake.state.jobs
    assert job["kind"] == "train" and list(job["plan"]) == ["cube_lifting"]


def test_waiting_prints_the_waking_message(capsys, home, fake, monkeypatch):
    login(capsys, fake, monkeypatch)
    code, _, err = run_cli(
        capsys, "train", "--remote", "--tasks", "cube_lifting", "--seeds", "1", "--out", "runs/w2"
    )
    assert code == 0 and "waking the shared model, about 4 minutes" in err


def test_model_requests_wait_while_the_model_wakes(capsys, home, fake, monkeypatch):
    login(capsys, fake, monkeypatch)
    config = load_config()
    fake.fake.state.waking = 2
    url = config.resolve_model_url()
    messages = []
    result = infra.wait_until_ready(
        lambda: infra.probe(url, "inkling-2-mini", config.api_key_for(url)),
        sleep=lambda s: None,
        log=messages.append,
    )
    assert result.ok and messages == ["waking the shared model, about 1 minute"] * 2


# --- the key goes to the gateway only -----------------------------------------------------------
def test_key_is_sent_only_to_the_gateway_origin(capsys, home, fake, monkeypatch):
    login(capsys, fake, monkeypatch)
    config = load_config()
    key = config.read_key()
    assert config.api_key_for(fake.url + "/v1") == key
    assert config.api_key_for(fake.url.replace("127.0.0.1", "localhost") + "/v1") is None
    port = int(fake.url.rsplit(":", 1)[1])
    assert config.api_key_for(f"http://127.0.0.1:{port + 1}/v1") is None
    assert config.api_key_for("https://api.openai.com/v1") is None
    assert config.api_key_for("http://localhost:11434/v1") is None


def test_redirects_are_not_followed_with_the_key(home):
    seen = []

    class Capture(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            seen.append(self.headers.get("Authorization"))
            self.send_response(200)
            self.end_headers()

        def log_message(self, *a):
            pass

    capture = ThreadingHTTPServer(("127.0.0.1", 0), Capture)
    target = f"http://127.0.0.1:{capture.server_address[1]}/steal"

    class Redirect(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            self.send_response(302)
            self.send_header("Location", target)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, *a):
            pass

    redirect = ThreadingHTTPServer(("127.0.0.1", 0), Redirect)
    for s in (capture, redirect):
        threading.Thread(target=s.serve_forever, daemon=True).start()
    try:
        gw = Gateway(f"http://127.0.0.1:{redirect.server_address[1]}", "ink_secret")
        with pytest.raises(GatewayError):
            gw.submissions()
        assert seen == []
    finally:
        capture.shutdown()
        redirect.shutdown()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX permissions")
def test_key_file_permissions_are_fixed_on_rewrite(home):
    config = load_config()
    config.key_path.parent.mkdir(parents=True)
    config.key_path.write_text("old")
    config.key_path.chmod(0o644)
    config.save_key("ink_new")
    assert stat.S_IMODE(config.key_path.stat().st_mode) == 0o600
    assert config.read_key() == "ink_new"


def test_results_with_unexpected_paths_are_refused(home, tmp_path):
    import io
    import tarfile

    from bench.exitcodes import CliError

    bad = tmp_path / "bad.tar.gz"
    with tarfile.open(bad, "w:gz") as tar:
        for name in ["cube_lifting/seed_1/result.json", "../evil.py"]:
            info = tarfile.TarInfo(name)
            info.size = 2
            tar.addfile(info, io.BytesIO(b"{}"))
    out = tmp_path / "run"
    out.mkdir()
    with pytest.raises(CliError, match="refused path"):
        remote.extract_results(bad, out, {"cube_lifting": [1]})
    assert not (tmp_path / "evil.py").exists() and not (out / "cube_lifting").exists()
    link = tmp_path / "link.tar.gz"
    with tarfile.open(link, "w:gz") as tar:
        info = tarfile.TarInfo("cube_lifting/seed_1/result.json")
        info.type = tarfile.SYMTYPE
        info.linkname = "/etc/passwd"
        tar.addfile(info)
    with pytest.raises(CliError, match="refused member"):
        remote.extract_results(link, out, {"cube_lifting": [1]})
    other = tmp_path / "other.tar.gz"
    with tarfile.open(other, "w:gz") as tar:
        info = tarfile.TarInfo("cube_lifting/seed_2/result.json")
        info.size = 2
        tar.addfile(info, io.BytesIO(b"{}"))
    with pytest.raises(CliError, match="unexpected path"):
        remote.extract_results(other, out, {"cube_lifting": [1]})


# --- remote videos ------------------------------------------------------------------------------
def test_choose_videos_spreads_across_tasks_in_plan_order():
    seeds = {"cube_lifting": [1, 2, 3], "cube_stack": [4], "spill_wipe": [5, 6]}
    assert remote.choose_videos(seeds, 4) == {
        "cube_lifting": [1, 2],
        "cube_stack": [4],
        "spill_wipe": [5],
    }
    assert remote.choose_videos(seeds, 99) == seeds
    from bench.splits import train_seeds

    full = train_seeds(None, remote.REMOTE_DEFAULT_SEEDS)  # 49 episodes, the default remote run
    chosen = remote.choose_videos(full)
    assert sum(map(len, chosen.values())) == remote.REMOTE_MAX_VIDEOS == 10
    assert list(chosen) == list(full)  # every task once, the first three twice
    assert [len(v) for v in chosen.values()] == [2, 2, 2, 1, 1, 1, 1]
    assert all(v == full[t][: len(v)] for t, v in chosen.items())


def test_train_remote_with_video(capsys, home, server, monkeypatch):
    login(capsys, server, monkeypatch)
    from bench.splits import train_seeds

    seeds = train_seeds(None, 2)  # 14 episodes: 10 get a video
    filmed = remote.choose_videos(seeds)
    argv = ["train", "--remote", "--video", "--seeds", "2", "--out", "runs/v", "--json"]
    code, out, err = run_cli(capsys, *argv)
    assert code == 0, err
    assert f"video for 10 of 14 episodes: {remote.episode_list(filmed)}" in err
    report = json.loads(out)
    run = home / "runs" / "v"
    want = [
        str(Path("runs/v") / t / f"seed_{s}" / "video.mp4") for t, ss in filmed.items() for s in ss
    ]
    assert report["videos"] == want and report["video_notes"] == []
    for t, ss in seeds.items():
        for s in ss:
            assert (run / t / f"seed_{s}" / "video.mp4").is_file() == (s in filmed.get(t, []))
    state = json.loads((run / "remote.json").read_text())
    assert state["video"] == filmed
    code, text, _ = run_cli(capsys, "status", state["id"])
    assert code == 0 and f"videos: 10 ({remote.episode_list(filmed)})" in text
    code, out, _ = run_cli(capsys, "status", state["id"], "--json")
    assert json.loads(out)["video"] == filmed and json.loads(out)["videos"] == 10
    first = next(iter(filmed))
    code, out, _ = run_cli(capsys, "view", "runs/v", "--task", first, "--json")
    traces = json.loads(out)["traces"]
    got = {t["seed"]: t["video"] for t in traces}
    assert got == {
        s: str(Path("runs/v") / first / f"seed_{s}" / "video.mp4") if s in filmed[first] else None
        for s in seeds[first]
    }
    # Plain text: where the files are.
    code, out, err = run_cli(capsys, *argv[:-1])
    assert code == 0 and "queued" not in err and f"videos (10):\n  {want[0]}" in out


def test_train_remote_without_video_asks_for_none(capsys, home, fake, monkeypatch):
    login(capsys, fake, monkeypatch)
    code, out, err = run_cli(capsys, "train", "--remote", "--tasks", "cube_lifting", "--seeds", "1")
    assert code == 0 and "video" not in err + out
    (job,) = fake.fake.state.jobs
    assert job["video"] == {} and job["videos"] == 0


def test_a_server_without_videos_is_reported(capsys, home, fake, monkeypatch):
    login(capsys, fake, monkeypatch)
    from bench.tests import fake_gateway

    monkeypatch.setattr(fake_gateway, "MAX_VIDEOS", 0)
    code, _, err = run_cli(
        capsys, "train", "--remote", "--video", "--tasks", "cube_lifting", "--seeds", "1"
    )
    assert code == ExitCode.NOT_AVAILABLE and "without --video" in err
    assert fake.fake.state.jobs == []


def test_videos_in_the_results_are_checked(home, tmp_path):
    import io
    import tarfile

    from bench.exitcodes import CliError

    def results(name, members):
        path = tmp_path / name
        with tarfile.open(path, "w:gz") as tar:
            for member, size in members:
                info = tarfile.TarInfo(member)
                info.size = size
                tar.addfile(info, io.BytesIO(b"\0" * size))
        return path

    out = tmp_path / "run"
    out.mkdir()
    plan, video = {"cube_lifting": [1, 2]}, {"cube_lifting": [1]}
    ok = results(
        "ok.tar.gz",
        [("cube_lifting/seed_1/video.mp4", 100), ("cube_lifting/seed_2/result.json", 2)],
    )
    assert remote.extract_results(ok, out, plan, video) == 2
    assert (out / "cube_lifting" / "seed_1" / "video.mp4").stat().st_size == 100
    for members in (
        [("cube_lifting/seed_2/video.mp4", 100)],  # not asked for
        [("cube_lifting/seed_1/video.mp4", remote.REMOTE_VIDEO_MAX_BYTES + 1)],  # too large
    ):
        bad = results("bad.tar.gz", members)
        with pytest.raises(CliError, match="unexpected video"):
            remote.extract_results(bad, out, plan, video)
    no_video = results("none.tar.gz", [("cube_lifting/seed_1/video.mp4", 100)])
    with pytest.raises(CliError, match="unexpected video"):
        remote.extract_results(no_video, out, plan)
