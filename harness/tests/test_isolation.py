"""Red-team agents and code runners against ``ContainerSandbox``: every attack must fail.

Marked ``container`` (``uv run pytest -m container``) and skipped without Docker or the agent
image (``docker/agent/build.sh``). Every test runs once per runtime Docker has, of ``runc`` and
``runsc``, so the same suite checks gVisor wherever it is registered. The fork-bomb and memory
tests use tiny limits so that they cannot hurt the machine running them.

The attacks that do not depend on the sandbox (malformed RPCs in depth, log bounds, motion
argument checks) are in ``test_attacks.py``, in plain-process mode.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import textwrap
import threading
import time
from pathlib import Path

import pytest

from harness.episode import run_episode
from harness.llm_relay import Budget
from harness.rpc import listen_unix
from harness.sandbox import AGENT_OUTPUT_CAP, ROOT, ContainerSandbox, agent_image, check_docker
from harness.tests.conftest import CHAT_AND_RUN, Episode
from harness.tests.fake_env import FakeEnv
from harness.tests.fake_model import FakeModel

pytestmark = pytest.mark.container

IMAGE = agent_image()
SMALL = {"memory": "256m", "pids_limit": 64, "tmpfs_size": "16m"}
"""Limits for the fork-bomb and memory tests."""


def _problem(runtime: str) -> str | None:
    try:
        check_docker(runtime, IMAGE)
    except RuntimeError as e:
        return str(e)
    return None


@pytest.fixture(scope="module", params=["runc", "runsc"])
def runtime(request):
    problem = _problem(request.param)
    if problem:
        pytest.skip(problem)
    return request.param


def containers_left(episode_id: str) -> list[str]:
    found = subprocess.run(
        ["docker", "ps", "-aq", "--filter", f"label=inkling.episode={episode_id}"],
        capture_output=True,
        text=True,
        check=True,
    )
    return found.stdout.split()


def host_processes_with(marker: str) -> list[int]:
    """Host pids whose command line contains ``marker`` (containers' processes under runc)."""
    found = []
    for pid in filter(str.isdigit, os.listdir("/proc")):
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                if marker.encode() in f.read():
                    found.append(int(pid))
        except OSError:
            pass
    return found


class ContainerEpisode(Episode):
    episode_id: str


def run_contained(runtime, tmp, agent_dir, model, budget=None, env=None, **limits):
    env = env or FakeEnv("cube_lifting", 3)
    out = tmp / "out"
    boxes: list[ContainerSandbox] = []

    def make(workdir: Path) -> ContainerSandbox:
        boxes.append(ContainerSandbox(workdir, runtime=runtime, **limits))
        return boxes[-1]

    result = run_episode(
        "cube_lifting",
        3,
        agent_dir,
        model.url,
        budget or Budget(wall_s=60),
        out,
        make_env=lambda t, s, r: env,
        sandbox=make,
    )
    ep = ContainerEpisode(result, env, out)
    ep.episode_id = boxes[0].episode
    return ep


@pytest.fixture
def episode(tmp_path, runtime):
    def run(agent_dir, model, budget=None, env=None, **limits) -> ContainerEpisode:
        return run_contained(runtime, tmp_path, agent_dir, model, budget, env, **limits)

    return run


def write_agent(folder: Path, source: str) -> Path:
    folder.mkdir()
    (folder / "agent.py").write_text(textwrap.dedent(source))
    return folder


# ---------------------------------------------------------------- the probe

PROBE = r"""
import ctypes, json, os, socket, stat

def attempt(fn):
    try:
        value = fn()
        return "OK" if value is None else ("OK " + repr(value))[:300]
    except BaseException as e:
        return type(e).__name__

def write(path):
    with open(path, "w") as f:
        f.write("x")

def read(path):
    if os.path.isdir(path):
        return os.listdir(path)
    with open(path, "rb") as f:
        return f.read(16)

def connect(family, address):
    s = socket.socket(family)
    s.settimeout(3)
    try:
        s.connect(address)
    finally:
        s.close()

def sockets():
    found = []
    for folder, dirs, files in os.walk("/"):
        if folder in ("/proc", "/sys"):
            dirs[:] = []
            continue
        for name in files:
            path = os.path.join(folder, name)
            try:
                if stat.S_ISSOCK(os.lstat(path).st_mode):
                    found.append(path)
            except OSError:
                pass
    return found

status = {}
for line in open("/proc/self/status").read().splitlines():
    key, _, value = line.partition(":")
    status[key] = value.strip()
prctl = ctypes.CDLL(None, use_errno=True).prctl
PROBE_RESULT = {
    "ids": [os.getuid(), os.geteuid(), os.getgid(), os.getegid(), sorted(os.getgroups())],
    "caps": {k: status.get(k) for k in ("CapInh", "CapPrm", "CapEff", "CapBnd", "CapAmb")},
    "no_new_privs": prctl(39, 0, 0, 0, 0),
    "setuid0": attempt(lambda: os.setuid(0)),
    "imports": {m: attempt(lambda m=m: __import__(m) and None) for m in
                ["robosuite", "mujoco", "capx", "harness", "pip", "torch"]},
    "tcp": {f"{h}:{p}": attempt(lambda h=h, p=p: connect(socket.AF_INET, (h, p))) for h, p in TCP},
    "dns": attempt(lambda: socket.getaddrinfo("pypi.org", 443)),
    "interfaces": [name for _, name in socket.if_nameindex()],
    "abstract_socket": attempt(lambda: connect(socket.AF_UNIX, "\0" + ABSTRACT)),
    "docker_socket": {p: attempt(lambda p=p: connect(socket.AF_UNIX, p))
                      for p in ["/var/run/docker.sock", "/run/docker.sock"]},
    "reads": {p: attempt(lambda p=p: read(p)) for p in READS},
    "srv": attempt(lambda: read("/srv")),
    "writes": {p: attempt(lambda p=p: write(p)) for p in WRITES},
    "sockets": sockets(),
    "pids": sorted(int(p) for p in os.listdir("/proc") if p.isdigit()),
}
print("PROBE " + json.dumps(PROBE_RESULT))
"""

WRITES = [
    "/x",
    "/agent/x",
    "/run/x",
    "/etc/x",
    "/usr/local/lib/python3.11/site-packages/x.py",
    "/var/tmp/x",
    "/home/x",
    "/dev/shm/x",
    "/dev/mqueue/x",
    "/tmp/x",
]


class Listener:
    """TCP listeners on the host's loopback and on every address (so on the Docker bridge
    gateway and the host's own IP), and an abstract Unix socket; counts what connects."""

    def __init__(self) -> None:
        self.accepted = 0
        self.socks = []
        for family, address in [
            (socket.AF_INET, ("127.0.0.1", 0)),
            (socket.AF_INET, ("0.0.0.0", 0)),
            (socket.AF_UNIX, f"\0inkling-isolation-{os.urandom(8).hex()}"),
        ]:
            s = socket.socket(family)
            s.bind(address)
            s.listen(8)
            self.socks.append(s)
            threading.Thread(target=self._accept, args=(s,), daemon=True).start()
        self.loopback_port = self.socks[0].getsockname()[1]
        self.any_port = self.socks[1].getsockname()[1]
        self.abstract = self.socks[2].getsockname()[1:].decode()

    def _accept(self, s: socket.socket) -> None:
        while True:
            try:
                conn, _ = s.accept()
            except OSError:
                return
            self.accepted += 1
            conn.close()

    def close(self) -> None:
        for s in self.socks:
            s.shutdown(socket.SHUT_RDWR)  # wakes the accept() that holds the socket open
            s.close()


def host_addresses() -> list[str]:
    """The Docker bridge's gateway and the host's address on its default route."""
    gateway = subprocess.run(
        ["docker", "network", "inspect", "bridge", "-f", "{{(index .IPAM.Config 0).Gateway}}"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    probe.connect(("192.0.2.1", 9))  # no packet is sent
    own = probe.getsockname()[0]
    probe.close()
    return [a for a in (gateway, own) if a]


@pytest.fixture(scope="module")
def other_episode(tmp_path_factory):
    """Another episode's run folder on the host, with something worth stealing."""
    folder = tmp_path_factory.mktemp("other-run") / "cube_lifting" / "seed_7"
    folder.mkdir(parents=True)
    (folder / "llm.jsonl").write_text('{"secret": "another episode"}\n')
    return folder


@pytest.fixture(scope="module", params=["agent", "runner"])
def probe(request, runtime, tmp_path_factory, other_episode):
    """Run the probe in the agent or in a code runner; return (its result, the listener)."""
    tmp = tmp_path_factory.mktemp(f"probe-{runtime}-{request.param}")
    listener = Listener()
    with FakeModel(["x"]) as model:
        model_port = int(model.url.rsplit(":", 1)[1].split("/")[0])
        tcp = [
            ("127.0.0.1", 8116),
            ("127.0.0.1", model_port),
            ("127.0.0.1", listener.loopback_port),
        ]
        tcp += [(a, listener.any_port) for a in host_addresses()]
        tcp += [(a, model_port) for a in host_addresses()]
        tcp += [("1.1.1.1", 53)]
        reads = [
            str(ROOT / "harness" / "sim_server.py"),
            str(ROOT),
            str(other_episode / "llm.jsonl"),
            str(tmp / "out"),
            str(Path.home()),
            "/root",
        ]
        code = (
            f"TCP = {tcp!r}\nABSTRACT = {listener.abstract!r}\nREADS = {reads!r}\n"
            f"WRITES = {WRITES!r}\n{PROBE}"
        )
        if request.param == "agent":
            agent = write_agent(
                tmp / "agent",
                f"CODE = {code!r}\n\ndef solve(task, robot, llm):\n    exec(CODE, {{}})\n",
            )
        else:
            agent = write_agent(tmp / "agent", CHAT_AND_RUN)
            model.replies = [f"```python\n{code}```"]
        ep = run_contained(runtime, tmp, agent, model)
    listener.close()
    if request.param == "agent":
        text = ep.log
    else:
        (run,) = ep.json("runs.json")
        assert run["ok"], run["stderr"]
        text = run["stdout"]
    line = next(ln for ln in text.splitlines() if ln.startswith("PROBE "))
    return json.loads(line.removeprefix("PROBE ")), listener, ep


def failed(outcome: str) -> bool:
    return not outcome.startswith("OK")


def test_runs_as_nobody_without_capabilities_or_new_privileges(probe):
    found, _, _ = probe
    uid, euid, gid, egid, groups = found["ids"]
    assert (uid, euid, gid, egid) == (65534, 65534, 65534, 65534) and set(groups) <= {65534}
    assert set(found["caps"].values()) == {"0000000000000000"}, found["caps"]
    assert found["no_new_privs"] == 1
    assert found["setuid0"] == "PermissionError"


def test_no_network(probe):
    found, listener, _ = probe
    assert all(failed(outcome) for outcome in found["tcp"].values()), found["tcp"]
    assert "127.0.0.1:8116" in found["tcp"]  # the PyRoKi IK server's address
    assert failed(found["dns"]) and failed(found["abstract_socket"])
    assert found["interfaces"] == ["lo"]
    assert listener.accepted == 0


def test_no_simulator_or_installer(probe):
    found, _, _ = probe
    assert set(found["imports"].values()) == {"ModuleNotFoundError"}, found["imports"]


def test_host_paths_are_invisible(probe):
    found, _, _ = probe
    assert all(failed(outcome) for outcome in found["reads"].values()), found["reads"]
    assert found["srv"] == "OK []"


def test_only_tmp_is_writable(probe):
    found, _, _ = probe
    writable = [path for path, outcome in found["writes"].items() if not failed(outcome)]
    assert writable == ["/tmp/x"], found["writes"]


def test_only_its_own_socket_and_no_other_process_is_visible(probe):
    found, _, ep = probe
    assert found["sockets"] == ["/run/sim.sock"]
    assert failed(found["docker_socket"]["/var/run/docker.sock"])
    assert found["pids"] == [1]
    assert containers_left(ep.episode_id) == []


# ---------------------------------------------------------------- attacks, one episode each


def test_agent_cannot_move_the_robot(tmp_path, episode):
    agent = write_agent(
        tmp_path / "agent",
        """
        from inkling_robo import Forbidden

        def solve(task, robot, llm):
            for name, args in [("goto_pose", ([0.5, 0, 0.3], [0, 0, 1, 0])),
                               ("open_gripper", ()), ("close_gripper", ())]:
                try:
                    getattr(robot.api, name)(*args)
                    print("MOVED", name)
                except Forbidden:
                    print("forbidden", name)
        """,
    )
    with FakeModel(["x"]) as model:
        ep = episode(agent, model)
    assert ep.env.motions == [] and "MOVED" not in ep.log
    assert ep.log.count("forbidden") == 3 and ep.result["ended"] == "returned"


def test_the_agent_cannot_reach_a_live_runner_or_its_token(tmp_path, episode):
    # While run 0 is live, the agent looks for any socket but its own, any process but its own,
    # and the run token in every process's environment and command line it can see.
    agent = write_agent(
        tmp_path / "agent",
        """
        import json, os, re, socket, stat, threading, time

        SEEN = {}

        def scan():
            time.sleep(3)
            sockets = []
            for folder, dirs, files in os.walk("/"):
                if folder in ("/proc", "/sys"):
                    dirs[:] = []
                    continue
                for name in files:
                    path = os.path.join(folder, name)
                    try:
                        if stat.S_ISSOCK(os.lstat(path).st_mode):
                            sockets.append(path)
                    except OSError:
                        pass
            hits, pids = [], sorted(p for p in os.listdir("/proc") if p.isdigit())
            for pid in pids:
                for kind in ("environ", "cmdline"):
                    try:
                        if b"INKLING_ROBO_TOKEN" in open(f"/proc/{pid}/{kind}", "rb").read():
                            hits.append([pid, kind])
                    except OSError:
                        pass
            guesses = {}
            for path in ["/run/r0.sock", "/tmp/r0.sock", "/run/sim.sock"]:
                s = socket.socket(socket.AF_UNIX)
                try:
                    s.connect(path)
                    guesses[path] = "CONNECTED"
                except OSError as e:
                    guesses[path] = type(e).__name__
            SEEN.update(sockets=sockets, pids=pids, hits=hits, guesses=guesses)

        def solve(task, robot, llm):
            reply = llm.chat([{"role": "user", "content": "go"}])
            code = re.search(r"```python\\n(.*?)```", reply, re.S).group(1)
            th = threading.Thread(target=scan)
            th.start()
            result = robot.run(code)
            th.join()
            print("SEEN " + json.dumps(SEEN))
            print("run ok", result.ok, result.stdout.strip())
        """,
    )
    with FakeModel(
        ["```python\nimport time\ntime.sleep(8)\nopen_gripper()\nprint('ran')\n```"]
    ) as m:
        ep = episode(agent, m)
    seen = json.loads(next(ln for ln in ep.log.splitlines() if ln.startswith("SEEN "))[5:])
    assert seen["sockets"] == ["/run/sim.sock"] and seen["pids"] == ["1"]
    assert seen["hits"] == []
    assert "CONNECTED" not in seen["guesses"].values()
    (run,) = ep.json("runs.json")
    assert run["ok"] and run["seconds"] < 20 and "run ok True ran" in ep.log
    assert ep.env.motions == [("open_gripper", (), {})]


def test_run_requires_code_from_a_model_reply(tmp_path, episode):
    agent = write_agent(
        tmp_path / "agent",
        """
        from inkling_robo import RunRefused

        def solve(task, robot, llm):
            for code in ["open_gripper()", None, "open_gripper()", "close_gripper()"]:
                if code is None:
                    llm.chat([{"role": "user", "content": "close it"}])
                    continue
                try:
                    print("RAN", code, robot.run(code).ok)
                except RunRefused:
                    print("refused", code)
        """,
    )
    with FakeModel(["close_gripper()"]) as model:
        ep = episode(agent, model)
    assert ep.log.count("refused open_gripper()") == 2 and "RAN close_gripper() True" in ep.log
    assert ep.env.motions == [("close_gripper", (), {})]


def test_token_budget(tmp_path, episode):
    agent = write_agent(
        tmp_path / "agent",
        """
        from inkling_robo import BudgetExceeded

        def solve(task, robot, llm):
            n = 0
            try:
                while True:
                    llm.chat([{"role": "user", "content": "more"}], max_tokens=50000)
                    n += 1
            except BudgetExceeded as e:
                print("calls", n, e.message)
        """,
    )
    with FakeModel(["x"], prompt_tokens=30000, completion_tokens=30000) as model:
        ep = episode(agent, model, budget=Budget(calls=10, tokens=100_000, wall_s=60))
    assert "calls 2 token budget (100000) spent" in ep.log
    assert len(model.requests) == 2 and ep.json("usage.json")["limit_hit"] == "tokens"


@pytest.mark.parametrize("who", ["agent", "runner"])
def test_outliving_the_time_limit(tmp_path, episode, who):
    marker = f"inkling-hang-{os.getpid()}-{who}"
    hang = f"""
import subprocess, sys
subprocess.Popen([sys.executable, "-c", "while True: pass", {marker!r}])
while True:
    pass
"""
    if who == "agent":
        source = "def solve(task, robot, llm):\n" + textwrap.indent(hang, "    ")
        replies = ["x"]
    else:
        source, replies = CHAT_AND_RUN, [f"```python\n{hang}```"]
    agent = write_agent(tmp_path / "agent", source)
    with FakeModel(replies) as model:
        t0 = time.monotonic()
        ep = episode(agent, model, budget=Budget(wall_s=6))
        took = time.monotonic() - t0
    assert ep.result["ended"] == "timeout" and ep.result["success"] is False
    assert took < 6 + 15, took
    if who == "runner":
        (run,) = ep.json("runs.json")
        assert run["timed_out"] and not run["ok"]
    assert containers_left(ep.episode_id) == []
    assert host_processes_with(marker) == []


@pytest.mark.parametrize("who", ["agent", "runner"])
def test_no_process_survives_its_container(tmp_path, episode, who):
    # A child in a new session and a double-forked daemon, left behind on purpose.
    marker = f"inkling-daemon-{os.getpid()}-{who}"
    spawn = f"""
import os, subprocess, sys, time
subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)", {marker!r}],
                 start_new_session=True)
if os.fork() == 0:
    os.setsid()
    if os.fork() == 0:
        os.execv(sys.executable, [sys.executable, "-c", "import time; time.sleep(600)", {marker!r}])
    os._exit(0)
time.sleep(0.5)
print("spawned")
"""
    if who == "agent":
        source = "def solve(task, robot, llm):\n" + textwrap.indent(spawn, "    ")
        replies = ["x"]
    else:
        source, replies = CHAT_AND_RUN, [f"```python\n{spawn}```"]
    agent = write_agent(tmp_path / "agent", source)
    with FakeModel(replies) as model:
        ep = episode(agent, model)
    assert ep.result["ended"] == "returned" and "spawned" in (
        ep.log + json.dumps(ep.json("runs.json"))
    )
    assert containers_left(ep.episode_id) == []
    assert host_processes_with(marker) == []


def test_fork_bomb_hits_the_pids_limit_and_the_host_stays_responsive(tmp_path, episode):
    marker = f"inkling-bomb-{os.getpid()}"
    agent = write_agent(
        tmp_path / "agent",
        f"""
        import os, subprocess, sys, time

        def solve(task, robot, llm):
            if os.fork() == 0:
                sys.argv.append({marker!r})
                while True:
                    try:
                        os.fork()
                    except OSError:
                        time.sleep(0.01)
            time.sleep(3)
            try:
                subprocess.run(["true"])
                print("SPAWNED")
            except OSError as e:
                print("spawn refused:", type(e).__name__)
            t = time.monotonic()
            robot.observe()
            print("observe took", round(time.monotonic() - t, 3))
        """,
    )
    slowest = []
    done = threading.Event()

    def watch_host() -> None:  # how long the host takes to start a process, meanwhile
        while not done.is_set():
            t = time.monotonic()
            subprocess.run(["true"])
            slowest.append(time.monotonic() - t)
            time.sleep(0.1)

    watcher = threading.Thread(target=watch_host)
    watcher.start()
    try:
        with FakeModel(["x"]) as model:
            ep = episode(agent, model, budget=Budget(wall_s=30), **SMALL)
    finally:
        done.set()
        watcher.join()
    assert "spawn refused: BlockingIOError" in ep.log and "SPAWNED" not in ep.log
    observe = float(ep.log.split("observe took ")[1].split()[0])
    assert observe < 2.0 and max(slowest) < 2.0, (observe, max(slowest))
    assert ep.result["ended"] == "returned"
    assert containers_left(ep.episode_id) == [] and host_processes_with(marker) == []


@pytest.mark.parametrize("who", ["agent", "runner"])
def test_memory_exhaustion_is_killed_and_the_harness_survives(tmp_path, episode, who):
    grab = "hog = []\nwhile True:\n    hog.append(b'x' * 2**24)\n"
    if who == "agent":
        source = "def solve(task, robot, llm):\n" + textwrap.indent(grab, "    ")
        replies = ["x"]
    else:
        source, replies = CHAT_AND_RUN, [f"```python\n{grab}```"]
    agent = write_agent(tmp_path / "agent", source)
    with FakeModel(replies) as model:
        ep = episode(agent, model, budget=Budget(wall_s=60), **SMALL)
    assert ep.result["success"] is False and ep.env.checks == 1
    if who == "agent":
        assert ep.result["ended"] == "error"
    else:  # the run is killed; the agent carries on and returns
        (run,) = ep.json("runs.json")
        assert not run["ok"] and run["exit_status"] != 0 and not run["timed_out"]
        assert ep.result["ended"] == "returned" and "run: False" in ep.log
    assert containers_left(ep.episode_id) == []


@pytest.mark.parametrize(
    "garbage",
    [
        "struct.pack('>I', 2**31)",
        "struct.pack('>I', 3) + b'\\xc1\\xc1\\xc1'",
        "frame(nested(40))",
    ],
)
def test_malformed_and_oversized_messages(tmp_path, episode, garbage):
    agent = write_agent(
        tmp_path / "agent",
        f"""
        import socket, struct
        import msgpack

        def frame(obj):
            body = msgpack.packb(obj, use_bin_type=True)
            return struct.pack(">I", len(body)) + body

        def nested(n):
            x = []
            for _ in range(n):
                x = [x]
            return {{"id": 1, "method": "observe", "params": {{"x": x}}}}

        def solve(task, robot, llm):
            try:
                robot._conn._sock.sendall({garbage})
            except OSError:
                pass
            try:
                robot.observe()
                print("STILL SERVED")
            except Exception as e:
                print("closed:", type(e).__name__)
            try:
                socket.socket(socket.AF_UNIX).connect("/run/sim.sock")
                print("RECONNECTED")
            except OSError:
                print("no second connection")
        """,
    )
    with FakeModel(["x"]) as model:
        ep = episode(agent, model)
    assert "STILL SERVED" not in ep.log and "RECONNECTED" not in ep.log
    assert "no second connection" in ep.log and ep.result["ended"] == "returned"
    assert any("bad frame" in e["message"] for e in ep.json("errors.json"))


def test_output_is_capped(tmp_path, episode):
    agent = write_agent(
        tmp_path / "agent",
        CHAT_AND_RUN.replace(
            "def solve(task, robot, llm):",
            "def solve(task, robot, llm):\n    print('y' * 3_000_000)",
        ),
    )
    with FakeModel(["```python\nprint('z' * 2_000_000)\n```"]) as model:
        ep = episode(agent, model)
    assert len(ep.log) < AGENT_OUTPUT_CAP + 1000 and "more bytes dropped" in ep.log
    (run,) = ep.json("runs.json")
    assert run["ok"] and len(run["stdout"]) < 300_000


def test_kill_right_after_start_is_quick_and_leaves_nothing(tmp_path, runtime):
    # As when the episode ends while a run is starting: kill() lands before the container runs.
    # The socket's listener accepts every connection and holds it, as the sim server does. One
    # that never accepts fills its backlog, and a later runner then blocks in connect(), which
    # under gVisor (--host-uds) SIGKILL cannot interrupt: docker kill would wait out its 10 s.
    srv = listen_unix(tmp_path / "r0.sock")
    held: list[socket.socket] = []

    def accept_all() -> None:
        while True:
            try:
                held.append(srv.accept()[0])
            except OSError:
                return

    threading.Thread(target=accept_all, daemon=True).start()
    box = ContainerSandbox(tmp_path / "work", runtime=runtime)
    try:
        for _ in range(5):
            proc = box.start_runner(tmp_path / "r0.sock", "0" * 32)
            t0 = time.monotonic()
            proc.kill()
            assert time.monotonic() - t0 < 5 and proc.wait(timeout=0) is not None
            assert containers_left(box.episode) == []
    finally:
        box.close()
        srv.shutdown(socket.SHUT_RDWR)
        srv.close()
        for conn in held:
            conn.close()


# ---------------------------------------------------------------- the real thing works


def test_base_agent_completes_an_episode(tmp_path, episode):
    from tests.base_agent import AGENT_DIR
    from tests.test_base_agent import LIFT_CODE

    with FakeModel([LIFT_CODE]) as model:
        ep = episode(AGENT_DIR, model)
    (run,) = ep.json("runs.json")
    assert run["accepted"] and run["ok"] and run["stdout"] == "lifted\n", run
    assert [m[0] for m in ep.env.motions] == [
        "open_gripper",
        "goto_pose",
        "close_gripper",
        "goto_pose",
    ]
    assert ep.result["success"] is True and ep.result["ended"] == "returned"
    assert containers_left(ep.episode_id) == []


def test_agent_files_never_reach_the_runner(tmp_path, episode):
    # The runner container mounts only its socket: a skills.py (or anything else) in the agent
    # folder is neither mounted nor executed there. If it ran, the run would fail.
    agent = write_agent(
        tmp_path / "agent",
        CHAT_AND_RUN.replace(
            "def solve", "import os\nprint('agent sees', sorted(os.listdir('/agent')))\n\ndef solve"
        ),
    )
    (agent / "skills.py").write_text("raise SystemExit('skills.py ran')\nlift_height = 0.1\n")
    (agent / "notes.md").write_text("private notes")
    code = (
        "import os, sys\n"
        "print(os.path.exists('/agent'), 'lift_height' in globals(), 'skills' in sys.modules)"
    )
    with FakeModel([f"```python\n{code}\n```"]) as model:
        ep = episode(agent, model)
    assert "agent sees ['agent.py', 'notes.md', 'skills.py']" in ep.log
    (run,) = ep.json("runs.json")
    assert run["ok"] and run["stdout"] == "False False False\n", run


@pytest.mark.sim
def test_oracle_reply_succeeds_on_the_simulator_in_containers(tmp_path, runtime, ik_server):
    from harness.capx_env import make_env
    from tests.base_agent import AGENT_DIR
    from tests.oracle_model import OracleModel

    with OracleModel(["cube_lifting"]) as model:
        boxes = []

        def make(workdir):
            boxes.append(ContainerSandbox(workdir, runtime=runtime))
            return boxes[-1]

        result = run_episode(
            "cube_lifting", 1, AGENT_DIR, model.url, Budget(), tmp_path / "out",
            make_env=make_env, sandbox=make,
        )  # fmt: skip
    (run,) = json.loads((tmp_path / "out" / "runs.json").read_text())
    assert run["accepted"] and run["ok"], run
    assert result["success"] is True and result["ended"] == "returned"
    assert containers_left(boxes[0].episode) == []


# ---------------------------------------------------------------- job labels and cgroup parent


def test_job_label_and_cgroup_parent_reach_the_container(tmp_path, runtime):
    # What the box's scorer relies on: every container of a job carries its job id (so the
    # job's leftovers can be swept by job), and with a cgroup parent the container's cgroup sits
    # under it (so the job's memory cap covers the episode containers).
    import secrets

    tag = secrets.token_hex(4)
    driver = subprocess.run(
        ["docker", "info", "--format", "{{.CgroupDriver}}"], capture_output=True, text=True
    ).stdout.strip()
    parent = f"inklingtest-{tag}.slice" if driver == "systemd" else f"/inklingtest-{tag}"
    job = f"p{tag[:4].translate(str.maketrans('abcdef', '123456'))}-1"
    box = ContainerSandbox(
        tmp_path / "work", runtime=runtime, labels={"inkling.job": job}, cgroup_parent=parent
    )
    name = f"inkling-{box.episode}-agent"
    proc = box._start(name, [], ["-c", "import time; time.sleep(60)"], AGENT_OUTPUT_CAP, True, {})
    try:
        deadline = time.monotonic() + 30
        pid = 0
        while time.monotonic() < deadline and not pid:
            info = subprocess.run(
                ["docker", "inspect", "-f", "{{.State.Pid}}", name], capture_output=True, text=True
            )
            pid = int(info.stdout.strip() or 0)
            time.sleep(0.2)
        assert pid, "the container did not start"
        inspect = json.loads(
            subprocess.run(["docker", "inspect", name], capture_output=True, text=True).stdout
        )[0]
        assert inspect["Config"]["Labels"]["inkling.job"] == job
        assert inspect["Config"]["Labels"]["inkling.episode"] == box.episode
        assert inspect["HostConfig"]["CgroupParent"] == parent
        cgroup = Path(f"/proc/{pid}/cgroup").read_text()
        want = f"/{parent}/" if driver != "systemd" else f"/inklingtest.slice/{parent}/"
        assert want in cgroup, cgroup
        by_job = subprocess.run(
            ["docker", "ps", "-aq", "--no-trunc", "--filter", f"label=inkling.job={job}"],
            capture_output=True,
            text=True,
        ).stdout.split()
        assert by_job == [inspect["Id"]]
    finally:
        proc.kill()
        box.close()
    assert containers_left(box.episode) == []
