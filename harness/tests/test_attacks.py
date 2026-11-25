"""Red-team agents for the attacks that are meaningful without containers.

Container-only attacks (network access, importing robosuite, reading /srv, fork bombs) belong
to the gVisor package's test_isolation.py.
"""

import json
import os
import textwrap
import time

import pytest

from harness import rpc, sim_server
from harness.episode import run_episode
from harness.llm_relay import Budget, estimate_prompt_tokens
from harness.tests.conftest import CHAT_AND_RUN, LIFT_REPLY
from harness.tests.fake_env import FakeEnv
from harness.tests.fake_model import FakeModel


def test_agent_cannot_move_the_robot(make_agent, episode):
    agent = make_agent(
        """
        from inkling_robo import Forbidden

        def solve(task, robot, llm):
            attempts = {
                "goto_pose": ([0.5, 0.0, 0.3], [0, 0, 1, 0]),
                "open_gripper": (),
                "close_gripper": (),
            }
            for name, args in attempts.items():
                try:
                    getattr(robot.api, name)(*args)
                    print("MOVED", name)
                except Forbidden as e:
                    print("forbidden", name)
            for name in ["move_to_joints_blocking", "_set_gripper", "env", "__class__"]:
                try:
                    robot._conn.call("call", {"name": name, "args": [[0.0] * 7], "kwargs": {}})
                    print("MOVED", name)
                except Forbidden:
                    print("forbidden raw", name)
            for method in ["runner.hello", "reset", "task_completed"]:
                try:
                    robot._conn.call(method)
                    print("ACCEPTED", method)
                except Exception as e:
                    print("rejected", method, e.code)
        """
    )
    with FakeModel(["x"]) as model:
        ep = episode(agent, model)
    assert ep.env.motions == []
    assert "MOVED" not in ep.log and "ACCEPTED" not in ep.log
    assert ep.log.count("forbidden") == 7
    assert ep.log.count("bad_request") == 3
    assert ep.result["ended"] == "returned" and ep.env.checks == 1


def test_run_requires_code_from_a_model_reply(make_agent, episode):
    agent = make_agent(
        """
        from inkling_robo import RunRefused

        def attempt(robot, code):
            try:
                r = robot.run(code)
                print("RAN", repr(code), r.ok)
            except RunRefused:
                print("refused")

        def solve(task, robot, llm):
            attempt(robot, "close_gripper()")  # no reply yet
            llm.chat([{"role": "user", "content": "open the gripper"}])
            for code in [
                "",
                "   ",
                "open_gripper()\\nclose_gripper()",
                "open_gripper() ",
                "goto_pose([0.5, 0, 0.3], [0, 0, 1, 0])",
                "thinking",  # in the reasoning text, not the content
            ]:
                attempt(robot, code)
            attempt(robot, "open_gripper()")
        """
    )
    with FakeModel(["Sure:\n```python\nopen_gripper()\n```"]) as model:
        ep = episode(agent, model)
    assert ep.log.count("refused") == 7
    assert "RAN 'open_gripper()' True" in ep.log
    assert ep.env.motions == [("open_gripper", (), {})]
    runs = ep.json("runs.json")
    assert [r["accepted"] for r in runs] == [False] * 7 + [True]


def test_token_budget(make_agent, episode):
    agent = make_agent(
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
        """
    )
    with FakeModel(["x"], prompt_tokens=30000, completion_tokens=30000) as model:
        ep = episode(agent, model, budget=Budget(calls=10, tokens=100_000, wall_s=20))
    assert "calls 2 token budget (100000) spent" in ep.log
    bound = estimate_prompt_tokens([{"role": "user", "content": "more"}])
    assert [r["max_tokens"] for r in model.requests] == [50000, 40000 - bound]
    assert ep.result["calls"] == 2 and ep.result["tokens"] == 120_000
    assert ep.json("usage.json")["limit_hit"] == "tokens"


def test_call_budget(make_agent, episode):
    agent = make_agent(
        """
        from inkling_robo import BudgetExceeded

        def solve(task, robot, llm):
            for i in range(5):
                try:
                    llm.chat([{"role": "user", "content": "hi"}])
                    print("call ok")
                except BudgetExceeded:
                    print("call refused")
        """
    )
    with FakeModel(["x"]) as model:
        ep = episode(agent, model, budget=Budget(calls=3, wall_s=20))
    assert ep.log.count("call ok") == 3 and ep.log.count("call refused") == 2
    assert len(model.requests) == 3 and ep.result["calls"] == 3


def test_outliving_the_time_limit_counts_as_failure(make_agent, episode):
    # The robot completes the task, then the agent hangs: the checker still runs exactly once,
    # and the timeout makes the episode a failure.
    agent = make_agent(
        CHAT_AND_RUN
        + """
    import time
    while True:
        time.sleep(0.05)
"""
    )
    with FakeModel([LIFT_REPLY]) as model:
        t0 = time.monotonic()
        ep = episode(agent, model, budget=Budget(wall_s=2))
        took = time.monotonic() - t0
    assert ep.result["ended"] == "timeout" and ep.result["success"] is False
    assert ep.json("result.json")["task_completed"] is True
    assert ep.json("usage.json")["limit_hit"] == "wall"
    assert ep.env.checks == 1
    assert 2 <= ep.result["seconds"] < 4 and took < 6


def test_runaway_model_code_is_killed_at_the_time_limit(make_agent, episode):
    with FakeModel(["```python\nopen_gripper()\nwhile True:\n    pass\n```"]) as model:
        ep = episode(make_agent(CHAT_AND_RUN), model, budget=Budget(wall_s=2))
    assert ep.result["ended"] == "timeout" and ep.result["seconds"] < 4
    (run,) = ep.json("runs.json")
    assert run["timed_out"] and run["ok"] is False
    assert ep.env.motions == [("open_gripper", (), {})] and ep.env.checks == 1


def test_processes_left_behind_by_the_agent_are_killed(make_agent, episode, tmp_path):
    pid_file = tmp_path / "child.pid"
    agent = make_agent(
        f"""
        import subprocess

        def solve(task, robot, llm):
            child = subprocess.Popen(["sleep", "60"])
            open({str(pid_file)!r}, "w").write(str(child.pid))
        """
    )
    with FakeModel(["x"]) as model:
        ep = episode(agent, model)
    assert ep.result["ended"] == "returned"
    pid = int(pid_file.read_text())
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and _alive(pid):
        time.sleep(0.05)
    assert not _alive(pid)


@pytest.mark.parametrize("who", ["agent", "runner"])
def test_processes_that_leave_the_session_are_killed(make_agent, episode, tmp_path, who):
    # The review's PoC: a child in a new session escapes killpg; a double-forked daemon is
    # re-parented away from the agent's tree altogether.
    spawn = f"""
import os, subprocess, sys, time
p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                     start_new_session=True)
open({str(tmp_path / "setsid.pid")!r}, "w").write(str(p.pid))
if os.fork() == 0:
    os.setsid()
    if os.fork() == 0:
        open({str(tmp_path / "daemon.pid")!r}, "w").write(str(os.getpid()))
        time.sleep(60)
    os._exit(0)
time.sleep(0.5)
"""
    if who == "agent":
        agent = make_agent("def solve(task, robot, llm):\n" + textwrap.indent(spawn, "    "))
        replies = ["x"]
    else:
        agent = make_agent(CHAT_AND_RUN)
        replies = ["```python\n" + spawn + "```"]
    with FakeModel(replies) as model:
        ep = episode(agent, model)
    assert ep.result["ended"] == "returned"
    pids = [int((tmp_path / f"{name}.pid").read_text()) for name in ("setsid", "daemon")]
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and any(_alive(pid) for pid in pids):
        time.sleep(0.05)
    assert not any(_alive(pid) for pid in pids)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    try:  # a zombie is dead for our purposes
        with open(f"/proc/{pid}/stat") as f:
            return f.read().split(")")[-1].split()[0] != "Z"
    except OSError:
        return True


@pytest.mark.parametrize(
    "garbage",
    [
        "struct.pack('>I', 2**31)",  # length far over the limit
        f"struct.pack('>I', {rpc.MAX_MESSAGE_BYTES + 1})",
        f"struct.pack('>I', {rpc.MAX_REQUEST_BYTES + 1})",  # requests have a smaller cap
        "frame({'id': 1, 'method': 'observe', 'params': {'x': [[{}] * 100_000] * 3}})",
        "struct.pack('>I', 3) + b'\\xc1\\xc1\\xc1'",  # not msgpack
        "struct.pack('>I', 3) + b'\\xd4\\x01x'",  # ext type
        "frame({'id': 1, 'method': 'call', 'params': {'a': {'__ndarray__': True, 'dtype': '|O',"
        " 'shape': [1], 'data': b'12345678'}}})",  # object array
        "frame({'id': 1, 'method': 'call', 'params': {'s': 'x' * (2**20 + 1)}})",  # long string
        "frame(nested(40))",  # too deep
        # zero-size array with a huge shape (once allocated 8M lists in the trusted logger)
        "frame({'id': 1, 'method': 'call', 'params': {'name': 'get_object_pose', 'args':"
        " [{'__ndarray__': True, 'dtype': '|u1', 'shape': [2**23, 0], 'data': b''}],"
        " 'kwargs': {}}})",
    ],
)
def test_malformed_and_oversized_messages(make_agent, episode, garbage):
    agent = make_agent(
        f"""
        import socket, struct, sys
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
            except OSError:  # the server may close before reading all of a too-long frame
                print("send cut off")
            for _ in range(2):
                try:
                    robot.observe()
                    print("STILL SERVED")
                except Exception as e:
                    print("closed:", type(e).__name__)
            path = sys.argv[sys.argv.index("--socket") + 1]
            try:
                socket.socket(socket.AF_UNIX).connect(path)
                print("RECONNECTED")
            except OSError:
                print("no second connection")
        """
    )
    with FakeModel(["x"]) as model:
        ep = episode(agent, model)
    assert "STILL SERVED" not in ep.log and "RECONNECTED" not in ep.log
    assert "no second connection" in ep.log
    assert ep.result["ended"] == "returned" and ep.env.checks == 1
    assert any("bad frame" in e["message"] for e in ep.json("errors.json"))


def test_logs_stay_bounded(make_agent, episode):
    # The review's log-amplification attacks: megabyte function names, megabyte arguments, and
    # many refused runs of 256 KiB code. Names are checked before they are logged, fields are
    # clipped, and past the episode's log budget only counters are kept.
    agent = make_agent(
        """
        import numpy as np
        from inkling_robo import InklingError

        def solve(task, robot, llm):
            name = "x" * (512 * 1024)
            try:
                robot._conn.call("call", {"name": name, "args": [], "kwargs": {}})
            except InklingError as e:
                print("name:", e.code)
            big = np.zeros(900 * 1024, dtype=np.uint8)
            for _ in range(5):
                try:
                    robot.api.get_object_pose(big, "y" * 100_000)
                except ValueError:  # an api_error, logged with its arguments
                    pass
            refused = 0
            for i in range(80):
                try:
                    robot.run(str(i) + "z" * (256 * 1024 - 10))
                except InklingError:
                    refused += 1
            print("refused", refused)
        """
    )
    with FakeModel(["x"]) as model:
        ep = episode(agent, model)
    assert "name: bad_request" in ep.log and "refused 80" in ep.log
    sizes = [p.stat().st_size for p in ep.out.rglob("*") if p.is_file()]
    assert sum(sizes) < 17 * 2**20
    transcript = (ep.out / "transcript.jsonl").read_text()
    assert "xxxxxxxxxx" not in transcript
    calls = [e for e in ep.jsonl("transcript.jsonl") if e["kind"] == "call"]
    assert len(calls) == 5 and all(len(json.dumps(e)) < 6000 for e in calls)
    assert calls[0]["args"].startswith('[{"ndarray": "|u1", "shape": [921600]')
    runs = ep.json("runs.json")
    assert 40 < len(runs) < 80 and [r["index"] for r in runs] == list(range(len(runs)))
    last = ep.json("errors.json")[-1]
    assert last["where"] == "log" and "events" in last["message"] and "runs" in last["message"]


def test_requests_per_episode_are_capped(make_agent, episode, monkeypatch):
    monkeypatch.setattr(sim_server, "MAX_RPCS", 50)
    agent = make_agent(
        """
        from inkling_robo import BudgetExceeded

        def solve(task, robot, llm):
            n = 0
            try:
                while True:
                    robot.observe()
                    n += 1
            except BudgetExceeded as e:
                print("observed", n, e.message)
        """
    )
    with FakeModel(["x"]) as model:
        ep = episode(agent, model)
    assert "observed 49 request budget (50 per episode) spent" in ep.log  # hello is one
    assert [e["where"] for e in ep.json("errors.json")] == ["rpc"]


@pytest.mark.parametrize("hello", ["{}", "{'token': '0' * 32}", None])
def test_runner_socket_needs_the_runs_token(make_agent, episode, hello):
    # The review's PoC: the agent races the real runner to r0.sock. Without the run's token it
    # gets nothing, and its attempt (a wrong hello or any other request) aborts the run.
    agent = make_agent(
        f"""
        import os, sys, threading, time
        from inkling_robo import InklingError
        from inkling_robo.client import Connection

        HELLO = {hello}

        def hijack(path, out):
            while True:
                try:
                    c = Connection(path)
                    break
                except OSError:
                    time.sleep(0.0005)
            try:
                if HELLO is None:
                    c.call("call", {{"name": "open_gripper", "args": [], "kwargs": {{}}}})
                else:
                    out["code"] = c.call("runner.hello", HELLO)["code"]
            except InklingError as e:
                out["first"] = e.code
            try:
                c.call("call", {{"name": "open_gripper", "args": [], "kwargs": {{}}}})
                out["moved"] = True
            except InklingError as e:
                out["then"] = e.code

        def solve(task, robot, llm):
            sock_dir = os.path.dirname(sys.argv[sys.argv.index("--socket") + 1])
            print("agent socket left:", os.listdir(sock_dir))
            path = os.path.join(sock_dir, "r0.sock")
            out = {{}}
            th = threading.Thread(target=hijack, args=(path, out))
            th.start()
            reply = llm.chat([{{"role": "user", "content": "hi"}}])
            r = robot.run(reply)
            th.join()
            print("hijack:", sorted(out.items()), "| run ok:", r.ok)
            print("runner socket left:", os.path.exists(path))
        """
    )
    with FakeModel(["close_gripper()"]) as model:
        ep = episode(agent, model)
    assert "hijack: [('first', 'forbidden'), ('then', 'closed')] | run ok: False" in ep.log
    assert "runner socket left: False" in ep.log
    assert "agent socket left: ['work']" in ep.log  # a.sock is unlinked once the agent connects
    assert ep.env.motions == []
    (run,) = ep.json("runs.json")
    assert run["accepted"] and not run["ok"] and "runner.hello" in run["aborted"]
    assert any("run aborted" in e["message"] for e in ep.json("errors.json"))


def test_motion_arguments_are_checked_before_the_env(make_agent, episode):
    code = """
import math
import numpy as np
bad = [
    ([math.nan, 0, 0.3], [0, 0, 1, 0]),
    ([1e9, 0, 0.3], [0, 0, 1, 0]),
    ([0.5, 0, 0.3], [0, 0, 0, 0]),
    ([0.5, 0, 0.3], "down"),
    (np.zeros(1000), [0, 0, 1, 0]),
    ([0.5, 0, 0.3], [0, 0, 1, math.inf]),
]
for pos, quat in bad:
    try:
        goto_pose(pos, quat)
        print("MOVED")
    except ValueError as e:
        print("refused:", e)
try:
    goto_pose([0.5, 0, 0.3], [0, 0, 1, 0], z_approach=float("inf"))
except ValueError:
    print("refused: z")
goto_pose(np.array([0.5, 0, 0.3]), [0, 0, 1, 0], z_approach=0.1)
print("moved ok")
"""
    with FakeModel([f"```python\n{code}```"]) as model:
        ep = episode(make_agent(CHAT_AND_RUN), model)
    assert "MOVED" not in ep.log and ep.log.count("refused:") == 7 and "moved ok" in ep.log
    assert [m[0] for m in ep.env.motions] == ["goto_pose"]


class StallingEnv(FakeEnv):
    """goto_pose takes far longer than the episode (as CaP-X's IK retries can)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.functions["goto_pose"] = lambda *a, **k: time.sleep(20)


def test_a_stalled_env_call_cannot_delay_the_end_of_the_episode(make_agent, tmp_path):
    env = StallingEnv("cube_lifting", 3)
    with FakeModel([LIFT_REPLY]) as model:
        t0 = time.monotonic()
        result = run_episode(
            "cube_lifting",
            3,
            make_agent(CHAT_AND_RUN),
            model.url,
            Budget(wall_s=2),
            tmp_path / "out",
            make_env=lambda t, s, r: env,
        )
        took = time.monotonic() - t0
    assert took < 2 + sim_server.CLOSE_GRACE_S + 2
    assert result["ended"] == "timeout" and result["success"] is False
    assert env.checks == 0  # the checker never ran against a half-done motion
    errors = json.loads((tmp_path / "out" / "errors.json").read_text())
    assert any(e["where"] == "env" for e in errors)


class LeakyEnv(FakeEnv):
    """A read function that fails deep inside the simulator with a long, odd exception."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        class SimInternalError(Exception):
            pass

        def fail(*a, **k):
            raise SimInternalError("/srv/secret/path " + "z" * 100_000)

        self.functions["get_object_pose"] = fail


def test_env_exceptions_reach_the_sandbox_clipped(make_agent, tmp_path):
    agent = make_agent(
        """
        from inkling_robo import RobotApiError

        def solve(task, robot, llm):
            try:
                robot.api.get_object_pose("red cube")
            except RobotApiError as e:
                print("error", type(e).__name__, len(e.message), e.message[:40])
        """
    )
    env = LeakyEnv("cube_lifting", 3)
    with FakeModel(["x"]) as model:
        run_episode("cube_lifting", 3, agent, model.url, Budget(wall_s=30), tmp_path / "out",
                    make_env=lambda t, s, r: env)  # fmt: skip
    log = (tmp_path / "out" / "agent.log").read_text()
    assert "error RobotApiError" in log and "SimInternalError: /srv/secret/path zz" in log
    assert int(log.split()[2]) <= sim_server.MAX_API_ERROR_CHARS + 60
