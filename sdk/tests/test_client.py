import ast
import sys
import textwrap
import threading
from pathlib import Path

import numpy as np
import pytest
from inkling_robo import (
    BudgetExceeded,
    Forbidden,
    InklingError,
    LLMError,
    RunRefused,
    run,
    runner,
)
from inkling_robo.client import Connection
from inkling_robo.wire import MAX_REQUEST_BYTES, WireError

from harness.rpc import RpcError, listen_unix, serve_connection

SDK = Path(__file__).resolve().parent.parent / "inkling_robo"


class FakeServer:
    """Serves one connection on a Unix socket with a scripted handler."""

    def __init__(self, path: Path, handle) -> None:
        self.path = str(path)
        self.calls: list[tuple[str, dict]] = []
        srv = listen_unix(path)

        def accept() -> None:
            conn, _ = srv.accept()
            srv.close()
            serve_connection(conn, lambda m, p: self._record(handle, m, p))

        threading.Thread(target=accept, daemon=True).start()

    def _record(self, handle, method, params):
        self.calls.append((method, params))
        return handle(method, params)


def errors(method, params):
    codes = {"forbidden": "forbidden", "refused": "refused", "budget": "budget", "weird": "x"}
    if method == "value":
        raise RpcError("api_error", "Invalid object name: x", "ValueError")
    if method == "oserror":
        raise RpcError("api_error", "disk", "OSError")
    if method == "upstream":
        data = {"kind": "http", "status": 503, "retry_after": 240.0}
        raise RpcError("upstream", "model server returned HTTP 503", data=data)
    raise RpcError(codes[method], "no")


def test_errors_map_to_exceptions(tmp_path):
    conn = Connection(FakeServer(tmp_path / "s", errors).path)
    with pytest.raises(ValueError, match="Invalid object name"):
        conn.call("value")
    with pytest.raises(InklingError) as e:  # only whitelisted builtins are re-raised as such
        conn.call("oserror")
    assert e.value.code == "api_error"
    for method, exc in [
        ("forbidden", Forbidden),
        ("refused", RunRefused),
        ("budget", BudgetExceeded),
    ]:
        with pytest.raises(exc):
            conn.call(method)
    with pytest.raises(InklingError) as e:
        conn.call("weird")
    assert e.value.data == {}
    with pytest.raises(LLMError) as e:
        conn.call("upstream")
    assert e.value.data == {"kind": "http", "status": 503, "retry_after": 240.0}


def test_oversized_request_fails_locally_and_keeps_the_connection(tmp_path):
    conn = Connection(FakeServer(tmp_path / "s", lambda m, p: p).path)
    with pytest.raises(WireError):
        conn.call("echo", {"s": "x" * MAX_REQUEST_BYTES})
    assert conn.call("echo", {"s": "ok"}) == {"s": "ok"}


def test_runner_execs_code_with_api(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("INKLING_ROBO_TOKEN", "t0k")
    code = textwrap.dedent(
        """
        import os
        import numpy as np
        assert "INKLING_ROBO_TOKEN" not in os.environ
        def helper(p):
            goto_pose(p, [0, 0, 1, 0], z_approach=0.1)
            return float(p.sum())
        pos, quat = get_object_pose("red cube")
        assert isinstance(pos, np.ndarray) and pos.dtype == np.float32
        print(helper(pos), sorted(n for n in globals() if not n.startswith("_")))
        """
    )

    def handle(method, params):
        if method == "runner.hello":
            return {"code": code, "functions": ["get_object_pose", "goto_pose"]}
        return [np.ones(3, dtype=np.float32), np.array([0, 0, 1, 0.0])]

    server = FakeServer(tmp_path / "s", handle)
    assert runner.main(["--socket", server.path]) == 0
    names = ["get_object_pose", "goto_pose", "helper", "np", "os", "pos", "quat"]
    assert capsys.readouterr().out == f"3.0 {names}\n"
    assert server.calls[0] == ("runner.hello", {"token": "t0k"})
    methods = [(m, p.get("name")) for m, p in server.calls]
    assert methods == [("runner.hello", None), ("call", "get_object_pose"), ("call", "goto_pose")]
    goto = server.calls[2][1]
    assert np.array_equal(goto["args"][0], np.ones(3)) and goto["kwargs"] == {"z_approach": 0.1}


def test_runner_takes_no_participant_code(tmp_path):
    (tmp_path / "skills.py").write_text("raise SystemExit('loaded')\n")
    with pytest.raises(SystemExit) as e:
        runner.main(["--socket", str(tmp_path / "s"), "--skills", str(tmp_path / "skills.py")])
    assert e.value.code == 2  # argparse: unrecognised argument


def test_runner_reports_exceptions_like_capx(tmp_path, capsys):
    def handle(method, params):
        return {"code": "x = 1\nraise KeyError('boom')", "functions": []}

    assert runner.main(["--socket", FakeServer(tmp_path / "s", handle).path]) == 1
    err = capsys.readouterr().err
    assert 'File "<string>", line 2' in err and "KeyError: 'boom'" in err


def test_agent_entry_runs_solve(tmp_path, capsys):
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()
    (agent_dir / "helpers.py").write_text("X = 'helper module'\n")
    (agent_dir / "agent.py").write_text(
        textwrap.dedent(
            """
            import sys
            from helpers import X

            def solve(task, robot, llm):
                reply = llm.chat([{"role": "user", "content": "hi"}], temperature=0.5)
                print(task.instruction, X, reply, reply.reasoning, reply.prompt_tokens)
                print(robot.run(reply).ok, dir(robot.api))
                sys.exit(0)
            """
        )
    )

    def handle(method, params):
        if method == "hello":
            return {"instruction": "lift", "api_docs": "docs", "read_functions": ["f"]}
        if method == "llm.chat":
            assert params == {"messages": [{"role": "user", "content": "hi"}],
                              "options": {"temperature": 0.5}}  # fmt: skip
            return {"content": "c", "reasoning": "r", "finish_reason": "stop",
                    "prompt_tokens": 3, "completion_tokens": 4}  # fmt: skip
        if method == "run":
            return {"stdout": "", "stderr": "", "ok": True}

    server = FakeServer(tmp_path / "s", handle)
    assert run.main([str(agent_dir), "--socket", server.path]) == 0
    assert capsys.readouterr().out == "lift helper module c r 3\nTrue ['f']\n"
    sys.modules.pop("agent", None)
    sys.modules.pop("helpers", None)
    sys.path.remove(str(agent_dir.resolve()))


def test_sdk_depends_only_on_stdlib_msgpack_numpy():
    allowed = set(sys.stdlib_module_names) | {"msgpack", "numpy", "__future__"}
    for path in SDK.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                names = [node.module or ""]
            else:
                continue
            for name in names:
                assert name.split(".")[0] in allowed, f"{path.name} imports {name}"
