"""Sim server: owns the environment and is the only thing that touches it.

Two kinds of Unix-socket connection, each served by ``rpc.serve_connection``:

- agent socket (one connection, the agent): ``hello``, ``observe``, ``call`` for the task's
  read-only functions, ``llm.chat``, ``run``.
- runner socket (one per ``run``, one connection, only while that run is live):
  ``runner.hello``, ``observe``, ``call`` for reads and motion. The first request must be
  ``runner.hello`` with the run's random token, which only the runner is given (in its
  environment); anything else aborts the run.

Each socket's path is unlinked as soon as its one connection is accepted.

``run(code)`` is refused unless ``code`` is an exact substring of a model reply from this
episode; otherwise a fresh code runner is started for it and ``{stdout, stderr, ok}`` returned.
Motion is only ever accepted on a live runner socket, and only with finite arguments of sane
size (see ``check_motion_args``). The server never imports agent code.

``close()`` waits for an env call in flight only until ``CLOSE_GRACE_S`` past the episode's
deadline; if one is still running then, the checker is skipped and the episode fails.
"""

from __future__ import annotations

import builtins
import hmac
import json
import math
import secrets
import socket
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from harness.llm_relay import LLMRelay
from harness.rpc import RpcError, check_params, listen_unix, serve_connection
from harness.sandbox import Sandbox
from harness.tasks import TaskSpec

MAX_CODE_CHARS = 256 * 1024
MAX_RETURNED_OUTPUT = 64 * 1024
MAX_ARGS = 16
MAX_LOGGED_RUNNER_OUTPUT = 256 * 1024
LOGGED_ARRAY_VALUES = 8
MAX_LOGGED_FIELD = 2000
MAX_NAME_LEN = 64
MAX_RPCS = 10_000
MAX_MOTION_VALUE = 10.0
MAX_MOTION_VECTOR = 16
CLOSE_GRACE_S = 2.0
MAX_API_ERROR_CHARS = 1000


class EnvAdapter(Protocol):
    """What the sim server needs from an environment (``capx_env.CapxEnv`` or a fake)."""

    instruction: str
    api_docs: str
    functions: dict[str, Callable[..., Any]]

    def observe(self) -> dict[str, Any]: ...

    def task_completed(self) -> bool: ...

    def write_video(self, path: Path) -> bool: ...

    def close(self) -> None: ...


def _truncate(text: str, cap: int = MAX_RETURNED_OUTPUT) -> str:
    if len(text) <= cap:
        return text
    half = cap // 2
    return f"{text[:half]}\n[... {len(text) - cap} characters truncated ...]\n{text[-half:]}"


def jsonable(value: Any) -> Any:
    """Plain-JSON form of RPC values for the transcript (NaN/inf -> str). An array becomes a
    summary (dtype, shape, nbytes and its first few values): arrays from the sandbox are
    untrusted, and ``tolist()`` costs one Python object per element or index."""
    if isinstance(value, np.ndarray):
        return {
            "ndarray": value.dtype.str,
            "shape": list(value.shape),
            "nbytes": value.nbytes,
            "head": jsonable(value.flat[:LOGGED_ARRAY_VALUES].tolist()),
        }
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, bytes):
        return f"<{len(value)} bytes>"
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    return value


def check_motion_args(args: list[Any], kwargs: dict[str, Any]) -> None:
    """Raise ``ValueError`` unless every argument is a finite number of magnitude at most
    ``MAX_MOTION_VALUE`` (metres, radians, quaternion components) or a vector of at most
    ``MAX_MOTION_VECTOR`` of them, and every 4-vector (in every task's motion API, a
    quaternion) can be normalised. NaN or huge targets make CaP-X's IK retry for seconds under
    the env lock, and can make MuJoCo blow up and silently reset the scene."""
    for value in [*args, *kwargs.values()]:
        if not isinstance(value, (int, float, list, tuple, np.ndarray)):
            raise ValueError(f"motion arguments must be numbers or vectors, not {type(value)}")
        try:
            arr = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError):
            raise ValueError("motion arguments must be numbers or vectors of numbers") from None
        if arr.size > MAX_MOTION_VECTOR:
            raise ValueError(f"motion vectors have at most {MAX_MOTION_VECTOR} elements")
        if not np.isfinite(arr).all() or (np.abs(arr) > MAX_MOTION_VALUE).any():
            raise ValueError(f"motion arguments must be finite and within +-{MAX_MOTION_VALUE}")
        if arr.size == 4 and np.linalg.norm(arr) < 1e-6:
            raise ValueError("a quaternion must not be zero")


def _api_error(e: Exception) -> RpcError:
    """What the sandbox learns of an env exception: a builtin type's name and its message, or
    else the message prefixed by the class name; at most ``MAX_API_ERROR_CHARS`` characters."""
    name = type(e).__name__
    builtin = getattr(builtins, name, None) is type(e)
    message = str(e) if builtin else f"{name}: {e}"
    return RpcError("api_error", _truncate(message, MAX_API_ERROR_CHARS), name if builtin else None)


def _clipped(value: Any) -> Any:
    """``jsonable(value)``, or its JSON text cut to ``MAX_LOGGED_FIELD`` characters."""
    plain = jsonable(value)
    text = json.dumps(plain)
    return plain if len(text) <= MAX_LOGGED_FIELD else _truncate(text, MAX_LOGGED_FIELD)


class SimServer:
    def __init__(
        self,
        spec: TaskSpec,
        env: EnvAdapter,
        relay: LLMRelay,
        sandbox: Sandbox,
        socket_dir: Path,
    ) -> None:
        if set(env.functions) != spec.functions:
            raise RuntimeError(
                f"env API {sorted(env.functions)} does not match the task table "
                f"{sorted(spec.functions)}"
            )
        self.spec = spec
        self.env = env
        self.relay = relay
        self.sandbox = sandbox
        self.socket_dir = socket_dir
        self.start = time.monotonic()
        self.events: list[dict[str, Any]] = []
        self.errors: list[dict[str, Any]] = []
        self.runs: list[dict[str, Any]] = []
        self.log_budget = relay.log_budget
        self._run_count = 0
        self._rpcs = 0
        self._env_lock = threading.Lock()  # one env operation at a time
        self._run_lock = threading.Lock()  # one code runner at a time
        self._log_lock = threading.Lock()
        self._closed = False
        self.stalled = False
        """An env call was still running ``CLOSE_GRACE_S`` after the deadline; not checked."""
        self._checked: bool | None = None
        self._active_run: int | None = None
        self._runner_proc: Any = None
        self._aborted: str | None = None  # why the live run was aborted, if it was
        self._listeners: list[socket.socket] = []
        self._conns: list[socket.socket] = []

    # ------------------------------------------------------------ bookkeeping

    @property
    def tokens(self) -> int:
        return self.relay.tokens

    @property
    def calls(self) -> int:
        return self.relay.calls

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.start

    def _t(self) -> float:
        return round(self.elapsed, 3)

    def _event(self, **fields: Any) -> None:
        event = {"t": self._t(), **fields}
        with self._log_lock:
            if self.log_budget.admit("events", event):
                self.events.append(event)

    def _error(self, where: str, message: str) -> None:
        error = {"t": self._t(), "where": where, "message": _truncate(message, MAX_LOGGED_FIELD)}
        with self._log_lock:
            if self.log_budget.admit("errors", error):
                self.errors.append(error)

    def _keep_run(self, record: dict[str, Any]) -> None:
        with self._log_lock:
            if self.log_budget.admit("runs", record):
                self.runs.append(record)

    # ------------------------------------------------------------ sockets

    def _accept_one(
        self, srv: socket.socket, path: Path, handle: Callable[[str, dict], Any], role: str
    ) -> None:
        """Accept exactly one connection on ``srv``, then stop listening and serve it."""
        try:
            conn, _ = srv.accept()
        except OSError:
            return
        finally:
            srv.close()
            path.unlink(missing_ok=True)
        with self._log_lock:
            if self._closed:
                conn.close()
                return
            self._conns.append(conn)
        serve_connection(conn, handle, on_error=lambda m: self._error(role, m))

    def _listen(self, path: Path, handle: Callable[[str, dict], Any], role: str) -> None:
        srv = listen_unix(path)
        self._listeners.append(srv)
        counted = self._counted(handle)
        args = (srv, path, counted, role)
        threading.Thread(target=self._accept_one, args=args, daemon=True).start()

    def _counted(self, handle: Callable[[str, dict], Any]) -> Callable[[str, dict], Any]:
        """``handle``, refusing every request past the episode's ``MAX_RPCS``."""

        def counted(method: str, params: dict[str, Any]) -> Any:
            with self._log_lock:
                self._rpcs += 1
                count = self._rpcs
            if count > MAX_RPCS:
                if count == MAX_RPCS + 1:
                    self._error("rpc", f"request budget ({MAX_RPCS} per episode) spent")
                raise RpcError("budget", f"request budget ({MAX_RPCS} per episode) spent")
            return handle(method, params)

        return counted

    def listen_agent(self, path: Path) -> Path:
        self._listen(path, self._handle_agent, "agent")
        return path

    # ------------------------------------------------------------ handlers

    def _handle_agent(self, method: str, params: dict[str, Any]) -> Any:
        if method == "hello":
            check_params(params, {})
            return {
                "instruction": self.env.instruction,
                "api_docs": self.env.api_docs,
                "read_functions": sorted(self.spec.reads),
            }
        if method == "observe":
            check_params(params, {})
            return self._observe("agent")
        if method == "call":
            return self._call(params, "agent", allowed=self.spec.reads)
        if method == "llm.chat":
            check_params(params, {"messages": list, "options": dict})
            if self._closed:
                raise RpcError("closed", "episode is over")
            # The event's seq names its llm.jsonl entry; a call the relay refused without
            # forwarding it (bad_request, budget) has none. One agent connection: no race.
            forwarded = self.relay.forwarded
            try:
                reply = self.relay.chat(params["messages"], params["options"])
            except RpcError as e:
                seq = forwarded if self.relay.forwarded > forwarded else None
                error = _truncate(e.message, 500)
                self._event(kind="llm", index=None, seq=seq, ok=False, error=error)
                raise
            self._event(kind="llm", index=self.relay.calls - 1, seq=forwarded, ok=True)
            return reply
        if method == "run":
            check_params(params, {"code": str})
            return self._run(params["code"])
        raise RpcError("bad_request", f"unknown method {method!r}")

    def _runner_handler(self, run_index: int, code: str, token: str) -> Callable[[str, dict], Any]:
        hello_done = False

        def handle(method: str, params: dict[str, Any]) -> Any:
            nonlocal hello_done
            if self._active_run != run_index:
                raise RpcError("closed", "this code run is over")
            if not hello_done:
                given = params.get("token") if method == "runner.hello" else None
                if set(params) != {"token"} or not isinstance(given, str):
                    given = ""
                if not hmac.compare_digest(given.encode(), token.encode()):
                    self._abort_run(run_index, "the first request was not a valid runner.hello")
                    raise RpcError("forbidden", "runner.hello with the run's token must come first")
                hello_done = True
                return {"code": code, "functions": sorted(self.spec.functions)}
            if method == "runner.hello":
                raise RpcError("bad_request", "runner.hello was already sent")
            if method == "observe":
                check_params(params, {})
                return self._observe("runner", run_index)
            if method == "call":
                return self._call(params, "runner", self.spec.functions, run_index)
            raise RpcError("bad_request", f"unknown method {method!r}")

        return handle

    def _abort_run(self, run_index: int, reason: str) -> None:
        """End run ``run_index`` now: refuse its calls and kill its runner."""
        with self._env_lock:
            if self._active_run != run_index:
                return
            self._active_run = None
            self._aborted = reason
        self._error(f"runner {run_index}", f"run aborted: {reason}")
        proc = self._runner_proc
        if proc is not None:
            proc.kill()

    def _observe(self, source: str, run: int | None = None) -> dict[str, Any]:
        with self._env_lock:
            if self._closed:
                raise RpcError("closed", "episode is over")
            return self.env.observe()

    def _call(
        self,
        params: dict[str, Any],
        source: str,
        allowed: frozenset[str],
        run: int | None = None,
    ) -> Any:
        check_params(params, {"name": str, "args": list, "kwargs": dict})
        name, args, kwargs = params["name"], params["args"], params["kwargs"]
        if len(args) > MAX_ARGS or len(kwargs) > MAX_ARGS:
            raise RpcError("bad_request", "too many arguments")
        if len(name) > MAX_NAME_LEN or not name.isidentifier():  # checked before it is logged
            raise RpcError("bad_request", f"name must be an identifier of <= {MAX_NAME_LEN} chars")
        if name not in allowed:
            motion = name in self.spec.motions
            self._event(kind="call", source=source, run=run, name=name, ok=False, error="forbidden")
            if motion:
                raise RpcError(
                    "forbidden",
                    f"{name} moves the robot; only model-written code run via "
                    "robot.run() may call it",
                )
            raise RpcError("forbidden", f"{name!r} is not a robot API function of this task")
        with self._env_lock:
            if self._closed:
                raise RpcError("closed", "episode is over")
            if source == "runner" and self._active_run != run:
                raise RpcError("closed", "this code run is over")
            try:
                if name in self.spec.motions:
                    check_motion_args(args, kwargs)
                result = self.env.functions[name](*args, **kwargs)
            except Exception as e:
                self._event(
                    kind="call",
                    source=source,
                    run=run,
                    name=name,
                    args=_clipped(args),
                    kwargs=_clipped(kwargs),
                    ok=False,
                    error=_truncate(f"{type(e).__name__}: {e}", MAX_LOGGED_FIELD),
                )
                raise _api_error(e) from None
        self._event(
            kind="call",
            source=source,
            run=run,
            name=name,
            args=_clipped(args),
            kwargs=_clipped(kwargs),
            ok=True,
            result=_clipped(result),
        )
        return result

    def _run(self, code: str) -> dict[str, Any]:
        if len(code) > MAX_CODE_CHARS:
            raise RpcError("bad_request", "code too long")
        with self._run_lock:
            index = self._run_count
            self._run_count += 1
            record: dict[str, Any] = {"index": index, "code": code, "accepted": False}
            if self._closed:
                record["refused"] = "episode is over"
                self._keep_run(record)
                raise RpcError("closed", "episode is over")
            if not self.relay.is_substring_of_reply(code):
                record["refused"] = "not a substring of a model reply"
                self._keep_run(record)
                self._event(kind="run", index=index, accepted=False, refused=record["refused"])
                raise RpcError(
                    "refused",
                    "code must be a non-blank exact substring of a model reply from this episode",
                )
            record["accepted"] = True
            self._event(kind="run", index=index, accepted=True, phase="start")
            started = time.monotonic()
            path = self.socket_dir / f"r{index}.sock"
            token = secrets.token_hex(16)  # 128 bits
            self._active_run = index
            self._aborted = None
            proc = None
            try:
                self._listen(path, self._runner_handler(index, code, token), f"runner {index}")
                proc = self.sandbox.start_runner(path, token)
                self._runner_proc = proc
                # close() or an abort raced with the start and may not have seen proc
                if self._closed or self._active_run != index:
                    proc.kill()
                remaining = self.relay.deadline - time.monotonic()
                status = proc.wait(timeout=max(remaining, 0.0))
            finally:
                with self._env_lock:  # no motion call is half-done once we hold the lock
                    self._active_run = None
                if proc is not None:
                    proc.kill()
                self._runner_proc = None
            stdout, stderr = proc.stdout(), proc.stderr()
            timed_out = status is None or self._closed
            if timed_out:
                stderr += "\n[killed: the episode ended (time limit or agent exit)]\n"
            if self._aborted:
                stderr += f"\n[killed: {self._aborted}]\n"
            ok = status == 0 and not self._aborted
            record.update(
                ok=ok,
                exit_status=status,
                timed_out=timed_out,
                aborted=self._aborted,
                seconds=round(time.monotonic() - started, 3),
                stdout=_truncate(stdout, MAX_LOGGED_RUNNER_OUTPUT),
                stderr=_truncate(stderr, MAX_LOGGED_RUNNER_OUTPUT),
            )
            self._keep_run(record)
            self._event(kind="run", index=index, accepted=True, phase="end", ok=ok)
            return {"stdout": _truncate(stdout), "stderr": _truncate(stderr), "ok": ok}

    # ------------------------------------------------------------ end of episode

    def close(self) -> None:
        """Stop serving: refuse every further call, kill any live runner, close sockets."""
        with self._log_lock:
            self._closed = True
        proc = self._runner_proc
        if proc is not None:
            proc.kill()
        for s in self._listeners + self._conns:
            try:
                s.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            s.close()
        # Wait for an env call in flight, but not past the deadline (plus a little).
        wait = self.relay.deadline + CLOSE_GRACE_S - time.monotonic()
        if self._env_lock.acquire(timeout=max(wait, 0.0)):
            self._active_run = None
            self._env_lock.release()
        else:
            self.stalled = True
            self._active_run = None
            self._error("env", "an env call was still running after the deadline; not checked")

    def task_completed(self) -> bool:
        """Run the task checker. Allowed once, after ``close()``."""
        if not self._closed:
            raise RuntimeError("task_completed() before close()")
        if self._checked is not None:
            raise RuntimeError("task_completed() called twice")
        if self.stalled:
            self._checked = False
            return False
        with self._env_lock:
            self._checked = bool(self.env.task_completed())
        return self._checked

    def save(self, out: Path) -> None:
        """Write transcript.jsonl, llm.jsonl, code/run_NNN.py, runs.json and errors.json.

        All of them together are bounded by the episode's ``LogBudget``; what did not fit is
        counted in a last errors.json entry."""
        code_dir = out / "code"
        code_dir.mkdir(parents=True, exist_ok=True)
        with open(out / "transcript.jsonl", "w") as f:
            for event in self.events:
                f.write(json.dumps(event) + "\n")
        with open(out / "llm.jsonl", "w") as f:
            for entry in self.relay.log:
                f.write(json.dumps(jsonable(entry)) + "\n")
        runs = []
        for run in self.runs:
            name = f"run_{run['index']:03d}.py"
            (code_dir / name).write_text(run["code"])
            runs.append(
                {k: v for k, v in run.items() if k != "code"} | {"code_file": f"code/{name}"}
            )
        (out / "runs.json").write_text(json.dumps(runs, indent=1))
        errors = list(self.errors)
        if self.log_budget.dropped:
            counts = ", ".join(f"{n} {kind}" for kind, n in self.log_budget.dropped.items())
            message = f"log budget ({self.log_budget.limit} bytes) spent; not kept: {counts}"
            errors.append({"t": self._t(), "where": "log", "message": message})
        (out / "errors.json").write_text(json.dumps(errors, indent=1))
