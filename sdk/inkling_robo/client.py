"""Thin RPC clients for the participant interface: ``solve(task, robot, llm)``."""

from __future__ import annotations

import builtins
import itertools
import socket
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, NamedTuple

from .wire import MAX_REQUEST_BYTES, recv_message, send_message

# Exceptions from the robot API that the runner re-raises as the same builtin type, so
# model-written code sees what it would have seen in CaP-X.
_BUILTIN_ERRORS = ("ValueError", "TypeError", "RuntimeError", "KeyError", "IndexError")


def line_buffered_output() -> None:
    """Line-buffer stdout and stderr, for the entry points. Sandboxes start Python without -u:
    an unbuffered stream drops the rest of a short write, and gVisor's pipes take at most
    64 KiB per write; the buffered layer retries until everything is written."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(line_buffering=True)


class InklingError(Exception):
    """An error reported by the harness. ``code`` says which kind."""

    def __init__(self, code: str, message: str, data: dict[str, Any] | None = None) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message
        self.data: dict[str, Any] = data if isinstance(data, dict) else {}
        """Details as data. For ``LLMError``: ``kind`` ("http", "unreachable", "timeout",
        "malformed"), ``status`` (HTTP status or None) and ``retry_after`` (seconds or None)."""


class Forbidden(InklingError):
    """The call is not allowed on this connection (e.g. motion from the agent)."""


class RunRefused(InklingError):
    """``robot.run(code)``: the code is not an exact substring of a model reply."""


class BudgetExceeded(InklingError):
    """The episode's model-call, token or time budget is spent."""


class LLMError(InklingError):
    """The model server failed or returned something unusable."""


class RobotApiError(InklingError):
    """A robot API function raised an exception."""


_ERRORS: dict[str, type[InklingError]] = {
    "forbidden": Forbidden,
    "refused": RunRefused,
    "budget": BudgetExceeded,
    "upstream": LLMError,
    "api_error": RobotApiError,
}


class Connection:
    """One Unix-socket connection to the sim server; calls are serialised."""

    def __init__(self, path: str) -> None:
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock.connect(path)
        self._lock = threading.Lock()
        self._ids = itertools.count(1)

    def call(self, method: str, params: dict[str, Any] | None = None) -> Any:
        with self._lock:
            req_id = next(self._ids)
            request = {"id": req_id, "method": method, "params": params or {}}
            # Over the limit, WireError is raised here and nothing is sent.
            send_message(self._sock, request, MAX_REQUEST_BYTES)
            resp = recv_message(self._sock)
        if not isinstance(resp, dict) or resp.get("id") not in (req_id, None):
            raise InklingError("protocol", "unexpected response")
        if "error" in resp:
            err = resp["error"]
            code, message = str(err.get("code")), str(err.get("message"))
            if code == "api_error" and err.get("type") in _BUILTIN_ERRORS:
                raise getattr(builtins, err["type"])(message)
            raise _ERRORS.get(code, InklingError)(code, message, err.get("data"))
        return resp.get("result")

    def close(self) -> None:
        self._sock.close()


def _tuple_result(value: Any) -> Any:
    # CaP-X API functions return tuples; msgpack has only lists.
    return tuple(value) if isinstance(value, list) else value


def api_function(conn: Connection, name: str) -> Callable[..., Any]:
    """A local callable that forwards to robot API function ``name``."""

    def fn(*args: Any, **kwargs: Any) -> Any:
        return _tuple_result(conn.call("call", {"name": name, "args": args, "kwargs": kwargs}))

    fn.__name__ = fn.__qualname__ = name
    return fn


@dataclass(frozen=True)
class Task:
    instruction: str
    """The task prompt given to the model."""
    api_docs: str
    """Signatures and docstrings of the robot API, as text."""


class RunResult(NamedTuple):
    stdout: str
    stderr: str
    ok: bool


class Reply(str):
    """The text content of a model reply. Extra fields ride along as attributes."""

    reasoning: str | None
    finish_reason: str | None
    prompt_tokens: int
    completion_tokens: int


class _Api:
    def __init__(self, conn: Connection, names: list[str]) -> None:
        self._conn = conn
        self._names = names

    def __getattr__(self, name: str) -> Callable[..., Any]:
        if name.startswith("_"):
            raise AttributeError(name)
        return api_function(self._conn, name)

    def __dir__(self) -> list[str]:
        return list(self._names)


class Robot:
    def __init__(self, conn: Connection, read_functions: list[str]) -> None:
        self._conn = conn
        self.api = _Api(conn, read_functions)
        """Read-only robot API functions, e.g. ``robot.api.get_object_pose("red cube")``."""

    def observe(self) -> dict[str, Any]:
        """Gripper and joint state (no images)."""
        return self._conn.call("observe")

    def run(self, code: str) -> RunResult:
        """Run model-written ``code`` in a fresh code runner with the motion API in scope.

        ``code`` must be an exact substring of a model reply from this episode, or
        ``RunRefused`` is raised.
        """
        r = self._conn.call("run", {"code": code})
        return RunResult(r["stdout"], r["stderr"], r["ok"])


class LLM:
    def __init__(self, conn: Connection) -> None:
        self._conn = conn

    def chat(self, messages: list[dict[str, Any]], **options: Any) -> Reply:
        """OpenAI-style chat completion through the episode's relay.

        ``options`` may be temperature, top_p, max_tokens, stop, presence_penalty and
        frequency_penalty. Raises ``BudgetExceeded`` once the episode budget is spent.
        """
        r = self._conn.call("llm.chat", {"messages": messages, "options": options})
        reply = Reply(r["content"])
        reply.reasoning = r["reasoning"]
        reply.finish_reason = r["finish_reason"]
        reply.prompt_tokens = r["prompt_tokens"]
        reply.completion_tokens = r["completion_tokens"]
        return reply
