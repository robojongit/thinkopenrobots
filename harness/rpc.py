"""Trusted side of the RPC: length-prefixed plain msgpack over Unix sockets.

The codec and its hard limits live in ``inkling_robo.wire`` because the SDK, which runs inside
the sandbox, must speak the same format; this module re-exports it and adds the server loop and
strict request validation. Every message from the sandbox is untrusted:

- a bad frame (over ``MAX_REQUEST_BYTES``, undecodable body, ext type, bad array, too deep, too
  long, too many values) gets an error reply and the connection is closed;
- a well-formed frame with a bad envelope or bad params gets an error reply;
- an exception inside a handler becomes an ``internal`` error reply.

Nothing a client sends can raise out of ``serve_connection``.
"""

from __future__ import annotations

import logging
import os
import socket
from collections.abc import Callable
from pathlib import Path
from typing import Any

from inkling_robo.wire import (  # noqa: F401  (re-exported)
    DTYPES,
    MAX_ARRAY_BYTES,
    MAX_CONTAINER_LEN,
    MAX_DEPTH,
    MAX_MESSAGE_BYTES,
    MAX_REQUEST_BYTES,
    MAX_STR_LEN,
    WireError,
    decode_body,
    encode,
    recv_message,
    send_message,
)

log = logging.getLogger(__name__)

MAX_METHOD_LEN = 64


class RpcError(Exception):
    """Raised by handlers; sent to the client as ``{"code", "message"[, "type"]}``."""

    def __init__(
        self,
        code: str,
        message: str,
        type_name: str | None = None,
        data: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message
        self.type_name = type_name
        self.data = data
        """Machine-readable details, e.g. an upstream HTTP status and Retry-After."""

    def to_wire(self) -> dict[str, Any]:
        err: dict[str, Any] = {"code": self.code, "message": self.message[:4000]}
        if self.type_name:
            err["type"] = self.type_name
        if self.data:
            err["data"] = self.data
        return err


Handler = Callable[[str, dict[str, Any]], Any]


def check_params(params: dict[str, Any], schema: dict[str, type]) -> None:
    """Require exactly the keys in ``schema`` with values of the given types."""
    if set(params) != set(schema):
        raise RpcError("bad_request", f"expected params {sorted(schema)}, got {sorted(params)}")
    for key, typ in schema.items():
        if not isinstance(params[key], typ):
            raise RpcError("bad_request", f"param {key!r} must be {typ.__name__}")


def _parse_request(msg: Any) -> tuple[int, str, dict[str, Any]]:
    if not isinstance(msg, dict) or set(msg) != {"id", "method", "params"}:
        raise RpcError("bad_request", "request must be {id, method, params}")
    req_id, method, params = msg["id"], msg["method"], msg["params"]
    if type(req_id) is not int or not 0 <= req_id < 2**53:
        raise RpcError("bad_request", "bad id")
    if not isinstance(method, str) or not 0 < len(method) <= MAX_METHOD_LEN:
        raise RpcError("bad_request", "bad method")
    if not isinstance(params, dict):
        raise RpcError("bad_request", "params must be a map")
    return req_id, method, params


def _send_quietly(sock: socket.socket, obj: Any) -> bool:
    try:
        send_message(sock, obj)
        return True
    except WireError as e:  # result not encodable: tell the client instead
        try:
            send_message(
                sock, {"id": obj.get("id"), "error": {"code": "internal", "message": str(e)}}
            )
            return True
        except OSError:
            return False
    except OSError:
        return False


def serve_connection(
    sock: socket.socket, handle: Handler, on_error: Callable[[str], None] | None = None
) -> None:
    """Serve requests on ``sock`` until the peer closes or sends a bad frame. Never raises."""
    report = on_error or (lambda m: log.warning("rpc: %s", m))
    try:
        while True:
            try:
                msg = recv_message(sock, MAX_REQUEST_BYTES)
            except EOFError:
                return
            except WireError as e:
                report(f"bad frame, closing connection: {e}")
                _send_quietly(sock, {"id": None, "error": {"code": "bad_frame", "message": str(e)}})
                return
            except OSError:
                return
            req_id = None
            try:
                req_id, method, params = _parse_request(msg)
                reply: dict[str, Any] = {"id": req_id, "result": handle(method, params)}
            except RpcError as e:
                reply = {"id": req_id, "error": e.to_wire()}
            except Exception as e:  # handler bug: report, keep serving
                report(f"internal error in {msg.get('method')!r}: {type(e).__name__}: {e}")
                reply = {"id": req_id, "error": {"code": "internal", "message": "internal error"}}
            if not _send_quietly(sock, reply):
                return
    except Exception as e:  # last line of defence
        report(f"connection handler crashed: {type(e).__name__}: {e}")
    finally:
        try:
            sock.close()
        except OSError:
            pass


def listen_unix(path: Path) -> socket.socket:
    """Bind a listening Unix socket at ``path`` (owner-only permissions)."""
    if path.exists():
        path.unlink()
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(str(path))
    os.chmod(path, 0o600)
    srv.listen(1)
    return srv
