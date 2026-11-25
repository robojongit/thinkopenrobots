import pickle
import socket
import struct
import threading

import msgpack
import msgpack.fallback
import numpy as np
import pytest

from harness import rpc
from harness.rpc import RpcError, check_params, recv_message, send_message, serve_connection


def handler(method, params):
    if method == "echo":
        return params
    if method == "fail":
        raise RpcError("api_error", "nope", "ValueError")
    if method == "crash":
        raise ZeroDivisionError("bug")
    if method == "unencodable":
        return object()
    raise RpcError("bad_request", "unknown method")


@pytest.fixture
def conn():
    server_sock, client = socket.socketpair()
    errors: list[str] = []
    t = threading.Thread(target=serve_connection, args=(server_sock, handler, errors.append))
    t.start()
    client.settimeout(5)
    yield client, errors, t
    client.close()
    t.join(timeout=5)
    assert not t.is_alive()


def call(client, msg):
    send_message(client, msg)
    return recv_message(client)


def is_closed(client) -> bool:
    try:
        return client.recv(1) == b""
    except OSError:
        return True


def test_request_response_with_arrays(conn):
    client, _, _ = conn
    arr = np.arange(6, dtype=np.float32).reshape(2, 3)
    resp = call(client, {"id": 1, "method": "echo", "params": {"a": arr}})
    assert resp["id"] == 1 and np.array_equal(resp["result"]["a"], arr)


def test_errors_keep_the_connection_open(conn):
    client, errors, _ = conn
    resp = call(client, {"id": 2, "method": "fail", "params": {}})
    assert resp["error"] == {"code": "api_error", "message": "nope", "type": "ValueError"}
    resp = call(client, {"id": 3, "method": "crash", "params": {}})
    assert resp["error"]["code"] == "internal" and "bug" not in resp["error"]["message"]
    assert any("ZeroDivisionError" in e for e in errors)
    resp = call(client, {"id": 4, "method": "unencodable", "params": {}})
    assert resp == {"id": 4, "error": {"code": "internal", "message": "cannot encode object"}}
    assert call(client, {"id": 5, "method": "echo", "params": {}})["result"] == {}


@pytest.mark.parametrize(
    "msg",
    [
        [1, 2, 3],
        {"id": 1, "method": "echo"},
        {"id": 1, "method": "echo", "params": {}, "extra": 1},
        {"id": -1, "method": "echo", "params": {}},
        {"id": True, "method": "echo", "params": {}},
        {"id": 1.0, "method": "echo", "params": {}},
        {"id": 1, "method": "", "params": {}},
        {"id": 1, "method": "x" * 65, "params": {}},
        {"id": 1, "method": 5, "params": {}},
        {"id": 1, "method": "echo", "params": []},
    ],
)
def test_bad_envelopes_get_bad_request(conn, msg):
    client, _, _ = conn
    resp = call(client, msg)
    assert resp["error"]["code"] == "bad_request"
    assert call(client, {"id": 9, "method": "echo", "params": {}})["id"] == 9


@pytest.mark.parametrize("length", [rpc.MAX_REQUEST_BYTES + 1, rpc.MAX_MESSAGE_BYTES + 1])
def test_oversized_length_prefix_closes_the_connection(conn, length):
    client, errors, _ = conn
    client.sendall(struct.pack(">I", length))
    resp = recv_message(client)
    assert resp["error"]["code"] == "bad_frame"
    assert is_closed(client)
    assert errors


@pytest.mark.parametrize(
    "raw",
    [
        b"\xc1",  # never-used msgpack byte
        msgpack.packb(msgpack.ExtType(5, b"x")),
        msgpack.packb({"id": 1, "method": "echo", "params": {"a": {"__ndarray__": True}}}),
        msgpack.packb({1: 2}),
    ],
)
def test_malformed_frames_close_the_connection(conn, raw):
    client, _, _ = conn
    client.sendall(struct.pack(">I", len(raw)) + raw)
    assert recv_message(client)["error"]["code"] == "bad_frame"
    assert is_closed(client)


def test_truncated_frame_and_disconnect_end_quietly(conn):
    client, errors, t = conn
    client.sendall(struct.pack(">I", 100) + b"\x80")
    client.close()
    t.join(timeout=5)
    assert not t.is_alive() and errors == []


def test_check_params():
    check_params({"a": "x", "b": []}, {"a": str, "b": list})
    for bad in ({"a": "x"}, {"a": "x", "b": [], "c": 1}, {"a": 1, "b": []}):
        with pytest.raises(RpcError):
            check_params(bad, {"a": str, "b": list})


def test_listen_unix_is_owner_only(tmp_path):
    srv = rpc.listen_unix(tmp_path / "s.sock")
    try:
        assert (tmp_path / "s.sock").stat().st_mode & 0o777 == 0o600
    finally:
        srv.close()


def _unpickled() -> None:
    _UNPICKLED.append(True)


_UNPICKLED: list[bool] = []


class _Payload:
    def __reduce__(self):
        return (_unpickled, ())


def test_decoder_is_immune_to_msgpack_numpy_patch(monkeypatch):
    # CaP-X calls msgpack_numpy.patch() on import, in the harness process. The patched unpackb
    # unpickles {b"nd": True, b"kind": b"O", b"data": ...}: the decoder must not use it.
    m = pytest.importorskip("msgpack_numpy")
    for name in ("Packer", "Unpacker", "packb", "unpackb"):
        monkeypatch.setattr(msgpack, name, getattr(m, name))
    evil = {
        b"nd": True,
        b"kind": b"O",
        b"type": "|O",
        b"shape": [],
        b"data": pickle.dumps(_Payload()),
    }
    raw = msgpack.fallback.Packer(use_bin_type=True).pack(
        {"id": 1, "method": "echo", "params": {"x": evil}}
    )
    with pytest.raises(rpc.WireError):
        rpc.decode_body(raw)
    assert _UNPICKLED == []
    assert rpc.decode_body(rpc.encode({"a": np.arange(3)})[4:])["a"].tolist() == [0, 1, 2]
