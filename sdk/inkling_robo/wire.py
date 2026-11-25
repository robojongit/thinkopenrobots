"""Wire format shared by the SDK and the trusted harness.

Frames are a 4-byte big-endian length followed by one plain msgpack object. No pickle and no
msgpack ext types. A NumPy array travels as a map ``{"__ndarray__": True, "dtype": str,
"shape": [int, ...], "data": bytes}`` with a whitelisted little-endian dtype. Decoding enforces
hard limits on message size, string and bytes length, container length, nesting depth, the
number of values in a message and array size, and raises ``WireError`` for anything else. Frames
the sandbox sends to the harness have a smaller size cap (``MAX_REQUEST_BYTES``) than the ones
the harness sends back (``MAX_MESSAGE_BYTES``). An array's size is bounded counting every zero
dimension as 1, so a zero-size array cannot carry a huge shape. Depends only on the stdlib,
msgpack and numpy.
"""

from __future__ import annotations

import math
import socket
import struct
from typing import Any

import numpy as np

# msgpack's own implementation, bound here because the module-level functions can be replaced:
# msgpack_numpy.patch(), which CaP-X calls on import (so in the harness process), swaps
# msgpack.unpackb for one whose object hook unpickles maps like {b"nd": True, b"kind": b"O"}.
try:
    from msgpack._cmsgpack import Packer, unpackb
except ImportError:  # msgpack without its C extension
    from msgpack.fallback import Packer, unpackb

MAX_MESSAGE_BYTES = 8 * 1024 * 1024
MAX_REQUEST_BYTES = 1024 * 1024
MAX_NODES = 200_000
MAX_STR_LEN = 1024 * 1024
MAX_BIN_LEN = 4 * 1024 * 1024
MAX_CONTAINER_LEN = 100_000
MAX_DEPTH = 32
MAX_ARRAY_BYTES = 4 * 1024 * 1024
MAX_NDIM = 8

ND_KEY = "__ndarray__"
DTYPES = frozenset(["|b1", "|i1", "|u1", "<i2", "<u2", "<i4", "<u4", "<i8", "<u8", "<f4", "<f8"])

_HEADER = struct.Struct(">I")


class WireError(Exception):
    """Malformed, oversized or unsupported data on the wire."""


# ---------------------------------------------------------------- encoding


def _to_plain(obj: Any, depth: int = 0) -> Any:
    if depth > MAX_DEPTH:
        raise WireError("nesting too deep")
    if obj is None or isinstance(obj, (bool, str, bytes)):
        return obj
    if isinstance(obj, int):
        return obj
    if isinstance(obj, float):
        return obj
    if isinstance(obj, np.ndarray):
        return _encode_array(obj)
    if isinstance(obj, np.generic):
        return _to_plain(obj.item(), depth)
    if isinstance(obj, (list, tuple)):
        return [_to_plain(x, depth + 1) for x in obj]
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if not isinstance(k, str):
                raise WireError(f"map keys must be str, got {type(k).__name__}")
            out[k] = _to_plain(v, depth + 1)
        return out
    raise WireError(f"cannot encode {type(obj).__name__}")


def _shape_bytes(shape: tuple[int, ...] | list[int], itemsize: int) -> int:
    """Bytes the shape would take if every zero dimension were 1: what bounds the work of
    walking the array (``tolist()`` builds one list per index even when there is no data)."""
    return math.prod(max(d, 1) for d in shape) * itemsize


def _encode_array(arr: np.ndarray) -> dict[str, Any]:
    if arr.dtype.byteorder == ">":
        arr = arr.astype(arr.dtype.newbyteorder("<"))
    arr = np.ascontiguousarray(arr)
    dtype = arr.dtype.str
    if dtype not in DTYPES:
        raise WireError(f"array dtype {arr.dtype} not allowed")
    if arr.ndim > MAX_NDIM or _shape_bytes(arr.shape, arr.itemsize) > MAX_ARRAY_BYTES:
        raise WireError("array too large")
    return {ND_KEY: True, "dtype": dtype, "shape": list(arr.shape), "data": arr.tobytes()}


def encode(obj: Any, max_bytes: int = MAX_MESSAGE_BYTES) -> bytes:
    """Encode ``obj`` as one frame (length prefix + msgpack body)."""
    try:
        body = Packer(use_bin_type=True).pack(_to_plain(obj))
    except (OverflowError, ValueError, TypeError) as e:
        raise WireError(f"cannot encode: {e}") from None
    if len(body) > max_bytes:
        raise WireError(f"message of {len(body)} bytes exceeds the {max_bytes}-byte limit")
    return _HEADER.pack(len(body)) + body


# ---------------------------------------------------------------- decoding


def _reject_ext(code: int, data: bytes) -> Any:
    raise WireError("msgpack ext types are not allowed")


def _decode_array(m: dict[str, Any]) -> np.ndarray:
    if set(m) != {ND_KEY, "dtype", "shape", "data"} or m[ND_KEY] is not True:
        raise WireError("bad ndarray encoding")
    dtype, shape, data = m["dtype"], m["shape"], m["data"]
    if not isinstance(dtype, str) or dtype not in DTYPES:
        raise WireError("array dtype not allowed")
    if not isinstance(shape, list) or len(shape) > MAX_NDIM:
        raise WireError("bad array shape")
    if not all(type(d) is int and 0 <= d <= MAX_ARRAY_BYTES for d in shape):
        raise WireError("bad array shape")
    if not isinstance(data, bytes):
        raise WireError("array data must be bytes")
    dt = np.dtype(dtype)
    if _shape_bytes(shape, dt.itemsize) > MAX_ARRAY_BYTES:
        raise WireError("array too large")
    if math.prod(shape) * dt.itemsize != len(data):
        raise WireError("array data size does not match shape")
    return np.frombuffer(data, dtype=dt).reshape(shape).copy()


def _from_plain(obj: Any, depth: int = 0) -> Any:
    if depth > MAX_DEPTH:
        raise WireError("nesting too deep")
    if obj is None or isinstance(obj, (bool, int, float, str, bytes)):
        return obj
    if isinstance(obj, list):
        return [_from_plain(x, depth + 1) for x in obj]
    if isinstance(obj, dict):
        if ND_KEY in obj:
            return _decode_array(obj)
        out = {}
        for k, v in obj.items():
            if not isinstance(k, str):
                raise WireError("map keys must be str")
            out[k] = _from_plain(v, depth + 1)
        return out
    raise WireError(f"unexpected type {type(obj).__name__}")


def decode_body(body: bytes, max_bytes: int = MAX_MESSAGE_BYTES) -> Any:
    """Decode one msgpack body with every limit enforced. Raises ``WireError``."""
    if len(body) > max_bytes:
        raise WireError("message too large")
    # msgpack builds every container before we see it; the hooks count the values as they are
    # built and stop a message of many tiny containers early (one byte each on the wire).
    nodes = 1

    def count(n: int) -> None:
        nonlocal nodes
        nodes += n
        if nodes > MAX_NODES:
            raise WireError(f"more than {MAX_NODES} values in one message")

    def list_hook(items: list) -> list:
        count(len(items))
        return items

    def pairs_hook(pairs: list) -> dict:
        count(len(pairs))
        return dict(pairs)

    try:
        raw = unpackb(
            body,
            raw=False,
            use_list=True,
            strict_map_key=True,
            list_hook=list_hook,
            object_pairs_hook=pairs_hook,
            ext_hook=_reject_ext,
            max_str_len=MAX_STR_LEN,
            max_bin_len=MAX_BIN_LEN,
            max_array_len=MAX_CONTAINER_LEN,
            max_map_len=MAX_CONTAINER_LEN,
            max_ext_len=0,
        )
    except WireError:
        raise
    except Exception as e:  # msgpack raises many types for bad input
        raise WireError(f"undecodable message: {type(e).__name__}") from None
    return _from_plain(raw)


# ---------------------------------------------------------------- sockets


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(min(n - len(buf), 1 << 20))
        if not chunk:
            raise EOFError("connection closed")
        buf += chunk
    return bytes(buf)


def recv_message(sock: socket.socket, max_bytes: int = MAX_MESSAGE_BYTES) -> Any:
    """Read one frame. Raises ``EOFError`` on a clean close, ``WireError`` on bad data."""
    (length,) = _HEADER.unpack(_recv_exact(sock, _HEADER.size))
    if length > max_bytes:
        raise WireError(f"message of {length} bytes exceeds the {max_bytes}-byte limit")
    return decode_body(_recv_exact(sock, length), max_bytes)


def send_message(sock: socket.socket, obj: Any, max_bytes: int = MAX_MESSAGE_BYTES) -> None:
    sock.sendall(encode(obj, max_bytes))
