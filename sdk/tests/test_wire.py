import random
import struct

import msgpack
import numpy as np
import pytest
from inkling_robo import wire
from inkling_robo.wire import WireError, decode_body, encode


def roundtrip(obj):
    frame = encode(obj)
    (n,) = struct.unpack(">I", frame[:4])
    assert n == len(frame) - 4
    return decode_body(frame[4:])


def body(obj) -> bytes:
    return msgpack.packb(obj, use_bin_type=True)


def nd(dtype="<f8", shape=(2,), data=None):
    if data is None:
        data = b"\0" * (np.dtype(dtype).itemsize * int(np.prod(shape)))
    return {"__ndarray__": True, "dtype": dtype, "shape": list(shape), "data": data}


def test_roundtrip_plain_values():
    obj = {"a": [1, -2, 3.5, None, True, "x", b"\x00\x01"], "b": {"c": []}}
    assert roundtrip(obj) == obj


def test_tuples_and_numpy_scalars_become_plain():
    assert roundtrip((1, np.float32(2.5), np.int64(3), np.bool_(True))) == [1, 2.5, 3, True]


@pytest.mark.parametrize("dtype", sorted(wire.DTYPES))
def test_array_roundtrip_every_allowed_dtype(dtype):
    arr = (np.arange(12) % 2).astype(dtype).reshape(3, 4)
    out = roundtrip({"x": arr})["x"]
    assert out.dtype == arr.dtype and out.shape == (3, 4) and np.array_equal(out, arr)
    out[0, 0] = 1  # decoded arrays are writable copies


def test_big_endian_and_non_contiguous_arrays_are_normalised():
    arr = np.arange(6, dtype=">f8").reshape(2, 3).T
    out = roundtrip(arr)
    assert out.dtype == np.dtype("<f8") and np.array_equal(out, arr)


@pytest.mark.parametrize(
    "arr",
    [np.array([object()]), np.array(["a"]), np.array([1 + 2j]), np.zeros(2, dtype=np.float16)],
)
def test_encode_rejects_disallowed_dtypes(arr):
    with pytest.raises(WireError):
        encode(arr)


@pytest.mark.parametrize("obj", [object(), {1: 2}, {1.5}, 2**70, lambda: 0])
def test_encode_rejects_unsupported_values(obj):
    with pytest.raises(WireError):
        encode(obj)


def test_encode_rejects_oversized_array():
    with pytest.raises(WireError):
        encode(np.zeros(wire.MAX_ARRAY_BYTES // 8 + 1))
    with pytest.raises(WireError):
        encode(np.zeros((2**23, 0), dtype=np.uint8))


def test_small_zero_size_arrays_roundtrip():
    for shape in [(0,), (3, 0), (0, 5)]:
        out = roundtrip(np.zeros(shape))
        assert out.shape == shape


@pytest.mark.parametrize(
    "bad",
    [
        nd(dtype="|O", data=b"\0" * 16),
        nd(dtype="<c16"),
        nd(dtype=">f8"),
        nd(dtype="<f2"),
        nd(dtype=7, data=b"\0" * 16),
        nd(shape=(3,), data=b"\0" * 16),  # data size does not match shape
        nd(shape=(-1,), data=b""),
        nd(shape=(1.5,), data=b""),
        nd(shape=(True,), data=b"\0"),
        nd(shape=(1,) * 9, data=b"\0" * 8),
        {**nd(), "extra": 1},
        {**nd(), "__ndarray__": 1},
        {"__ndarray__": True, "dtype": "<f8", "shape": [1]},
        nd(data="not bytes"),
        nd(shape=(2**31, 2**31), data=b""),
        # Zero-size, but a huge shape: tolist() would build 8M empty lists.
        nd(dtype="|u1", shape=(2**23, 0), data=b""),
        nd(dtype="|u1", shape=(2**21, 4, 0), data=b""),
    ],
)
def test_decode_rejects_bad_ndarray(bad):
    with pytest.raises(WireError):
        decode_body(body({"x": bad}))


def test_decode_rejects_ext_types():
    with pytest.raises(WireError):
        decode_body(msgpack.packb(msgpack.ExtType(1, b"payload")))
    with pytest.raises(WireError):
        decode_body(msgpack.packb([1, msgpack.ExtType(42, b"")]))


def test_decode_rejects_non_string_keys():
    with pytest.raises(WireError):
        decode_body(body({1: "x"}))
    with pytest.raises(WireError):
        decode_body(body({b"k": "x"}))


def test_decode_limits_depth():
    ok = []
    for _ in range(wire.MAX_DEPTH):
        ok = [ok]
    decode_body(body(ok))
    with pytest.raises(WireError):
        decode_body(body([ok]))
    deep = b"\x91" * 5000 + b"\xc0"  # 5000 nested one-element arrays, raw
    with pytest.raises(WireError):
        decode_body(deep)


def test_decode_limits_lengths():
    with pytest.raises(WireError):
        decode_body(body("x" * (wire.MAX_STR_LEN + 1)))
    with pytest.raises(WireError):
        decode_body(body(b"x" * (wire.MAX_BIN_LEN + 1)))
    with pytest.raises(WireError):
        decode_body(body([0] * (wire.MAX_CONTAINER_LEN + 1)))
    # A header claiming a huge array with no data behind it.
    with pytest.raises(WireError):
        decode_body(b"\xdd\xff\xff\xff\xff")


def test_decode_limits_the_number_of_values():
    decode_body(body([0] * wire.MAX_CONTAINER_LEN))
    # Empty maps are one byte each: 300k of them would cost ~20 MB and seconds to build.
    with pytest.raises(WireError, match="values"):
        decode_body(body([[{}] * wire.MAX_CONTAINER_LEN] * 3))
    # The review's worst case, cut to the request size: arrays of 65535 empty maps.
    inner = b"\xdc" + struct.pack(">H", 65535) + b"\x80" * 65535
    n = (wire.MAX_REQUEST_BYTES - 16) // len(inner)
    with pytest.raises(WireError):
        decode_body(b"\xdc" + struct.pack(">H", n) + inner * n, wire.MAX_REQUEST_BYTES)


def test_requests_have_a_smaller_size_cap():
    big = body(["x" * (wire.MAX_REQUEST_BYTES // 2)] * 3)
    decode_body(big)
    with pytest.raises(WireError):
        decode_body(big, wire.MAX_REQUEST_BYTES)
    with pytest.raises(WireError):
        encode({"s": "x" * wire.MAX_REQUEST_BYTES}, wire.MAX_REQUEST_BYTES)


def test_decode_fuzz_only_raises_wire_error():
    rng = random.Random(1234)
    valid = encode({"id": 1, "method": "call", "params": {"x": np.arange(4.0), "s": "abc"}})[4:]
    for i in range(3000):
        if i % 2:
            data = bytes(rng.getrandbits(8) for _ in range(rng.randint(0, 64)))
        else:
            data = bytearray(valid)
            for _ in range(rng.randint(1, 6)):
                data[rng.randrange(len(data))] = rng.getrandbits(8)
            data = bytes(data[: rng.randint(0, len(data))])
        try:
            decode_body(data)
        except WireError:
            pass
