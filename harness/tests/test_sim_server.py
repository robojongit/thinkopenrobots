"""Unit tests of the sim server's own helpers (episodes are tested in test_episode/test_attacks)."""

import numpy as np

from harness.sim_server import jsonable


def test_arrays_are_logged_as_summaries():
    arr = np.arange(20, dtype=np.float32).reshape(4, 5)
    assert jsonable(arr) == {
        "ndarray": "<f4",
        "shape": [4, 5],
        "nbytes": 80,
        "head": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
    }
    # A zero-size array with a large shape costs nothing to log.
    assert jsonable(np.zeros((500_000, 0), dtype=np.uint8))["head"] == []
    assert jsonable(np.array([np.nan, 1.0]))["head"] == ["nan", 1.0]
    assert jsonable((np.float64(1.5), {"a": b"xy"})) == [1.5, {"a": "<2 bytes>"}]
