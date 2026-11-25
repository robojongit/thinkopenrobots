"""Scene seeding: ``reseed()`` and a walker that finds stray random generators."""

from __future__ import annotations

import functools
import random
import types
from collections import deque
from typing import Any

import numpy as np


def reseed(robosuite_env: Any, seed: int) -> None:
    """All 7 CaP-X tasks share robosuite_env.rng by identity
    (placement samplers and the wipe arena hold the same object),
    so resetting its state in place reseeds every sampler."""
    rng = robosuite_env.rng
    rng.bit_generator.state = np.random.default_rng(seed).bit_generator.state
    np.random.seed(seed)
    random.seed(seed)


def _has_contents(cell: types.CellType) -> bool:
    try:
        cell.cell_contents  # noqa: B018
    except ValueError:  # empty cell
        return False
    return True


_LEAVES = (str, bytes, bytearray, int, float, complex, bool, type(None), np.ndarray, np.generic)
_SKIP = (types.ModuleType, type, types.BuiltinFunctionType)


def find_stray_generators(
    root: Any, allowed: Any, *, root_name: str = "env", max_nodes: int = 500_000
) -> list[str]:
    """Walk the object graph from ``root`` and return the attribute path of every
    ``np.random.Generator`` or legacy ``np.random.RandomState`` that is not ``allowed``
    (compared by identity). Each object is visited once, by the first path found.

    Follows instance ``__dict__`` and ``__slots__`` attributes, dict values, items of lists,
    tuples, sets and deques, closure cells, bound methods' ``__self__`` and
    ``functools.partial`` arguments. Skips modules, classes and NumPy arrays.
    """
    found: list[str] = []
    seen: set[int] = set()
    queue: deque[tuple[Any, str]] = deque([(root, root_name)])
    while queue and len(seen) < max_nodes:
        obj, path = queue.popleft()
        if id(obj) in seen or isinstance(obj, _LEAVES) or isinstance(obj, _SKIP):
            continue
        seen.add(id(obj))
        if isinstance(obj, types.FunctionType):
            # Closures (e.g. lambdas registered as factories) can capture generators.
            cells = [c.cell_contents for c in obj.__closure__ or () if _has_contents(c)]
            queue.extend((v, f"{path}.<closure>") for v in cells)
            continue
        if isinstance(obj, types.MethodType):
            queue.append((obj.__self__, f"{path}.__self__"))
            continue
        if isinstance(obj, functools.partial):
            queue.extend([(obj.args, f"{path}.args"), (obj.keywords, f"{path}.keywords")])
            continue
        if isinstance(obj, (np.random.Generator, np.random.RandomState)):
            if obj is not allowed:
                found.append(path)
            continue
        if isinstance(obj, dict):
            queue.extend((v, f"{path}[{k!r}]") for k, v in obj.items())
            continue
        if isinstance(obj, (list, tuple, deque)):
            queue.extend((v, f"{path}[{i}]") for i, v in enumerate(obj))
            continue
        if isinstance(obj, (set, frozenset)):
            queue.extend((v, f"{path}{{}}") for v in obj)
            continue
        attrs = getattr(obj, "__dict__", None)
        if isinstance(attrs, dict):
            queue.extend((v, f"{path}.{k}") for k, v in attrs.items())
        for cls in type(obj).__mro__:
            for name in getattr(cls, "__slots__", ()):
                if isinstance(name, str) and hasattr(obj, name):
                    queue.append((getattr(obj, name), f"{path}.{name}"))
    return found
