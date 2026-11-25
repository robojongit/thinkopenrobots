"""The task table, and (when a CaP-X checkout is available) a static check against CaP-X.

Set ``INKLING_CAPX_ROOT`` to a CaP-X checkout to run the CaP-X checks; they read source with
``ast`` and never import CaP-X.
"""

import ast
import re

import pytest

from harness.capx_env import capx_root
from harness.tasks import TASKS

# Calls that move an arm or a gripper inside the CaP-X privileged APIs.
MOTION_MARKERS = ("move_to_joints", "_set_gripper", "_step_once", "open_gripper(", "close_gripper(")


def test_table_is_consistent():
    assert sorted(TASKS) == sorted(
        [
            "cube_lifting",
            "cube_stack",
            "cube_restack",
            "nut_assembly",
            "spill_wipe",
            "two_arm_lift",
            "two_arm_handover",
        ]
    )
    for spec in TASKS.values():
        assert spec.reads and spec.motions
        assert not spec.reads & spec.motions
        assert spec.observe_keys


ROOT = capx_root()
needs_capx = pytest.mark.skipif(
    not (ROOT / "capx").is_dir(), reason=f"no CaP-X checkout at {ROOT} (set INKLING_CAPX_ROOT)"
)


def _find_class(name: str) -> tuple[ast.ClassDef, str]:
    for path in sorted((ROOT / "capx" / "integrations" / "franka").glob("*.py")):
        source = path.read_text()
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.ClassDef) and node.name == name:
                return node, source
    raise AssertionError(f"class {name} not found in CaP-X")


def _functions_keys(cls: ast.ClassDef) -> set[str]:
    (fn,) = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "functions"]
    dicts = [n for n in ast.walk(fn) if isinstance(n, ast.Dict)]
    return {k.value for d in dicts for k in d.keys if isinstance(k, ast.Constant)}


def _reachable_source(cls: ast.ClassDef, source: str, name: str) -> str:
    """Source of method ``name`` plus every method of the class it reaches via ``self.x(``."""
    methods = {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}
    todo, seen, parts = [name], set(), []
    while todo:
        current = todo.pop()
        if current in seen or current not in methods:
            continue
        seen.add(current)
        parts.append(ast.get_source_segment(source, methods[current]) or "")
        for node in ast.walk(methods[current]):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "self"
            ):
                todo.append(node.attr)
    return "\n".join(parts)


@needs_capx
@pytest.mark.parametrize("task", sorted(TASKS))
def test_table_matches_capx_privileged_api(task):
    spec = TASKS[task]
    config = (ROOT / spec.config).read_text()
    assert f"- {spec.api}" in config, f"{spec.config} does not list {spec.api}"
    # The config names the API; the registry maps the name to a class (here: same name).
    registry = (ROOT / "capx" / "integrations" / "__init__.py").read_text()
    assert re.search(rf'register_api\(\s*"{spec.api}"', registry)
    cls, source = _find_class(spec.api)
    assert _functions_keys(cls) == set(spec.functions)
    for name in spec.reads:
        body = _reachable_source(cls, source, name)
        assert not any(m in body for m in MOTION_MARKERS), f"{name} looks like motion"
    for name in spec.motions:
        body = _reachable_source(cls, source, name)
        assert any(m in body for m in MOTION_MARKERS), f"{name} does not look like motion"
