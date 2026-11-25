"""`bench doctor`: check the setup and print an exact fix command for each failing check."""

from __future__ import annotations

import ast
import importlib.util
import shutil
import sys
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urlparse

from harness.capx_env import capx_root

OK, FAIL, WARN, SKIP = "ok", "fail", "warn", "skip"
PYTHON_RANGE = ((3, 10), (3, 13))
UV_INSTALL = {
    "win32": 'powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"',
    "posix": "curl -LsSf https://astral.sh/uv/install.sh | sh",
}
# The local simulator is the CaP-X submodule, its RoboSuite submodule (no other CaP-X submodule
# is needed) and the default `sim` dependency group that installs them.
SIM_INSTALL = (
    "git submodule update --init harness/capx && "
    "git -C harness/capx submodule update --init capx/third_party/robosuite && uv sync"
)
REACH_TIMEOUT_S = 5.0


@dataclass(frozen=True)
class Check:
    name: str
    status: str
    detail: str
    fix: str | None = None


def is_native_windows() -> bool:
    return sys.platform == "win32"


def check_python() -> Check:
    version = sys.version_info[:2]
    text = ".".join(map(str, sys.version_info[:3]))
    if PYTHON_RANGE[0] <= version < PYTHON_RANGE[1]:
        return Check("python", OK, text)
    return Check(
        "python", FAIL, f"{text}; need 3.10, 3.11 or 3.12", "uv python install 3.12 && uv sync"
    )


def check_uv() -> Check:
    path = shutil.which("uv")
    if path:
        return Check("uv", OK, path)
    return Check(
        "uv", FAIL, "uv not found on PATH", UV_INSTALL["win32" if is_native_windows() else "posix"]
    )


def check_agent(agent_dir: Path) -> Check:
    entry, notes = agent_dir / "agent.py", agent_dir / "NOTES.md"
    if not entry.is_file():
        return Check("agent", FAIL, f"{entry} is missing", f"git checkout -- {entry.as_posix()}")
    try:
        tree = ast.parse(entry.read_text())
    except SyntaxError as e:
        return Check(
            "agent",
            FAIL,
            f"{entry} has a syntax error at line {e.lineno}",
            f"uv run python -m py_compile {entry.as_posix()}",
        )
    if not any(isinstance(n, ast.FunctionDef) and n.name == "solve" for n in tree.body):
        return Check(
            "agent",
            FAIL,
            f"{entry} does not define solve(task, robot, llm)",
            f"git checkout -- {entry.as_posix()}",
        )
    if not notes.is_file():
        return Check("agent", FAIL, f"{notes} is missing", f'echo "# Notes" > {notes.as_posix()}')
    return Check("agent", OK, f"{entry} defines solve; {notes.name} present")


def check_simulator() -> Check:
    if is_native_windows():
        return Check(
            "simulator",
            SKIP,
            "local simulation is unsupported on native Windows; remote is the default",
        )
    missing = [m for m in ("robosuite",) if importlib.util.find_spec(m) is None]
    if importlib.util.find_spec("capx") is None and not (capx_root() / "capx").is_dir():
        missing.append("capx")
    if missing:
        return Check("simulator", FAIL, f"not importable: {', '.join(missing)}", SIM_INSTALL)
    return Check("simulator", OK, "robosuite and CaP-X found")


def check_model_url(url: str | None) -> Check:
    if not url:
        return Check(
            "model_url",
            FAIL,
            "no model URL configured",
            "export INKLING_MODEL_URL=http://localhost:11434/v1",
        )
    probe_url = url.rstrip("/") + "/models"
    try:
        with urllib.request.urlopen(probe_url, timeout=REACH_TIMEOUT_S):
            return Check("model_url", OK, f"{url} answers")
    except urllib.error.HTTPError as e:
        if e.code == 503 and e.headers.get("Retry-After"):
            return Check("model_url", WARN, f"{url} is waking up (503); wait and retry")
        if e.code < 500:  # it answered, so it is reachable (401/404 on /models is normal)
            return Check("model_url", OK, f"{url} answers (HTTP {e.code} on /models)")
        return Check("model_url", FAIL, f"{url} returned HTTP {e.code}", _url_fix(url))
    except (urllib.error.URLError, OSError, ValueError) as e:
        return Check(
            "model_url", FAIL, f"{url} unreachable: {getattr(e, 'reason', e)}", _url_fix(url)
        )


def _url_fix(url: str) -> str:
    if urlparse(url).hostname in ("localhost", "127.0.0.1"):
        return "ollama serve   # or start your local model server, then rerun bench doctor"
    return f"curl -sS -o /dev/null -w '%{{http_code}}\\n' {url.rstrip('/')}/models"


def check_platform() -> Check:
    if is_native_windows():
        return Check(
            "platform",
            WARN,
            "native Windows: local simulation is unsupported, remote is the default",
            None,
        )
    return Check("platform", OK, sys.platform)


def run_checks(agent_dir: Path, model_url: str | None) -> list[Check]:
    return [
        check_platform(),
        check_python(),
        check_uv(),
        check_agent(agent_dir),
        check_simulator(),
        check_model_url(model_url),
    ]


def format_checks(checks: list[Check]) -> str:
    lines = []
    for c in checks:
        lines.append(f"[{c.status:>4}] {c.name}: {c.detail}")
        if c.fix:
            lines.append(f"       fix: {c.fix}")
    return "\n".join(lines)


def as_dicts(checks: list[Check]) -> list[dict[str, str | None]]:
    return [asdict(c) for c in checks]
