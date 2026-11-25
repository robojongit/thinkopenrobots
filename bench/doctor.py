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

from .coding_agent import EXAMPLE, CodingAgentProblem, parse_notes
from .config import Config
from .gateway import Gateway, GatewayError

OK, FAIL, WARN, SKIP, TODO = "ok", "fail", "warn", "skip", "todo"
"""Only FAIL fails `bench doctor`. TODO is something the person still has to do (log in)."""
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
LOGIN_FIX = "uv run bench login --no-wait"
ASLEEP_NOTE = (
    "the shared model may be asleep and takes about 5 minutes to wake on first use; "
    "bench train and submit wait by themselves"
)


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


def check_coding_agent(agent_dir: Path) -> Check:
    """The ``Coding agent:`` line of NOTES.md that `bench submit` needs (see coding_agent.py)."""
    notes = agent_dir / "NOTES.md"
    if not notes.is_file():
        return Check("coding_agent", SKIP, f"{notes} is missing (see the agent check)")
    try:
        label = parse_notes(notes.read_bytes())
    except CodingAgentProblem as e:
        return Check(
            "coding_agent",
            FAIL,
            e.message,
            f'edit {notes.as_posix()}: one line "{EXAMPLE}" naming the coding agent you use',
        )
    return Check("coding_agent", OK, f"{label} (shown on the leaderboard with each submission)")


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


def check_model_url(url: str | None, via_gateway: bool = False) -> Check:
    """Is the model server reachable? ``via_gateway``: it is the competition's gateway, which
    does not say whether the shared model is awake (asking would wake it), so say so."""
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
            return Check("model_url", OK, _answers(url, via_gateway))
    except urllib.error.HTTPError as e:
        if e.code == 503 and e.headers.get("Retry-After"):
            return Check("model_url", WARN, f"{url} is waking up (503); wait and retry")
        if e.code < 500:  # it answered, so it is reachable (401/404 on /models is normal)
            return Check("model_url", OK, _answers(url, via_gateway, e.code))
        return Check("model_url", FAIL, f"{url} returned HTTP {e.code}", _url_fix(url))
    except (urllib.error.URLError, OSError, ValueError) as e:
        return Check(
            "model_url", FAIL, f"{url} unreachable: {getattr(e, 'reason', e)}", _url_fix(url)
        )


def _answers(url: str, via_gateway: bool, http_code: int | None = None) -> str:
    if via_gateway:
        return f"gateway {url} reachable; {ASLEEP_NOTE}"
    return f"{url} answers" + (f" (HTTP {http_code} on /models)" if http_code else "")


def check_login(config: Config, own_model: bool = False) -> Check:
    """Is the saved gateway key accepted? One authenticated `GET /v1/me`; the key is never
    printed. Nothing here fails `doctor`: with ``own_model`` (a model URL that is not the
    gateway) being logged out is fine, and otherwise it is a todo with the exact command."""
    if not config.gateway_url:
        return Check("login", SKIP, "no gateway configured")
    key = config.read_key()
    # Without a key, or with a rejected one, the person logs in; with their own model they may
    # never need to (submit, status, pick and --remote do).
    missing = TODO if not own_model else SKIP
    note = " (not needed while you use your own model; submit, status and pick need it)"
    if not key:
        return Check("login", missing, "not logged in" + note * own_model, LOGIN_FIX)
    try:
        account = Gateway(config.gateway_url, key, timeout=REACH_TIMEOUT_S).me().get("account")
    except GatewayError as e:
        if e.status in (401, 403) and e.code != "account_disabled":
            return Check(
                "login", missing, "the saved key was rejected" + note * own_model, LOGIN_FIX
            )
        return Check("login", WARN, f"could not check the saved key: {e.message}")
    return Check("login", OK, f"logged in as {account}" if account else "logged in")


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


def run_checks(agent_dir: Path, model_url: str | None, config: Config | None = None) -> list[Check]:
    via_gateway = bool(config and model_url and config.is_gateway_url(model_url))
    checks = [
        check_platform(),
        check_python(),
        check_uv(),
        check_agent(agent_dir),
        check_coding_agent(agent_dir),
        check_simulator(),
        check_model_url(model_url, via_gateway),
    ]
    if config is not None:
        checks.append(check_login(config, own_model=bool(model_url) and not via_gateway))
    return checks


def format_checks(checks: list[Check]) -> str:
    lines = []
    for c in checks:
        lines.append(f"[{c.status:>4}] {c.name}: {c.detail}")
        if c.fix:
            lines.append(f"       fix: {c.fix}")
    return "\n".join(lines)


def as_dicts(checks: list[Check]) -> list[dict[str, str | None]]:
    return [asdict(c) for c in checks]
