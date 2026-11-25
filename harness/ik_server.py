"""The PyRoKi inverse-kinematics server the CaP-X privileged APIs call over HTTP.

CaP-X's APIs post to a fixed address, ``http://127.0.0.1:8116/ik``. One server serves every
simulator process on the machine. It is stateless (each request carries the previous joint
configuration), so sharing it does not couple episodes.

- ``ensure_ik_server()``: reuse a server that is already up, or start one owned by this process
  and stopped when it exits. ``CapxEnv`` calls it before building an env.
- ``python -m harness.ik_server``: run one in the foreground, for a long-lived shared server
  (several simulator processes, or a box). Processes that find it running reuse it.
"""

from __future__ import annotations

import atexit
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request

HOST = "127.0.0.1"
PORT = 8116  # fixed in CaP-X (capx.integrations.motion.pyroki.DEFAULT_URL)
START_TIMEOUT_S = 180.0

_lock = threading.Lock()
_owned: subprocess.Popen | None = None


def _port_open(timeout: float = 1.0) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        return s.connect_ex((HOST, PORT)) == 0


def ik_server_ready(timeout: float = 5.0) -> bool:
    """True if a PyRoKi IK server answers on 127.0.0.1:8116 (its OpenAPI schema lists /ik)."""
    if not _port_open():
        return False
    try:
        with urllib.request.urlopen(f"http://{HOST}:{PORT}/openapi.json", timeout=timeout) as r:
            return "/ik" in json.loads(r.read()).get("paths", {})
    except (OSError, ValueError):
        return False


def _command() -> list[str]:
    return [sys.executable, "-m", "capx.serving.launch_pyroki_server"]


def _child_env() -> dict[str, str]:
    from harness.capx_env import capx_root

    env = dict(os.environ)
    root = str(capx_root())
    env["PYTHONPATH"] = os.pathsep.join(p for p in (root, env.get("PYTHONPATH")) if p)
    return env


def start_ik_server(timeout: float = START_TIMEOUT_S) -> subprocess.Popen:
    """Start a server process and wait until it answers. Raises RuntimeError if it does not."""
    log = tempfile.TemporaryFile()
    proc = subprocess.Popen(
        _command(),
        env=_child_env(),
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            break
        if ik_server_ready():
            return proc
        time.sleep(0.5)
    if proc.poll() is None:
        proc.kill()
        proc.wait()
    log.seek(0)
    tail = log.read()[-4000:].decode(errors="replace")
    if ik_server_ready():  # lost a start race to another process: use its server
        return proc
    raise RuntimeError(f"PyRoKi IK server did not come up on {HOST}:{PORT}:\n{tail}")


def stop_ik_server() -> None:
    """Stop the server this process started, if any."""
    global _owned
    with _lock:
        proc, _owned = _owned, None
    if proc is not None and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def ensure_ik_server() -> None:
    """Make sure an IK server is up: reuse a running one, else start one owned by this
    process (stopped at exit)."""
    global _owned
    with _lock:
        if _owned is not None and _owned.poll() is None:
            return
        if ik_server_ready():
            return
        proc = start_ik_server()
        if proc.poll() is None:
            _owned = proc
            atexit.register(stop_ik_server)


def main() -> int:
    if ik_server_ready():
        print(f"an IK server is already running on {HOST}:{PORT}", file=sys.stderr)
        return 1
    os.execvpe(_command()[0], _command(), _child_env())  # replaces this process


if __name__ == "__main__":
    sys.exit(main())
