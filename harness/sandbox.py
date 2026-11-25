"""Launching the untrusted processes: the agent and one code runner per ``robot.run``.

``ProcessSandbox`` runs them as ordinary subprocesses ("plain process" mode, for laptops and
tests). It is NOT a security boundary: the processes share the host user and filesystem. It does
give each process a minimal environment (no inherited secrets), its own session so the whole
process group is killed at the end, capped captured output, and one BLAS thread.

A child that leaves the process group (``setsid``, a daemon's double fork) would survive
``killpg``. On Linux the harness process therefore makes itself a child subreaper: such orphans
are re-parented to it instead of to init, and ``kill()`` and ``close()`` also kill, and reap,
every process orphaned onto the harness since the sandbox was created, with all their
descendants. That assumes one episode at a time per harness process (as ``bench`` and the
scorer run them); after ``close()``, ``kill()`` only kills its own process's tree, so a late
kill from a thread that outlived its episode cannot hit the next episode's processes.

``ContainerSandbox`` is the security boundary (scoring uses it): the agent and each code runner
run in a container of their own, from the agent image (``docker/agent``), under gVisor
(``runtime="runsc"``) or, on machines without it, plain ``runc``. See its docstring.
"""

from __future__ import annotations

import ctypes
import functools
import hashlib
import json
import os
import re
import secrets
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import IO, Protocol

ROOT = Path(__file__).resolve().parent.parent
SDK_DIR = ROOT / "sdk"
AGENT_OUTPUT_CAP = 1024 * 1024
RUNNER_OUTPUT_CAP = 256 * 1024
KILL_WAIT_S = 10.0
SWEEP_ROUNDS = 100
PR_SET_CHILD_SUBREAPER = 36

CONTAINER_UID = 65534  # nobody:nogroup
CONTAINER_USER = f"{CONTAINER_UID}:{CONTAINER_UID}"
CONTAINER_SOCKET = "/run/sim.sock"
EPISODE_LABEL = "inkling.episode"
DOCKER_TIMEOUT_S = 30.0
LABEL_KEY_RE = re.compile(r"[a-z0-9]+(?:[.-][a-z0-9]+)*")
LABEL_VALUE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,62}")
MAX_LABELS = 8
SLICE_RE = re.compile(r"[A-Za-z0-9_]+(?:-[A-Za-z0-9_]+)*\.slice")
CGROUP_PATH_RE = re.compile(r"(?:/[A-Za-z0-9_][A-Za-z0-9_.@-]{0,99}){1,8}")
CONTAINER_ENV = {
    "HOME": "/tmp",
    "LANG": "C.UTF-8",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
}


class SandboxProcess(Protocol):
    def wait(self, timeout: float | None) -> int | None:
        """Exit status, or None if still running after ``timeout`` seconds."""

    def kill(self) -> None:
        """Kill the process and everything it started. Idempotent."""

    def stdout(self) -> str:
        """Captured stdout (for the agent: stdout and stderr interleaved)."""

    def stderr(self) -> str: ...


class Sandbox(Protocol):
    def start_agent(self, agent_dir: Path, socket_path: Path) -> SandboxProcess:
        """Start ``python -m inkling_robo.run <agent_dir>`` talking to ``socket_path``."""

    def start_runner(self, socket_path: Path, token: str) -> SandboxProcess:
        """Start ``python -m inkling_robo.runner`` talking to ``socket_path``, with ``token`` in
        its environment as ``INKLING_ROBO_TOKEN`` (and nowhere the agent can see it). Nothing
        from the agent folder reaches the runner."""

    def close(self) -> None:
        """At the end of the episode: kill whatever the sandbox's processes left running."""


class _Capture:
    """Drains a pipe in a thread, keeping at most ``cap`` bytes."""

    def __init__(self, pipe: IO[bytes], cap: int) -> None:
        self._buf = bytearray()
        self._dropped = 0
        self._thread = threading.Thread(target=self._drain, args=(pipe, cap), daemon=True)
        self._thread.start()

    def _drain(self, pipe: IO[bytes], cap: int) -> None:
        with pipe:
            for chunk in iter(lambda: pipe.read1(65536), b""):
                keep = max(0, cap - len(self._buf))
                self._buf += chunk[:keep]
                self._dropped += len(chunk) - min(keep, len(chunk))

    def text(self) -> str:
        self._thread.join(timeout=5)
        out = self._buf.decode("utf-8", errors="replace")
        if self._dropped:
            out += f"\n[... {self._dropped} more bytes dropped]\n"
        return out


def _become_subreaper() -> bool:
    if sys.platform != "linux":
        return False
    libc = ctypes.CDLL(None, use_errno=True)
    return libc.prctl(PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) == 0


def _children() -> dict[int, list[int]]:
    """Parent pid -> child pids, for every process in /proc."""
    found: dict[int, list[int]] = {}
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/stat") as f:
                ppid = int(f.read().rsplit(")", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            continue
        found.setdefault(ppid, []).append(int(entry))
    return found


def _tree(roots: Iterable[int], children: dict[int, list[int]]) -> set[int]:
    seen: set[int] = set()
    todo = list(roots)
    while todo:
        pid = todo.pop()
        if pid not in seen:
            seen.add(pid)
            todo.extend(children.get(pid, []))
    return seen


class _Process:
    def __init__(
        self,
        argv: list[str],
        cwd: Path,
        cap: int,
        merge_stderr: bool,
        env: dict[str, str],
        sweep: Callable[[int], None],
    ) -> None:
        self._sweep = sweep
        self._proc = subprocess.Popen(
            argv,
            cwd=cwd,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT if merge_stderr else subprocess.PIPE,
            start_new_session=True,
        )
        assert self._proc.stdout is not None
        self._out = _Capture(self._proc.stdout, cap)
        self._err = _Capture(self._proc.stderr, cap) if self._proc.stderr else None

    def wait(self, timeout: float | None) -> int | None:
        try:
            return self._proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return None

    def kill(self) -> None:
        try:
            os.killpg(self._proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        self._sweep(self._proc.pid)
        self._proc.wait(timeout=KILL_WAIT_S)  # a process that outlives SIGKILL is a harness error

    def stdout(self) -> str:
        return self._out.text()

    def stderr(self) -> str:
        return self._err.text() if self._err else ""


def _child_env() -> dict[str, str]:
    return {
        "PATH": os.defpath,
        "PYTHONPATH": str(SDK_DIR),
        "PYTHONUNBUFFERED": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "LANG": "C.UTF-8",
    }


class ProcessSandbox:
    """Plain subprocesses with the harness's own Python. ``workdir`` is their cwd."""

    def __init__(self, workdir: Path) -> None:
        self.workdir = workdir
        workdir.mkdir(parents=True, exist_ok=True)
        self._subreaper = _become_subreaper()
        self._own_children = set(_children().get(os.getpid(), [])) if self._subreaper else set()
        """The harness's children that are not orphans from this sandbox: the ones it had
        before, and the processes the sandbox starts itself."""
        self._closed = False

    def close(self) -> None:
        self._sweep(None)
        self._closed = True

    def _start(self, argv: list[str], cap: int, merge_stderr: bool, env: dict[str, str]):
        proc = _Process(argv, self.workdir, cap, merge_stderr, env, self._sweep)
        self._own_children.add(proc._proc.pid)
        return proc

    def _sweep(self, pid: int | None) -> None:
        """SIGKILL ``pid``'s descendants, and (until ``close()``) every process orphaned onto
        the harness with its descendants, and reap the orphans, until none is left."""
        if not self._subreaper:
            return
        me = os.getpid()
        for _ in range(SWEEP_ROUNDS):
            children = _children()
            orphans = set() if self._closed else set(children.get(me, [])) - self._own_children
            descendants = _tree([pid], children) - {pid} if pid is not None else set()
            victims = descendants | _tree(orphans, children)
            if not victims:
                return
            for victim in victims:
                try:
                    os.kill(victim, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
            for orphan in orphans:
                try:
                    os.waitpid(orphan, os.WNOHANG)
                except ChildProcessError:
                    pass
            time.sleep(0.01)

    def start_agent(self, agent_dir: Path, socket_path: Path) -> SandboxProcess:
        argv = [sys.executable, "-s", "-m", "inkling_robo.run", str(agent_dir)]
        argv += ["--socket", str(socket_path)]
        return self._start(argv, AGENT_OUTPUT_CAP, True, _child_env())

    def start_runner(self, socket_path: Path, token: str) -> SandboxProcess:
        argv = [sys.executable, "-s", "-m", "inkling_robo.runner", "--socket", str(socket_path)]
        env = {**_child_env(), "INKLING_ROBO_TOKEN": token}
        return self._start(argv, RUNNER_OUTPUT_CAP, False, env)


# ---------------------------------------------------------------- containers


def agent_image() -> str:
    """The agent image's tag, ``inkling-agent:<version>``: the version is a hash of everything
    the image is built from (``docker/agent/`` and the SDK), so a stale image is never run."""
    digest = hashlib.sha256()
    for folder in (ROOT / "docker" / "agent", SDK_DIR / "inkling_robo"):
        for path in sorted(folder.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                digest.update(path.relative_to(ROOT).as_posix().encode() + b"\0")
                digest.update(path.read_bytes() + b"\0")
    return f"inkling-agent:{digest.hexdigest()[:12]}"


def _docker(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["docker", *args], capture_output=True, text=True, timeout=DOCKER_TIMEOUT_S
    )


def docker_runtimes() -> dict[str, dict]:
    """Docker's registered runtimes (name -> {"path", "runtimeArgs"}); {} without Docker."""
    try:
        info = _docker("info", "--format", "{{json .Runtimes}}")
        return json.loads(info.stdout) if info.returncode == 0 else {}
    except (OSError, subprocess.TimeoutExpired, ValueError):
        return {}


@functools.cache
def check_docker(runtime: str, image: str) -> None:
    """Raise ``RuntimeError`` unless Docker has ``runtime`` and ``image``. gVisor must be
    registered with ``--host-uds=open`` in its runtimeArgs: without it, a process in the sandbox
    cannot connect to a Unix socket bind-mounted from the host."""
    runtimes = docker_runtimes()
    if not runtimes:
        raise RuntimeError("Docker is not available (is the daemon running, may we use it?)")
    if runtime not in runtimes:
        raise RuntimeError(f"Docker has no runtime {runtime!r}; registered: {sorted(runtimes)}")
    args = runtimes[runtime].get("runtimeArgs") or []
    if runtime == "runsc" and not {"--host-uds=open", "--host-uds=all"} & set(args):
        raise RuntimeError(
            'runsc must be registered with "runtimeArgs": ["--host-uds=open"] in '
            f"/etc/docker/daemon.json (has {args})"
        )
    if _docker("image", "inspect", image).returncode != 0:
        raise RuntimeError(f"no Docker image {image}; build it with docker/agent/build.sh")


RUNSC_SANDBOX_TASKS = 32
"""Host tasks a gVisor sandbox uses before its first process forks (26 measured, runsc
release-20261123.0, systrap platform)."""
RUNSC_TASKS_PER_PROCESS = 4
"""Host tasks each further process in the sandbox may add: 2 measured for sleeping ones (a
systrap stub process and thread), about 3.5 for processes busy in a fork loop."""


def runsc_nproc(pids_limit: int) -> int:
    """``RLIMIT_NPROC`` for a gVisor container whose host ``--pids-limit`` is ``pids_limit``.

    Under gVisor ``--pids-limit`` bounds the sandbox's host tasks, not the processes inside it,
    so a fork bomb does not get EAGAIN: it takes the whole sandbox down when the host limit is
    reached. gVisor's own kernel does enforce ``RLIMIT_NPROC``, per sandbox (under runc the
    same limit would count every container's uid-65534 processes together, so it is not set
    there and the pids cgroup does the job). This leaves the host limit room to spare.
    """
    return max(1, (pids_limit - RUNSC_SANDBOX_TASKS) // RUNSC_TASKS_PER_PROCESS)


class _Container(_Process):
    """One container, created by ``docker create`` (so it exists, by name, before anyone can
    kill it) and run attached by ``docker start -a``, whose exit status is the container's."""

    def __init__(self, name: str, create: list[str], cap: int, merge_stderr: bool, env: dict):
        self.name = name
        self._lock = threading.Lock()
        self._killed = False
        made = subprocess.run(
            create, env=env, capture_output=True, text=True, timeout=DOCKER_TIMEOUT_S
        )
        if made.returncode != 0:
            raise RuntimeError(f"docker create failed: {made.stderr.strip()[:500]}")
        argv = ["docker", "start", "--attach", name]
        super().__init__(argv, Path("/"), cap, merge_stderr, env, sweep=lambda pid: None)

    def kill(self) -> None:
        """``docker kill``, a bounded wait for it to exit, then ``docker rm -f``. The kill is
        repeated while waiting: one sent before ``docker start`` has started it misses."""
        with self._lock:
            if self._killed:
                return
            deadline = time.monotonic() + KILL_WAIT_S
            while self._proc.poll() is None and time.monotonic() < deadline:
                _docker("kill", self.name)
                try:
                    self._proc.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    pass
            if self._proc.poll() is None:  # the attached client hangs: end it
                os.killpg(self._proc.pid, signal.SIGKILL)
                self._proc.wait(timeout=KILL_WAIT_S)
            _docker("rm", "-f", self.name)
            self._killed = True


def check_labels(labels: dict[str, str] | None) -> dict[str, str]:
    """Extra container labels, checked: at most 8, lowercase dotted keys (``inkling.job``) that
    are not ``inkling.episode`` (the sandbox's own), short plain values. Raises ValueError."""
    if labels is None:
        return {}
    if not isinstance(labels, dict) or len(labels) > MAX_LABELS:
        raise ValueError(f"labels must be a dict of at most {MAX_LABELS} entries")
    for key, value in labels.items():
        if not isinstance(key, str) or not LABEL_KEY_RE.fullmatch(key) or len(key) > 64:
            raise ValueError(f"bad label key {key!r} (want e.g. inkling.job)")
        if key == EPISODE_LABEL:
            raise ValueError(f"{EPISODE_LABEL} is set by the sandbox itself")
        if not isinstance(value, str) or not LABEL_VALUE_RE.fullmatch(value):
            raise ValueError(f"bad value for label {key}: {value!r}")
    return dict(labels)


def check_cgroup_parent(parent: str | None) -> str | None:
    """``--cgroup-parent``, checked: a systemd slice name (``inkling-s0012_1.slice``: what
    Docker's systemd cgroup driver requires; a ``-`` nests it under ``inkling.slice``) or, for
    the cgroupfs driver, an absolute path of plain components. Raises ValueError."""
    if parent is None:
        return None
    if not isinstance(parent, str) or len(parent) > 200:
        raise ValueError("cgroup_parent must be a short string")
    if SLICE_RE.fullmatch(parent):
        return parent
    if CGROUP_PATH_RE.fullmatch(parent) and not {".", ".."} & set(parent.split("/")):
        return parent
    raise ValueError(f"bad cgroup_parent {parent!r} (want e.g. inkling-job.slice or /inkling/job)")


class ContainerSandbox:
    """The agent, and every code runner, each in a container of its own.

    Per container (see ``create_argv``): ``runtime`` (``runsc`` for gVisor; ``runc`` where gVisor
    is missing), no network, read-only root, no capabilities, no new privileges, user
    65534:65534, private PID and IPC namespaces with no /dev/shm, 512 MB noexec tmpfs on /tmp as
    the only writable place, one CPU, ``memory`` with no swap, ``pids_limit`` processes (under
    gVisor: host tasks, with ``RLIMIT_NPROC`` bounding the processes inside; see
    ``runsc_nproc``), no core dumps or POSIX message queues, no log driver (output is read from
    the attached client and capped), and the image's Python run explicitly (``-I``: no PYTHON*
    variables, user site or cwd on the path), so nothing depends on the image's entrypoint or
    environment.

    Mounts, and nothing else: the agent gets its socket at /run/sim.sock and a copy of the agent
    folder at /agent, read-only; runner N gets run N's socket at /run/sim.sock only (nothing
    from the agent folder), with the run's token in its environment (passed by name, so it never
    appears on a command line). A socket is never reached through its directory: that stays 0700
    and the harness's. The socket itself is made connectable by uid 65534: chowned to it (0600)
    when the harness may chown, otherwise mode 0602 (connect needs only write permission), which
    the 0700 directory keeps from every other host user.

    Containers are labelled ``inkling.episode=<id>``; ``close()`` removes every one left.

    Optional, for a scorer that runs episodes as one job: ``labels`` adds ``--label k=v`` to
    every container (e.g. ``{"inkling.job": "s0012-1"}``, so a job's leftovers can be found and
    removed by job, see ``check_labels``), and ``cgroup_parent`` sets ``--cgroup-parent`` (e.g.
    the job's systemd slice, so its memory cap also covers the containers; see
    ``check_cgroup_parent``). Both are validated here and nothing else is passed through.
    """

    def __init__(
        self,
        workdir: Path,
        *,
        runtime: str = "runsc",
        image: str | None = None,
        cpus: float = 1.0,
        memory: str = "4g",
        pids_limit: int = 256,
        tmpfs_size: str = "512m",
        cpuset: str | None = None,
        labels: dict[str, str] | None = None,
        cgroup_parent: str | None = None,
    ) -> None:
        self.labels = check_labels(labels)
        self.cgroup_parent = check_cgroup_parent(cgroup_parent)
        self.workdir = workdir
        workdir.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.runtime = runtime
        self.image = image or agent_image()
        self.cpus = cpus
        self.memory = memory
        self.pids_limit = pids_limit
        self.tmpfs_size = tmpfs_size
        self.cpuset = cpuset
        self.episode = secrets.token_hex(8)
        self._runs = 0
        check_docker(runtime, self.image)

    def create_argv(
        self, name: str, mounts: list[tuple[Path, str]], command: list[str], env: Iterable[str]
    ) -> list[str]:
        """``docker create`` for one container; ``mounts`` are (host path, container path),
        all read-only except the socket; ``env`` names variables passed from our environment."""
        argv = ["docker", "create", "--rm", "--pull", "never", "--name", name]
        argv += ["--label", f"{EPISODE_LABEL}={self.episode}"]
        for key, value in sorted(self.labels.items()):
            argv += ["--label", f"{key}={value}"]
        argv += ["--runtime", self.runtime]
        if self.cgroup_parent:
            argv += ["--cgroup-parent", self.cgroup_parent]
        argv += ["--network", "none", "--ipc", "none", "--read-only", "--cap-drop", "ALL"]
        argv += ["--security-opt", "no-new-privileges", "--user", CONTAINER_USER]
        argv += ["--tmpfs", f"/tmp:rw,nosuid,nodev,noexec,size={self.tmpfs_size}"]
        argv += ["--cpus", str(self.cpus), "--memory", self.memory, "--memory-swap", self.memory]
        argv += ["--pids-limit", str(self.pids_limit)]
        if self.runtime == "runsc":
            argv += ["--ulimit", f"nproc={runsc_nproc(self.pids_limit)}"]
        argv += ["--ulimit", "core=0", "--ulimit", "msgqueue=0", "--log-driver", "none"]
        if self.cpuset:
            argv += ["--cpuset-cpus", self.cpuset]
        for key, value in CONTAINER_ENV.items():
            argv += ["--env", f"{key}={value}"]
        for key in env:
            argv += ["--env", key]
        for src, dst in mounts:
            if "," in str(src):
                raise ValueError(f"cannot mount a path with a comma: {src}")
            readonly = "" if dst == CONTAINER_SOCKET else ",readonly"
            argv += ["--mount", f"type=bind,src={src},dst={dst}{readonly}"]
        argv += ["--workdir", "/tmp", "--entrypoint", "python", self.image]
        # No -u: gVisor's pipes take at most 64 KiB per write, and an unbuffered (-u) Python
        # drops the rest of a short write. The SDK's entry points line-buffer stdout instead.
        return argv + ["-I", "-B", *command]

    def _share_socket(self, path: Path) -> None:
        folder = path.parent.stat()
        if folder.st_uid != os.getuid() or folder.st_mode & 0o077:
            raise RuntimeError(f"{path.parent} must be ours and 0700")
        try:
            os.chown(path, CONTAINER_UID, CONTAINER_UID)
        except PermissionError:
            os.chmod(path, 0o602)

    def _private_copy(self, src: Path, name: str) -> Path:
        """A copy of folder ``src`` without __pycache__ (symlinks are copied as links, so they
        resolve inside the container) in the workdir, readable by the container."""
        dst = self.workdir / name
        shutil.copytree(src, dst, symlinks=True, ignore=shutil.ignore_patterns("__pycache__"))
        for path in [dst, *dst.rglob("*")]:
            if not path.is_symlink():
                path.chmod(0o755 if path.is_dir() else 0o644)
        return dst

    def _start(
        self,
        name: str,
        mounts: list[tuple[Path, str]],
        command: list[str],
        cap: int,
        merge_stderr: bool,
        secret_env: dict[str, str],
    ) -> SandboxProcess:
        create = self.create_argv(name, mounts, command, secret_env)
        return _Container(name, create, cap, merge_stderr, {**os.environ, **secret_env})

    def start_agent(self, agent_dir: Path, socket_path: Path) -> SandboxProcess:
        self._share_socket(socket_path)
        agent = self._private_copy(agent_dir, "agent")
        mounts = [(socket_path, CONTAINER_SOCKET), (agent, "/agent")]
        command = ["-m", "inkling_robo.run", "/agent", "--socket", CONTAINER_SOCKET]
        return self._start(
            f"inkling-{self.episode}-agent", mounts, command, AGENT_OUTPUT_CAP, True, {}
        )

    def start_runner(self, socket_path: Path, token: str) -> SandboxProcess:
        n = self._runs
        self._runs += 1
        self._share_socket(socket_path)
        mounts = [(socket_path, CONTAINER_SOCKET)]
        command = ["-m", "inkling_robo.runner", "--socket", CONTAINER_SOCKET]
        env = {"INKLING_ROBO_TOKEN": token}
        return self._start(
            f"inkling-{self.episode}-r{n}", mounts, command, RUNNER_OUTPUT_CAP, False, env
        )

    def close(self) -> None:
        found = _docker("ps", "-aq", "--filter", f"label={EPISODE_LABEL}={self.episode}")
        if found.stdout.split():
            _docker("rm", "-f", *found.stdout.split())


def make_sandbox(
    kind: str,
    runtime: str = "runsc",
    *,
    labels: dict[str, str] | None = None,
    cgroup_parent: str | None = None,
) -> Callable[[Path], Sandbox]:
    """The ``sandbox`` factory for ``run_episode``: "process" or "container" (on ``runtime``,
    with optional ``labels`` and ``cgroup_parent``, see ``ContainerSandbox``; the plain-process
    sandbox takes neither)."""
    if kind == "process":
        if labels or cgroup_parent:
            raise ValueError("labels and cgroup_parent need the container sandbox")
        return ProcessSandbox
    if kind == "container":
        return functools.partial(
            ContainerSandbox,
            runtime=runtime,
            labels=check_labels(labels),
            cgroup_parent=check_cgroup_parent(cgroup_parent),
        )
    raise ValueError(f"unknown sandbox {kind!r}")
