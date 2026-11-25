"""The only module that touches CaP-X. All CaP-X imports are lazy, inside ``CapxEnv``.

Builds the task's high-level CaP-X env (``CodeExecutionEnvBase``) from its privileged YAML config
with ``privileged: true`` and rendering only when a video is wanted, reseeds and resets it, and
exposes what the sim server needs (see ``sim_server.EnvAdapter``).

Requirements at runtime:
- CaP-X importable (``uv sync`` installs the ``sim`` group: the ``harness/capx`` submodule and
  its RoboSuite submodule). ``INKLING_CAPX_ROOT`` overrides where the checkout is (default
  ``harness/capx``); configs are read from it and its ``capx`` package is the one imported.
- the PyRoKi IK server on 127.0.0.1:8116, which ``ensure_ik_server()`` reuses or starts.

Every ``CapxEnv`` runs all of its simulator work on one thread of its own: the sim server calls
it from its connection threads, but MuJoCo's offscreen (EGL) render context only works on the
thread that created it, and video frames rendered from another thread came out as noise.

MuJoCo resets the whole scene on its own when the simulation blows up (NaN or huge positions,
velocities or accelerations), which would put objects back where they started in the middle of
an episode. ``CapxEnv`` watches MuJoCo's warning counters around every API call and, once one
fires, fails the episode: that call and every later one raise ``RuntimeError`` and
``task_completed()`` is False.

The end-of-episode check (``task_completed``) first lets the scene settle for ``SETTLE_STEPS``
control steps holding the last command, so an object that is only passing through a goal state
(thrown, falling, sliding off) does not count, and then, like CaP-X's own ``step()``, computes
the reward before asking ``task_completed()`` (spill_wipe's reward is what records the wiped
markers).

Every ``CapxEnv`` builds its own low-level env and API objects. CaP-X's ``get_env`` caches
low-level envs per process and its APIs keep their last IK solution as the next warm start, so
reusing either would make an episode depend on the ones before it.
"""

from __future__ import annotations

import concurrent.futures
import functools
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from harness.seeding import reseed
from harness.tasks import TASKS

# Warnings after which MuJoCo resets the simulation (unless auto-reset is disabled).
RESET_WARNINGS = ("mjWARN_BADQPOS", "mjWARN_BADQVEL", "mjWARN_BADQACC")
UNSTABLE = "the simulation became unstable and MuJoCo reset the scene; the episode is failed"
CLOSE_TIMEOUT_S = 10.0
SETTLE_STEPS = 10  # control steps; 0.5 s at robosuite's default 20 Hz


def capx_root() -> Path:
    default = Path(__file__).resolve().parent / "capx"
    return Path(os.environ.get("INKLING_CAPX_ROOT", default)).resolve()


def import_capx() -> Any:
    """Import the ``capx`` package from ``capx_root()``; fail loudly if another one is loaded."""
    root = capx_root()
    if not (root / "capx" / "__init__.py").is_file():
        raise RuntimeError(
            f"no CaP-X checkout at {root}: run `git submodule update --init harness/capx` and "
            "`git -C harness/capx submodule update --init capx/third_party/robosuite`, "
            "or set INKLING_CAPX_ROOT"
        )
    if "capx" not in sys.modules and str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import capx

    loaded = Path(capx.__file__).resolve().parent.parent
    if loaded != root:
        raise RuntimeError(f"capx is imported from {loaded}, but INKLING_CAPX_ROOT is {root}")
    return capx


class CapxEnv:
    def __init__(self, task: str, seed: int, render: bool = False) -> None:
        self.unstable = False
        self._thread = ThreadPoolExecutor(1, thread_name_prefix=f"capx-{task}")
        try:
            self._on_env_thread(self._build, task, seed, render)
        except BaseException:
            self._thread.shutdown(wait=False)
            raise
        # The API functions, each run on the env thread.
        self.functions: dict[str, Any] = {
            name: self._threaded(fn) for name, fn in self._api_functions.items()
        }

    def _on_env_thread(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
        return self._thread.submit(fn, *args, **kwargs).result()

    def _threaded(self, fn: Any) -> Any:
        @functools.wraps(fn)
        def call(*args: Any, **kwargs: Any) -> Any:
            return self._on_env_thread(self._watched, fn, *args, **kwargs)

        return call

    def _reset_warnings(self) -> int:
        import mujoco

        data = self.env.low_level_env.robosuite_env.sim.data._data
        return sum(data.warning[getattr(mujoco.mjtWarning, w)].number for w in RESET_WARNINGS)

    def _watched(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
        """Run ``fn`` unless the simulation has become unstable; fail if it becomes so."""
        if self.unstable:
            raise RuntimeError(UNSTABLE)
        before = self._reset_warnings()
        try:
            return fn(*args, **kwargs)
        finally:
            if self._reset_warnings() > before:
                self.unstable = True
                raise RuntimeError(UNSTABLE)

    def _build(self, task: str, seed: int, render: bool) -> None:
        import_capx()
        import capx.envs  # noqa: F401  (registers the low-level envs)
        import capx.integrations  # noqa: F401  (registers the APIs)
        from capx.envs.base import get_env
        from capx.envs.configs.instantiate import instantiate
        from capx.envs.configs.loader import DictLoader

        from harness.ik_server import ensure_ik_server

        ensure_ik_server()
        spec = TASKS[task]
        self.spec = spec
        env_cfg = DictLoader.load([str(capx_root() / spec.config)])["env"]
        cfg = env_cfg["cfg"]
        if cfg["apis"] != [spec.api]:
            raise RuntimeError(f"{spec.config} lists APIs {cfg['apis']}, not {spec.api}")
        if not isinstance(cfg["low_level"], str):
            raise RuntimeError(f"{spec.config}: low_level must be a registered env name")
        cfg["privileged"] = True
        cfg["enable_render"] = render
        # A fresh low-level env, not get_env()'s per-process cached one.
        cfg["low_level"] = get_env.__wrapped__(
            cfg["low_level"], privileged=True, enable_render=render, viser_debug=False
        )
        self.env = instantiate(env_cfg)
        self.render = render

        reseed(self.env.low_level_env.robosuite_env, seed)
        self.env.reset(seed=seed)
        if render:
            self.env.enable_video_capture(True, clear=True)

        self._api_functions: dict[str, Any] = {}
        for api in self.env._apis.values():
            self._api_functions.update(api.functions())
        if set(self._api_functions) != spec.functions:
            raise RuntimeError(
                f"CaP-X API for {task} exposes {sorted(self._api_functions)}, "
                f"task table says {sorted(spec.functions)}"
            )
        # Same text CaP-X puts in its prompt: f"{task_prompt}\nAPIs:\n{api_docs}".
        self.instruction: str = self.env._task_prompt
        self.api_docs: str = "\n".join(
            f"\n{api.combined_doc().strip()}" for api in self.env._apis.values()
        )

    def observe(self) -> dict[str, Any]:
        return self._on_env_thread(self._observe)

    def _observe(self) -> dict[str, Any]:
        obs = self.env.low_level_env.get_observation()
        return {k: np.asarray(obs[k]) for k in self.spec.observe_keys if k in obs}

    def task_completed(self) -> bool:
        if self.unstable:
            return False
        try:
            return self._on_env_thread(self._watched, self._settle_and_check)
        except RuntimeError:
            if self.unstable:  # the scene blew up while settling
                return False
            raise

    def _settle_and_check(self) -> bool:
        low = self.env.low_level_env
        for _ in range(SETTLE_STEPS):
            if low.robosuite_env.done:  # past robosuite's horizon nothing can step (or move)
                break
            low._step_once()  # each env's own "hold": last joint target (one-arm) or current
            # joints (two-arm), current gripper commands
        self.env.compute_reward()  # as CaP-X's step() does before task_completed()
        return bool(low.task_completed())

    def write_video(self, path: Path) -> bool:
        """Write the episode video to ``path`` (an .mp4). False if nothing was recorded."""
        if not self.render:
            return False
        frames = self._on_env_thread(self.env.get_video_frames, clear=True)
        if not frames:
            return False
        import imageio

        with imageio.get_writer(path, fps=30, format="FFMPEG", codec="libx264") as writer:
            for frame in frames:
                writer.append_data(np.ascontiguousarray(frame))
        return True

    def close(self) -> None:
        """Close the simulator, waiting at most ``CLOSE_TIMEOUT_S`` for a call still running on
        the env thread (the close then runs after it, in the background)."""
        close = getattr(self.env.low_level_env.robosuite_env, "close", None)
        try:
            if close is not None:
                self._thread.submit(close).result(timeout=CLOSE_TIMEOUT_S)
        except concurrent.futures.TimeoutError:  # not the builtin one before Python 3.11
            pass
        finally:
            self._thread.shutdown(wait=False)


def make_env(task: str, seed: int, render: bool = False) -> CapxEnv:
    return CapxEnv(task, seed, render)
