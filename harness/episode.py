"""Run one episode: env + sim server + relay + agent, then the checker exactly once.

Files written to ``out`` (see ``save`` in sim_server for the per-component ones):

    result.json       the result dict, minus "seed", plus "task_completed", "limit_hit",
                      "upstream_errors"
    usage.json        calls, tokens and seconds against the budget; upstream_failures lists
                      each failed model request (kind, HTTP status, Retry-After seconds)
    transcript.jsonl  timeline of llm / call / run events
    llm.jsonl         every model request and reply
    runs.json         every robot.run() attempt; code in code/run_NNN.py
    errors.json       harness-side errors (bad frames, handler errors)
    agent.log         the agent's stdout+stderr (capped)
    video.mp4         only with video=True and a renderable env

The result's ``ended`` is "returned", "error" (``solve`` raised), "timeout" or "crash": the
harness itself raised once the episode had started (its sandbox, sim server, checker, saving or
closing the env); ``crash`` then holds the reason, and the episode is a failure. Exceptions
before the episode starts (building the env, the sandbox or the sim server) are raised.

The seed is never written to disk, so test seeds stay in memory; name the folder after it for
training runs.
"""

from __future__ import annotations

import json
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from harness.llm_relay import DEFAULT_CONTEXT_LENGTH, DEFAULT_MODEL, Budget, LLMRelay
from harness.sandbox import ProcessSandbox, Sandbox
from harness.sim_server import EnvAdapter, SimServer
from harness.tasks import TASKS

EnvFactory = Callable[[str, int, bool], EnvAdapter]


def run_episode(
    task: str,
    seed: int,
    agent_dir: Path,
    model_url: str,
    budget: Budget,
    out: Path,
    video: bool = False,
    *,
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
    context_length: int = DEFAULT_CONTEXT_LENGTH,
    cache_salt: bool = True,
    make_env: EnvFactory | None = None,
    sandbox: Callable[[Path], Sandbox] = ProcessSandbox,
) -> dict[str, Any]:
    spec = TASKS[task]
    agent_dir = Path(agent_dir).resolve()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if make_env is None:
        from harness.capx_env import make_env
    env = make_env(task, seed, video)
    sim: SimServer | None = None
    crash: str | None = None
    try:
        # Socket paths must stay short (~100 bytes), so they live in a temp dir, not in out.
        with tempfile.TemporaryDirectory(prefix="inkling-", ignore_cleanup_errors=True) as tmp:
            tmp_dir = Path(tmp)
            box = sandbox(tmp_dir / "work")
            deadline = time.monotonic() + budget.wall_s
            relay = LLMRelay(
                model_url,
                seed,
                budget,
                deadline,
                model=model,
                api_key=api_key,
                context_length=context_length,
                cache_salt=cache_salt,
            )
            sim = SimServer(spec, env, relay, box, tmp_dir)
            # The episode has started: from here on a harness exception is a failed episode
            # with a reason, never an exception that a caller might answer with a re-run.
            agent = None
            try:
                agent = box.start_agent(agent_dir, sim.listen_agent(tmp_dir / "a.sock"))
                status = agent.wait(timeout=max(deadline - time.monotonic(), 0.0))
                agent.kill()  # also kills anything the agent left running
                ended = "timeout" if status is None else "returned" if status == 0 else "error"
                sim.close()
                box.close()
                completed = sim.task_completed()  # first and only check
            except Exception as e:
                crash = _reason(e)
                ended, completed = "crash", False
                if agent is not None:
                    _quietly(agent.kill)
                _quietly(sim.close)
                _quietly(box.close)
            seconds = round(sim.elapsed, 3)
            try:
                sim.save(out)
                (out / "agent.log").write_text(agent.stdout() if agent else "")
                if video and crash is None and not sim.stalled:
                    env.write_video(out / "video.mp4")
            except Exception as e:
                crash = crash or _reason(e)
                ended, completed = "crash", False
    finally:
        try:
            env.close()
        except Exception as e:
            if sim is None:  # the episode never started: report it as an exception
                raise
            crash = crash or _reason(e)
            ended, completed = "crash", False

    result = {
        "task": task,
        "seed": seed,
        "success": completed and ended != "timeout",
        "tokens": sim.tokens,
        "calls": sim.calls,
        "seconds": seconds,
        "ended": ended,
        "crash": crash,
    }
    usage = {
        "calls": relay.calls,
        "prompt_tokens": relay.prompt_tokens,
        "completion_tokens": relay.completion_tokens,
        "tokens": relay.tokens,
        "seconds": seconds,
        "budget": {"calls": budget.calls, "tokens": budget.tokens, "wall_s": budget.wall_s},
        "limit_hit": "wall" if ended == "timeout" else relay.limit_hit,
        "upstream_errors": relay.upstream_errors,
        "upstream_failures": relay.upstream_failures,
    }
    record = {k: v for k, v in result.items() if k != "seed"}
    record.update(
        task_completed=completed,
        limit_hit=usage["limit_hit"],
        upstream_errors=relay.upstream_errors,
    )
    (out / "usage.json").write_text(json.dumps(usage, indent=1))
    (out / "result.json").write_text(json.dumps(record, indent=1))
    return result


def _quietly(fn: Callable[[], Any]) -> None:
    try:
        fn()
    except Exception:
        pass


def _reason(e: Exception) -> str:
    return f"{type(e).__name__}: {e}"[:1000]
