"""`bench view`: readable, compact traces of episodes, as text or JSON, for an agent to read."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .exitcodes import CliError, ExitCode
from .runs import SCORED, Episode, list_episodes, read_json

REPLY_CHARS = (700, 300)
CODE_CHARS = (1500, 500)
STDOUT_CHARS = (400, 400)
STDERR_CHARS = (200, 600)
LOG_CHARS = (200, 600)
DEFAULT_LIMIT = 5


def clip(text: str, sizes: tuple[int, int], full: bool) -> str:
    """Keep the start and the end of a long text (tracebacks and answers end with what matters)."""
    head, tail = sizes
    if full or len(text) <= head + tail:
        return text
    return f"{text[:head]}\n[... {len(text) - head - tail} chars omitted ...]\n{text[-tail:]}"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    except (OSError, ValueError):
        return []


def outcome(ep: Episode) -> str:
    if ep.status != SCORED:
        return ep.status
    return "success" if ep.success else "failed"


def build_trace(ep: Episode, full: bool = False) -> dict[str, Any]:
    """Everything worth reading about one episode, in order, as plain data."""
    # A transcript llm event names its llm.jsonl entry by seq; refused calls have no entry.
    llm = {entry.get("seq"): entry for entry in read_jsonl(ep.path / "llm.jsonl")}
    runs = {r["index"]: r for r in read_json(ep.path / "runs.json", [])}
    events: list[dict[str, Any]] = []
    for e in read_jsonl(ep.path / "transcript.jsonl"):
        if e["kind"] == "llm":
            entry = llm.get(e.get("seq"), {}) if e.get("seq") is not None else {}
            if not e.get("ok", True):
                events.append({"type": "model_error", "error": entry.get("error", e.get("error"))})
                continue
            reply = entry.get("reply", {})
            events.append(
                {
                    "type": "reply",
                    "index": entry.get("index"),
                    "finish_reason": reply.get("finish_reason"),
                    "tokens": reply.get("prompt_tokens", 0) + reply.get("completion_tokens", 0),
                    "reasoning_chars": len(reply.get("reasoning") or ""),
                    "content": clip(reply.get("content", ""), REPLY_CHARS, full),
                }
            )
        elif e["kind"] == "run" and e.get("accepted") is False:
            events.append({"type": "run_refused", "index": e["index"], "reason": e["refused"]})
        elif e["kind"] == "run" and e.get("phase") == "end":
            run = runs.get(e["index"], {})
            try:
                code = (ep.path / run["code_file"]).read_text()
            except (KeyError, OSError):
                code = ""
            events.append(
                {
                    "type": "run",
                    "index": e["index"],
                    "ok": e["ok"],
                    "seconds": run.get("seconds"),
                    "timed_out": run.get("timed_out", False),
                    "code": clip(code, CODE_CHARS, full),
                    "stdout": clip(run.get("stdout", ""), STDOUT_CHARS, full),
                    "stderr": clip(run.get("stderr", ""), STDERR_CHARS, full),
                }
            )
        elif e["kind"] == "call" and not e.get("ok", True):
            events.append(
                {
                    "type": "api_error",
                    "source": e["source"],
                    "run": e.get("run"),
                    "name": e["name"],
                    "error": e["error"],
                }
            )
    marker = read_json(ep.path / "infra.json") or read_json(ep.path / "crash.json") or {}
    try:
        log = (ep.path / "agent.log").read_text()
    except OSError:
        log = ""
    return {
        "task": ep.task,
        "seed": ep.seed,
        "outcome": outcome(ep),
        "result": ep.result,
        "problem": marker.get("detail") or marker.get("error"),
        "events": events,
        "harness_errors": read_json(ep.path / "errors.json", []),
        "agent_log": clip(log, LOG_CHARS, full),
    }


def indent(text: str, prefix: str = "  ") -> str:
    return "\n".join(prefix + line for line in text.splitlines()) if text else prefix + "(empty)"


def summary_line(ep: Episode) -> str:
    r = ep.result or {}
    detail = f" ended={r['ended']} calls={r['calls']} tokens={r['tokens']}" if r else ""
    return f"{ep.task} seed {ep.seed}: {outcome(ep)}{detail}"


def render_trace(t: dict[str, Any]) -> str:
    r = t["result"] or {}
    head = f"=== {t['task']} seed {t['seed']}: {t['outcome'].upper()}"
    if r:
        head += (
            f"  ended={r['ended']} calls={r['calls']} tokens={r['tokens']} {r['seconds']}s"
            f" limit_hit={r.get('limit_hit')}"
        )
    lines = [head]
    if t["problem"]:
        lines.append(f"problem: {t['problem']}")
    for e in t["events"]:
        if e["type"] == "reply":
            lines.append(
                f"[model reply {e['index']}] finish={e['finish_reason']} tokens={e['tokens']}"
                f" reasoning={e['reasoning_chars']} chars"
            )
            lines.append(indent(e["content"]))
        elif e["type"] == "model_error":
            lines.append(f"[model error] {e['error']}")
        elif e["type"] == "run_refused":
            lines.append(f"[run {e['index']} refused] {e['reason']}")
        elif e["type"] == "run":
            status = "ok" if e["ok"] else "FAILED" + (" (timed out)" if e["timed_out"] else "")
            lines.append(f"[run {e['index']}] {status} {e['seconds']}s")
            lines += ["  code:", indent(e["code"], "    ")]
            lines += ["  stdout:", indent(e["stdout"], "    ")]
            lines += ["  stderr:", indent(e["stderr"], "    ")]
        elif e["type"] == "api_error":
            where = f"run {e['run']}" if e["run"] is not None else e["source"]
            lines.append(f"[api error] {where} {e['name']}: {e['error']}")
    for err in t["harness_errors"]:
        lines.append(f"[harness error] {err['where']}: {err['message']}")
    if t["agent_log"].strip():
        lines += ["[agent.log]", indent(t["agent_log"])]
    return "\n".join(lines)


def select_episodes(run: Path, failures: bool, task: str | None, seed: int | None) -> list[Episode]:
    if not run.is_dir():
        raise CliError(ExitCode.NOT_FOUND, f"no such run folder: {run}")
    episodes = [e for e in list_episodes(run) if task in (None, e.task) and seed in (None, e.seed)]
    if not episodes:
        raise CliError(ExitCode.NOT_FOUND, f"no episodes in {run} match the filters")
    return [e for e in episodes if outcome(e) != "success"] if failures else episodes


def view_run(
    run: Path,
    *,
    failures: bool = False,
    task: str | None = None,
    seed: int | None = None,
    limit: int = DEFAULT_LIMIT,
    full: bool = False,
) -> dict[str, Any]:
    episodes = select_episodes(run, failures, task, seed)
    shown, rest = episodes[:limit], episodes[limit:]
    return {
        "run": str(run),
        "matched": len(episodes),
        "traces": [build_trace(e, full) for e in shown],
        "not_shown": [{"task": e.task, "seed": e.seed, "line": summary_line(e)} for e in rest],
    }


def format_view(view: dict[str, Any]) -> str:
    parts = [f"{view['run']}: {view['matched']} episodes matched, {len(view['traces'])} shown"]
    parts += [render_trace(t) for t in view["traces"]]
    if view["not_shown"]:
        parts.append("not shown (narrow with --task/--seed, or raise --limit):")
        parts += [f"  {item['line']}" for item in view["not_shown"]]
    return "\n\n".join(parts)
