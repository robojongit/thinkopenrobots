"""Exit codes shared by every command. The table also appears in ``bench --help``."""

from __future__ import annotations

from enum import IntEnum


class ExitCode(IntEnum):
    OK = 0
    FAILED = 1
    USAGE = 2
    NOT_READY = 3
    INFRA = 4
    CRASHED = 5
    NOT_FOUND = 6
    NOT_AVAILABLE = 7
    LIMITED = 8
    INTERRUPTED = 130


MEANING: dict[ExitCode, str] = {
    ExitCode.OK: "success (a task failing is not an error; read the scores)",
    ExitCode.FAILED: "unexpected internal error",
    ExitCode.USAGE: "bad arguments or options",
    ExitCode.NOT_READY: "setup problem: a doctor check failed, or agent/ or model URL unusable",
    ExitCode.INFRA: "model server unreachable, 5xx or 429 for longer than bench waits; the run is"
    " incomplete, run it again",
    ExitCode.CRASHED: "episodes crashed in the harness (not task failures); see crash.json",
    ExitCode.NOT_FOUND: "run folder, episode or shared (task, seed) pairs not found",
    ExitCode.NOT_AVAILABLE: "needs the shared server, which is not configured or not available",
    ExitCode.LIMITED: "refused by a competition limit (daily submissions or model tokens, one"
    " queued, pick window); the message says when to retry; do not retry before",
    ExitCode.INTERRUPTED: "interrupted (Ctrl-C); a rerun with the same --out resumes",
}


class CliError(Exception):
    """Raised by a command to stop with a message on stderr and a given exit code."""

    def __init__(self, code: ExitCode, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def table() -> str:
    return "\n".join(f"  {int(c):>3}  {c.name.lower():<14} {MEANING[c]}" for c in ExitCode)
