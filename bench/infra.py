"""Telling model-server trouble from task failure, and waiting it out.

`LLMRelay` logs every failed model request in ``llm.jsonl`` with an ``upstream`` record:
``kind`` ("http", "unreachable", "timeout", "malformed"), the HTTP ``status``, the parsed
``Retry-After`` seconds and the gateway's error ``code``. That is what `classify` reads (logs
from before that record existed are read from the error message). Before retrying, `probe` asks
the server again with a one-token request.

Some answers are not worth waiting for: the account's daily model allowance is used
(`bench.allowance`), the shared model's daily allowance of idle time is used, or the server asks
to retry later than `MAX_RETRY_AFTER_S`. `wait_until_ready` returns those at once, and
`stop_reason` says what to tell the person and which exit code to use.
"""

from __future__ import annotations

import json
import math
import re
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from harness.llm_relay import chat_url

from .allowance import QUOTA_CODES, used_up
from .exitcodes import ExitCode

PROBE_TIMEOUT_S = 30.0
MAX_RETRY_AFTER_S = 600.0
"""The longest single wait; a server asking for more is reported, not waited for."""
LIMIT_CODES = frozenset({*QUOTA_CODES, "chat_allowance_used"})
MAX_ERROR_BYTES = 65536
_HTTP = re.compile(r"returned HTTP (\d{3})")


@dataclass(frozen=True)
class Probe:
    """One tiny request to the model server. ``status`` is None when it was unreachable."""

    status: int | None
    retry_after: float | None = None
    """As the server sent it; `wait_until_ready` waits at most `MAX_RETRY_AFTER_S` at a time."""
    error: str = ""
    code: str | None = None
    """The gateway's error code (``{"error": {"code": ...}}``), if the answer had one."""
    body: dict | None = None
    """The answer's ``error`` object, if it had one."""

    @property
    def ok(self) -> bool:
        return self.status == 200

    @property
    def transient(self) -> bool:
        """Worth waiting for: unreachable, 429, or any 5xx."""
        return self.status is None or self.status == 429 or self.status >= 500

    @property
    def limited(self) -> bool:
        """Refused by a competition limit that lasts hours: waiting minutes does not help."""
        return self.status in (429, 503) and self.code in LIMIT_CODES

    @property
    def too_long(self) -> bool:
        """A transient answer asking to retry later than bench waits."""
        return self.transient and (self.retry_after or 0) > MAX_RETRY_AFTER_S

    @property
    def stops(self) -> bool:
        return not self.ok and (self.limited or self.too_long)


def _retry_after(value: str | None) -> float | None:
    try:
        seconds = float(value) if value else None
    except ValueError:
        return None  # an HTTP date: treat as absent and use the default wait
    return max(seconds, 0.0) if seconds is not None else None


retry_after_header = _retry_after


def _error_object(raw: bytes) -> dict | None:
    try:
        error = json.loads(raw)["error"]
    except (ValueError, KeyError, TypeError):
        return None
    return error if isinstance(error, dict) else None


def _hours(seconds: float) -> str:
    return f"{seconds / 3600:.1f} h" if seconds >= 3600 else f"{math.ceil(seconds / 60)} minutes"


def stop_reason(result: Probe) -> tuple[ExitCode, str]:
    """Exit code and message for a probe that `stops` a run."""
    if result.code in QUOTA_CODES:
        return ExitCode.LIMITED, used_up(result.body, result.retry_after)
    server = str((result.body or {}).get("message") or result.error)[:500]
    later = "Rerun the same command (same --out) later to finish the run, or use your own model"
    if result.limited:
        return ExitCode.LIMITED, f"{server}. {later} (--model-url)."
    return ExitCode.INFRA, (
        f"the model server asks to retry in {_hours(result.retry_after or 0)} ({result.error}"
        f"{': ' + server if server != result.error else ''}), longer than bench waits "
        f"({_hours(MAX_RETRY_AFTER_S)}); stopping. {later} (--model-url)."
    )


def probe(model_url: str, model: str, api_key: str | None) -> Probe:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    body = {"model": model, "messages": [{"role": "user", "content": "ping"}], "max_tokens": 1}
    request = urllib.request.Request(
        chat_url(model_url), json.dumps(body).encode(), headers, method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=PROBE_TIMEOUT_S) as response:
            return Probe(response.status)
    except urllib.error.HTTPError as e:
        retry = _retry_after(e.headers.get("Retry-After") if e.headers else None)
        try:
            error = _error_object(e.read(MAX_ERROR_BYTES))
        except OSError:
            error = None
        code = error.get("code") if error else None
        code = code if isinstance(code, str) and code else None
        return Probe(e.code, retry, f"HTTP {e.code}" + (f" {code}" if code else ""), code, error)
    except (urllib.error.URLError, OSError) as e:
        return Probe(None, error=f"unreachable: {getattr(e, 'reason', e)}")


def classify(episode_dir: Path) -> str | None:
    """Description of the first infrastructure failure in an episode's model calls, or None.

    Infrastructure failure means the server was unreachable, answered 5xx, or answered 429.
    Other errors (a 400 for a bad request, say) belong to the agent and count as task failure.
    """
    path = episode_dir / "llm.jsonl"
    try:
        entries = [json.loads(line) for line in path.read_text().splitlines()]
    except (OSError, ValueError):
        return None
    for entry in entries:
        error = entry.get("error") or ""
        upstream = entry.get("upstream")
        if isinstance(upstream, dict):
            kind, status = upstream.get("kind"), upstream.get("status")
        else:  # older logs: only the message
            match = _HTTP.search(error)
            kind = "unreachable" if "unreachable" in error else "http" if match else None
            status = int(match[1]) if match else None
        if kind in ("unreachable", "timeout"):
            return error
        if kind == "http" and isinstance(status, int) and (status == 429 or status >= 500):
            return error
    return None


console: TextIO | None = None
"""Where `say` writes instead of ``sys.stderr``: set in quiet worker processes, whose own stderr
goes to ``sim.log``, so that their progress and waiting messages still reach the terminal."""


def say(message: str) -> None:
    print(message, file=console or sys.stderr, flush=True)


def wait_until_ready(
    check: Callable[[], Probe],
    *,
    default_wait_s: float = 10.0,
    max_wait_s: float = 900.0,
    sleep: Callable[[float], None] = time.sleep,
    log: Callable[[str], None] = say,
) -> Probe:
    """Probe until the server answers 200, a non-transient error, an answer that `stops` a run
    (returned at once), or ``max_wait_s`` passes.

    A 429 waits for ``Retry-After``; a 503 with ``Retry-After`` (code ``waking``, or none from
    other servers) means the shared model is waking up. Anything else transient waits
    ``default_wait_s``, doubling each time. Returns the last probe.
    """
    waited, delay = 0.0, default_wait_s
    while True:
        result = check()
        if result.ok or not result.transient or result.stops or waited >= max_wait_s:
            return result
        wait = min(result.retry_after or delay, MAX_RETRY_AFTER_S)
        if result.status == 503 and result.retry_after and result.code in (None, "waking"):
            minutes = math.ceil(wait / 60)
            log(f"waking the shared model, about {minutes} minute{'s' * (minutes != 1)}")
        elif result.status == 429:
            what = f" {result.code}" if result.code else ""
            log(f"rate limited by the model server (429{what}); waiting {wait:.0f}s")
        else:
            log(f"model server trouble ({result.error}); waiting {wait:.0f}s")
        delay = min(delay * 2, 120.0)
        sleep(wait)
        waited += wait
