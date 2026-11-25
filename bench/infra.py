"""Telling model-server trouble from task failure, and waiting it out.

`LLMRelay` logs every failed model request in ``llm.jsonl`` with an ``upstream`` record:
``kind`` ("http", "unreachable", "timeout", "malformed"), the HTTP ``status`` and the parsed
``Retry-After`` seconds. That is what `classify` reads (logs from before that record existed
are read from the error message). Before retrying, `probe` asks the server again with a
one-token request.
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

from harness.llm_relay import chat_url

PROBE_TIMEOUT_S = 30.0
MAX_RETRY_AFTER_S = 600.0
_HTTP = re.compile(r"returned HTTP (\d{3})")


@dataclass(frozen=True)
class Probe:
    """One tiny request to the model server. ``status`` is None when it was unreachable."""

    status: int | None
    retry_after: float | None = None
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.status == 200

    @property
    def transient(self) -> bool:
        """Worth waiting for: unreachable, 429, or any 5xx."""
        return self.status is None or self.status == 429 or self.status >= 500


def _retry_after(value: str | None) -> float | None:
    try:
        seconds = float(value) if value else None
    except ValueError:
        return None  # an HTTP date: treat as absent and use the default wait
    return min(max(seconds, 0.0), MAX_RETRY_AFTER_S) if seconds is not None else None


retry_after_header = _retry_after


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
        return Probe(e.code, _retry_after(e.headers.get("Retry-After")), f"HTTP {e.code}")
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


def say(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def wait_until_ready(
    check: Callable[[], Probe],
    *,
    default_wait_s: float = 10.0,
    max_wait_s: float = 900.0,
    sleep: Callable[[float], None] = time.sleep,
    log: Callable[[str], None] = say,
) -> Probe:
    """Probe until the server answers 200, a non-transient error, or ``max_wait_s`` passes.

    A 429 waits for ``Retry-After``; a 503 with ``Retry-After`` means the shared model is
    waking up. Anything else transient waits ``default_wait_s``, doubling each time. Returns the
    last probe.
    """
    waited, delay = 0.0, default_wait_s
    while True:
        result = check()
        if result.ok or not result.transient or waited >= max_wait_s:
            return result
        wait = result.retry_after or delay
        if result.status == 503 and result.retry_after:
            minutes = math.ceil(wait / 60)
            log(f"waking the shared model, about {minutes} minute{'s' * (minutes != 1)}")
        elif result.status == 429:
            log(f"rate limited by the model server (429); waiting {wait:.0f}s")
        else:
            log(f"model server trouble ({result.error}); waiting {wait:.0f}s")
        delay = min(delay * 2, 120.0)
        sleep(wait)
        waited += wait
