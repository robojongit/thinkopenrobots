"""The shared model's daily allowance of tokens per account: reading it and saying it plainly.

The gateway counts prompt plus completion tokens per account and UTC day; past the allowance
every model request answers 429 with code ``model_allowance_used`` (older gateways:
``quota_exceeded``) until 00:00 UTC. Waiting a few minutes never helps, so `bench` stops at once
(exit 8, ``limited``) and says when the allowance is back.

Newer gateways describe the allowance as ``{"model": {"tokens_per_day", "tokens_used",
"resets_at", "concurrent_requests"}}`` in ``GET /v1/me`` and ``GET /v1/submissions``, and in the
429's error body (``tokens_used``, ``tokens_per_day``, ``resets_at``). Older ones have only
``tokens_today`` and ``tokens_per_day`` in ``/v1/me``; every field may be missing.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from typing import Any

QUOTA_CODES = frozenset({"model_allowance_used", "quota_exceeded"})
"""Error codes of a model request refused because the account's daily allowance is used."""


def _int(value: Any) -> int | None:
    return value if type(value) is int and value >= 0 else None


def next_utc_midnight(now: datetime) -> datetime:
    return (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)


def _time(text: Any) -> datetime | None:
    if not isinstance(text, str):
        return None
    try:
        when = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return when if when.tzinfo else None


def from_answer(data: Any) -> dict[str, Any] | None:
    """``{tokens_per_day, tokens_used, resets_at, concurrent_requests}`` from a gateway answer
    (missing values are None; ``resets_at`` defaults to the next 00:00 UTC), or None if the
    answer does not give the allowance."""
    if not isinstance(data, dict):
        return None
    model = data.get("model") if isinstance(data.get("model"), dict) else {}
    per_day = _int(model.get("tokens_per_day", data.get("tokens_per_day")))
    used = _int(model.get("tokens_used", data.get("tokens_today")))
    if per_day is None:
        return None
    resets = _time(model.get("resets_at")) or next_utc_midnight(datetime.now(UTC))
    return {
        "tokens_per_day": per_day,
        "tokens_used": used,
        "resets_at": resets.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "concurrent_requests": _int(model.get("concurrent_requests")),
    }


def left(allowance: dict[str, Any]) -> int | None:
    per_day, used = allowance.get("tokens_per_day"), allowance.get("tokens_used")
    return None if per_day is None or used is None else max(per_day - used, 0)


def _in(seconds: float) -> str:
    if seconds < 3600:
        return f"{max(1, math.ceil(seconds / 60))} min"
    return f"{seconds / 3600:.1f} h"


def resets_text(resets_at: str | datetime | None, now: datetime | None = None) -> str:
    """``00:00 UTC (in 14.5 h)``."""
    now = now or datetime.now(UTC)
    when = resets_at if isinstance(resets_at, datetime) else _time(resets_at)
    when = when or next_utc_midnight(now)
    day = "" if when.date() <= (now + timedelta(days=1)).date() else f"{when:%Y-%m-%d} "
    return f"{day}{when.astimezone(UTC):%H:%M} UTC (in {_in((when - now).total_seconds())})"


def _n(value: int | None) -> str:
    return "?" if value is None else f"{value:,}"


def describe(allowance: dict[str, Any], now: datetime | None = None) -> str:
    """The line `bench status` and `bench train` print."""
    text = (
        f"model tokens today: {_n(allowance.get('tokens_used'))} of "
        f"{_n(allowance.get('tokens_per_day'))} used, resets at "
        f"{resets_text(allowance.get('resets_at'), now)}"
    )
    if allowance.get("concurrent_requests"):
        text += f"; at most {allowance['concurrent_requests']} model requests at a time"
    return text


AFTER = (
    " Finished episodes are kept: rerun the same command (same --out) after the reset to finish"
    " the run, or use your own model (--model-url). Retrying before the reset does not help."
)


def used_up(
    error: dict[str, Any] | None, retry_after: float | None, now: datetime | None = None
) -> str:
    """The message for a model request refused by the daily allowance, from the 429's error
    body (newer gateways give the numbers) and its ``Retry-After``."""
    now = now or datetime.now(UTC)
    error = error or {}
    resets = _time(error.get("resets_at"))
    if resets is None and retry_after:
        resets = now + timedelta(seconds=retry_after)
    used, per_day = _int(error.get("tokens_used")), _int(error.get("tokens_per_day"))
    amount = f": {_n(used)} of {_n(per_day)} tokens" if used is not None or per_day else ""
    return f"daily model allowance used{amount}, resets at {resets_text(resets, now)}." + AFTER
