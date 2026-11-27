"""LLM relay: the only route from the agent to a model.

Forwards ``llm.chat`` to an OpenAI-compatible ``/v1/chat/completions`` endpoint with the model
name forced, request fields whitelisted, and a per-request seed derived from (episode seed, call
index). Enforces the per-episode budget (model calls, total prompt + completion tokens, wall
clock) and logs every reply.

Before forwarding, the prompt's tokens are bounded from above without a tokenizer (one per UTF-8
byte, plus the chat template; see ``estimate_prompt_tokens``). A call whose bound does not fit
in the tokens left, or in the model's context, is refused, and ``max_tokens`` is capped so that
bound + ``max_tokens`` fits in both: the budget can no longer be overshot by a long prompt.

A request still unanswered when the episode's time runs out is refused as a "wall" limit, not
logged as a model-server failure. A 429 with a short ``Retry-After`` (at most
``SHORT_RETRY_AFTER_S``: too many requests at once, say from two runs on one account) is waited
out and sent again while the episode has time; the agent sees only the reply, and its log entry
says how long it waited. Any other forwarded request that fails (HTTP error, timeout, unusable
reply) is free for the first
``FREE_FAILED_CALLS`` of an episode and counts as a model call after that, so failures cannot
make the model server work without limit. Failed requests are logged without their prompt.

Every request carries the episode's random ``cache_salt`` (vLLM >= 0.9): prefix-cache hits only
happen between requests with the same salt, so an agent cannot time the shared server to learn
what prompts other episodes sent. Servers that do not know the field ignore it (vLLM before 0.9
logs and ignores unknown fields) or, if strict (OpenAI's own API), reject it with a 400: pass
``cache_salt=False`` for those.
"""

from __future__ import annotations

import email.utils
import hashlib
import json
import math
import secrets
import socket
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from harness.rpc import RpcError

DEFAULT_MODEL = "inkling-2-mini"
DEFAULT_MAX_TOKENS = 20480
DEFAULT_CONTEXT_LENGTH = 65536
TEMPLATE_TOKENS = 64
TEMPLATE_TOKENS_PER_MESSAGE = 8
FREE_FAILED_CALLS = 3
MAX_MESSAGES = 200
MAX_PROMPT_CHARS = 1_000_000
MAX_RESPONSE_BYTES = 8 * 1024 * 1024
ROLES = frozenset({"system", "user", "assistant"})

# Whitelisted request fields: name -> (type, min, max).
OPTIONS: dict[str, tuple[type, float, float]] = {
    "temperature": (float, 0.0, 2.0),
    "top_p": (float, 0.0, 1.0),
    "max_tokens": (int, 1, 1_000_000),
    "presence_penalty": (float, -2.0, 2.0),
    "frequency_penalty": (float, -2.0, 2.0),
}
MAX_STOP = 4
MAX_STOP_LEN = 64
SHORT_RETRY_AFTER_S = 30.0
"""A 429 asking to retry within this many seconds is waited out inside the episode."""
MAX_ERROR_BYTES = 65536


def retry_after_seconds(value: str | None, now: float | None = None) -> float | None:
    """``Retry-After`` as seconds (delta-seconds or an HTTP date), or None if absent/invalid."""
    if not value:
        return None
    value = value.strip()
    try:
        return max(float(value), 0.0)
    except ValueError:
        pass
    try:
        when = email.utils.parsedate_to_datetime(value).timestamp()
    except (TypeError, ValueError, OverflowError):
        return None
    return max(when - (time.time() if now is None else now), 0.0)


def _upstream(
    message: str,
    kind: str,
    status: int | None = None,
    retry: float | None = None,
    code: str | None = None,
):
    """An ``upstream`` RpcError whose data says what happened: ``kind`` is "http" (non-2xx),
    "unreachable", "timeout" or "malformed"; ``status`` the HTTP status if one was received;
    ``retry_after`` the parsed ``Retry-After`` in seconds; ``code``, when the server's error
    body had one (``{"error": {"code": ...}}``, as the competition gateway sends)."""
    data: dict[str, Any] = {"kind": kind, "status": status, "retry_after": retry}
    if code:
        data["code"] = code
    return RpcError("upstream", message, data=data)


def error_code(raw: bytes) -> str | None:
    """``error.code`` of an error body ``{"error": {"code": ...}}``, if it has a short one."""
    try:
        code = json.loads(raw)["error"]["code"]
    except (ValueError, KeyError, TypeError):
        return None
    return code if isinstance(code, str) and 0 < len(code) <= 64 else None


def _short_wait(error: RpcError) -> float | None:
    """Seconds to wait before sending again, for a 429 with a short ``Retry-After``."""
    data = error.data or {}
    retry = data.get("retry_after")
    if data.get("kind") == "http" and data.get("status") == 429 and retry:
        return retry if 0 < retry <= SHORT_RETRY_AFTER_S else None
    return None


@dataclass(frozen=True)
class Budget:
    calls: int = 10
    tokens: int = 66_000
    wall_s: float = 400.0


MAX_LOG_BYTES = 16 * 1024 * 1024


class LogBudget:
    """Bytes of log one episode may keep: transcript events, run records, harness errors and
    model-request records, measured as JSON. Once an entry does not fit, it and every later one
    is only counted, so the first entries (the ones worth reading) survive and the transcript
    and llm.jsonl stay in step."""

    def __init__(self, limit: int = MAX_LOG_BYTES) -> None:
        self.limit = limit
        self.left = limit
        self.dropped: dict[str, int] = {}

    def admit(self, kind: str, entry: Any) -> bool:
        size = len(json.dumps(entry, default=str))
        if self.dropped or size > self.left:
            self.dropped[kind] = self.dropped.get(kind, 0) + 1
            return False
        self.left -= size
        return True


def request_seed(episode_seed: int, call_index: int) -> int:
    """Sampling seed for model call ``call_index`` (0-based) of an episode. One-way, 31 bits."""
    digest = hashlib.sha256(f"inkling-robo:{episode_seed}:{call_index}".encode()).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFF_FFFF


def estimate_prompt_tokens(messages: list[dict[str, Any]]) -> int:
    """An upper bound on the prompt's token count: one token per UTF-8 byte of text (a
    byte-level BPE token covers at least one byte) plus a margin for the chat template.

    Real prompts take ~3-4x fewer tokens. The model server's ``/tokenize`` endpoint (vLLM has
    one) could replace this with an exact count, at one extra request per call."""
    total = TEMPLATE_TOKENS
    for m in messages:
        parts = (
            [m["content"]] if isinstance(m["content"], str) else [p["text"] for p in m["content"]]
        )
        total += TEMPLATE_TOKENS_PER_MESSAGE + sum(len(p.encode()) for p in parts)
    return total


def chat_url(model_url: str) -> str:
    url = model_url.rstrip("/")
    return url if url.endswith("/chat/completions") else url + "/chat/completions"


def _check_messages(messages: Any) -> list[dict[str, Any]]:
    if not isinstance(messages, list) or not 1 <= len(messages) <= MAX_MESSAGES:
        raise RpcError("bad_request", f"messages must be a list of 1..{MAX_MESSAGES} messages")
    total = 0
    clean = []
    for m in messages:
        if not isinstance(m, dict) or set(m) != {"role", "content"}:
            raise RpcError("bad_request", "each message must be {role, content}")
        if not isinstance(m["role"], str) or m["role"] not in ROLES:
            raise RpcError("bad_request", f"role must be one of {sorted(ROLES)}")
        content = m["content"]
        if isinstance(content, str):
            total += len(content)
        elif isinstance(content, list) and content:
            for part in content:
                if (
                    not isinstance(part, dict)
                    or set(part) != {"type", "text"}
                    or part["type"] != "text"
                    or not isinstance(part["text"], str)
                ):
                    raise RpcError("bad_request", "content parts must be {type: 'text', text}")
                total += len(part["text"])
        else:
            raise RpcError("bad_request", "content must be a string or a list of text parts")
        clean.append({"role": m["role"], "content": content})
    if total > MAX_PROMPT_CHARS:
        raise RpcError("bad_request", "prompt too long")
    return clean


def _check_options(options: Any) -> dict[str, Any]:
    if not isinstance(options, dict):
        raise RpcError("bad_request", "options must be a map")
    clean: dict[str, Any] = {}
    for key, value in options.items():
        if key == "stop":
            stops = [value] if isinstance(value, str) else value
            if (
                not isinstance(stops, list)
                or len(stops) > MAX_STOP
                or not all(isinstance(s, str) and 0 < len(s) <= MAX_STOP_LEN for s in stops)
            ):
                raise RpcError("bad_request", f"stop must be up to {MAX_STOP} short strings")
            clean["stop"] = stops
            continue
        if key not in OPTIONS:
            allowed = sorted([*OPTIONS, "stop"])
            raise RpcError("bad_request", f"option {key!r} not allowed; allowed: {allowed}")
        typ, lo, hi = OPTIONS[key]
        ok_type = (
            type(value) is int
            if typ is int
            else type(value) in (int, float) and math.isfinite(value)
        )
        if not ok_type or not lo <= value <= hi:
            raise RpcError("bad_request", f"option {key!r} must be {typ.__name__} in [{lo}, {hi}]")
        clean[key] = typ(value)
    return clean


class LLMRelay:
    """One episode's relay. ``chat()`` raises ``RpcError`` (codes: bad_request, budget,
    upstream) for anything it will not or cannot forward."""

    def __init__(
        self,
        model_url: str,
        episode_seed: int,
        budget: Budget,
        deadline: float,
        *,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        log_budget: LogBudget | None = None,
        context_length: int = DEFAULT_CONTEXT_LENGTH,
        cache_salt: bool = True,
    ) -> None:
        self.url = chat_url(model_url)
        self._cache_salt = secrets.token_hex(16) if cache_salt else None
        self.context_length = context_length
        """The model's context window (prompt + completion), as the model server enforces it."""
        self.log_budget = log_budget or LogBudget()
        """Shared with the sim server: one byte budget for everything the episode logs."""
        self.model = model
        self.budget = budget
        self.deadline = deadline  # time.monotonic() value
        self._episode_seed = episode_seed
        self._api_key = api_key
        self._lock = threading.Lock()
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.upstream_errors = 0
        """Failed forwarded requests; past ``FREE_FAILED_CALLS`` each is also in ``calls``."""
        self.upstream_failures: list[dict[str, Any]] = []
        """One entry per failed model request: index, kind, status, retry_after."""
        self.limit_hit: str | None = None  # "calls" | "tokens" | "wall"
        self.log: list[dict[str, Any]] = []
        self.forwarded = 0
        """Requests forwarded to the model server; each one's log entry has it as ``seq``."""
        self._replies: list[str] = []

    @property
    def tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def _refuse(self, limit: str, message: str) -> RpcError:
        self.limit_hit = self.limit_hit or limit
        return RpcError("budget", message)

    def chat(self, messages: Any, options: Any) -> dict[str, Any]:
        with self._lock:
            clean_messages = _check_messages(messages)
            clean_options = _check_options(options)
            remaining_s = self.deadline - time.monotonic()
            if remaining_s <= 0:
                raise self._refuse("wall", "episode time limit reached")
            if self.calls >= self.budget.calls:
                raise self._refuse("calls", f"model call budget ({self.budget.calls}) spent")
            remaining_tokens = self.budget.tokens - self.tokens
            if remaining_tokens <= 0:
                raise self._refuse("tokens", f"token budget ({self.budget.tokens}) spent")

            estimate = estimate_prompt_tokens(clean_messages)
            if estimate >= self.context_length:
                raise RpcError(
                    "bad_request",
                    f"prompt too long: up to {estimate} tokens by the relay's bound (one per UTF-8 "
                    f"byte), and the model's context is {self.context_length}",
                )
            if estimate >= remaining_tokens:
                raise self._refuse(
                    "tokens",
                    f"prompt of up to {estimate} tokens (one per UTF-8 byte) does not fit in the "
                    f"{remaining_tokens} tokens left of the budget ({self.budget.tokens})",
                )

            index = self.calls
            max_tokens = min(
                clean_options.pop("max_tokens", DEFAULT_MAX_TOKENS),
                remaining_tokens - estimate,
                self.context_length - estimate,
            )
            payload = {
                **clean_options,
                "model": self.model,
                "messages": clean_messages,
                "max_tokens": max_tokens,
                "seed": request_seed(self._episode_seed, index),
                "n": 1,
                "stream": False,
            }
            if self._cache_salt:
                payload["cache_salt"] = self._cache_salt
            entry: dict[str, Any] = {
                "seq": self.forwarded,
                "index": index,
                "request": {
                    k: v for k, v in payload.items() if k not in ("seed", "model", "cache_salt")
                },
            }
            self.forwarded += 1
            started = time.monotonic()
            waited = 0.0
            try:
                while True:
                    try:
                        body = self._post(payload, timeout=self.deadline - time.monotonic())
                        break
                    except RpcError as e:
                        wait = _short_wait(e)
                        if wait is None or time.monotonic() + wait >= self.deadline:
                            raise
                        time.sleep(wait)
                        waited += wait
                if waited:
                    entry["rate_limited_s"] = round(waited, 1)
                reply = _parse_reply(body)
            except RpcError as e:
                if (e.data or {}).get("kind") == "timeout" and time.monotonic() >= self.deadline:
                    # Cut off by the episode's own time limit (a long reply, say all tokens
                    # spent on reasoning): the budget's doing, not the model server's.
                    entry.update(
                        error="episode time limit reached while waiting for the model",
                        seconds=round(time.monotonic() - started, 3),
                    )
                    self._log(entry)
                    raise self._refuse("wall", "episode time limit reached") from None
                self.upstream_errors += 1
                if self.upstream_errors > FREE_FAILED_CALLS:
                    self.calls += 1
                upstream = e.data or {"kind": "malformed", "status": None, "retry_after": None}
                self.upstream_failures.append({"index": index, **upstream})
                entry["request"]["messages"] = {
                    "count": len(clean_messages),
                    "tokens_bound": estimate,
                    "kept": False,
                }
                entry.update(
                    error=e.message,
                    upstream=upstream,
                    seconds=round(time.monotonic() - started, 3),
                )
                self._log(entry)
                raise
            prompt, completion = reply.pop("usage")
            if prompt is None or completion is None:
                # No usage from the server: charge the worst case so the budget still holds.
                prompt, completion = estimate, max_tokens
            self.calls += 1
            self.prompt_tokens += prompt
            self.completion_tokens += completion
            reply.update(prompt_tokens=prompt, completion_tokens=completion)
            self._replies.append(reply["content"])
            entry.update(reply=reply, seconds=round(time.monotonic() - started, 3))
            self._log(entry)
            return reply

    def _log(self, entry: dict[str, Any]) -> None:
        if self.log_budget.admit("model requests", entry):
            self.log.append(entry)

    def _post(self, payload: dict[str, Any], timeout: float) -> Any:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        req = urllib.request.Request(
            self.url, data=json.dumps(payload).encode(), headers=headers, method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=max(timeout, 0.1)) as resp:
                status = resp.status
                raw = resp.read(MAX_RESPONSE_BYTES + 1)
        except urllib.error.HTTPError as e:
            retry = retry_after_seconds(e.headers.get("Retry-After") if e.headers else None)
            try:
                code = error_code(e.read(MAX_ERROR_BYTES))
            except OSError:
                code = None
            message = f"model server returned HTTP {e.code}" + (f" ({code})" if code else "")
            raise _upstream(message, "http", e.code, retry, code) from None
        except TimeoutError as e:
            raise _upstream(f"model server timed out: {e}", "timeout") from None
        except urllib.error.URLError as e:
            kind = "timeout" if isinstance(e.reason, (TimeoutError, socket.timeout)) else None
            message = f"model server {'timed out' if kind else 'unreachable'}: {e}"
            raise _upstream(message, kind or "unreachable") from None
        except OSError as e:
            raise _upstream(f"model server unreachable: {e}", "unreachable") from None
        if len(raw) > MAX_RESPONSE_BYTES:
            raise _upstream("model response too large", "malformed", status)
        try:
            return json.loads(raw)
        except ValueError:
            raise _upstream("model response is not JSON", "malformed", status) from None

    def is_substring_of_reply(self, code: str) -> bool:
        """True if ``code`` is non-blank and an exact substring of the content of a model
        reply from this episode."""
        if not code.strip():
            return False
        with self._lock:
            return any(code in reply for reply in self._replies)


def _parse_reply(body: Any) -> dict[str, Any]:
    try:
        choice = body["choices"][0]
        message = choice["message"]
        content = message.get("content") or ""
        reasoning = message.get("reasoning") or message.get("reasoning_content")
        finish_reason = choice.get("finish_reason")
        if not isinstance(content, str) or not isinstance(reasoning, (str, type(None))):
            raise TypeError
        usage = body.get("usage") or {}
        prompt, completion = usage.get("prompt_tokens"), usage.get("completion_tokens")
        if not all(isinstance(x, int) and x >= 0 for x in (prompt, completion)):
            prompt = completion = None
    except (KeyError, IndexError, TypeError, AttributeError):
        raise _upstream("malformed model response", "malformed", 200) from None
    return {
        "content": content,
        "reasoning": reasoning,
        "finish_reason": finish_reason if isinstance(finish_reason, str) else None,
        "usage": (prompt, completion),
    }
