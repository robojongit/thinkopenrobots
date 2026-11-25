"""A small client for the competition's gateway API (login, submissions, picks, remote runs).

Requests go only to the configured gateway URL; redirects are never followed, so the key cannot
be carried to another host. Errors arrive as ``{"error": {"code", "message"}}`` and become
`GatewayError`; `cli_error` turns them into the exit codes of `bench.exitcodes`.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlencode

from .exitcodes import CliError, ExitCode
from .infra import retry_after_header

TIMEOUT_S = 60.0
MAX_JSON_BYTES = 4 * 1024 * 1024
LIMIT_CODES = {"quota_exceeded", "already_queued", "window_closed", "not_pickable"}
SETUP_CODES = {"invalid_archive", "unauthorized", "account_disabled"}


class GatewayError(Exception):
    def __init__(
        self, status: int | None, code: str, message: str, retry_after: float | None = None
    ) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.retry_after = retry_after


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        return None  # a 3xx then surfaces as an HTTPError


_OPENER = urllib.request.build_opener(_NoRedirect)


class Gateway:
    def __init__(self, base_url: str, key: str | None = None, timeout: float = TIMEOUT_S) -> None:
        self.base = base_url.rstrip("/")
        self.key = key
        self.timeout = timeout

    def _request(
        self,
        method: str,
        path: str,
        *,
        body: Any = None,
        data: bytes | None = None,
        params: dict[str, str] | None = None,
        auth: bool = True,
    ):
        url = self.base + path + (f"?{urlencode(params)}" if params else "")
        headers = {"Accept": "application/json", "User-Agent": "bench"}
        if body is not None:
            data = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
        elif data is not None:
            headers["Content-Type"] = "application/gzip"
        if auth:
            if not self.key:
                raise GatewayError(401, "unauthorized", "not logged in: run bench login")
            headers["Authorization"] = f"Bearer {self.key}"
        request = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            return _OPENER.open(request, timeout=self.timeout)
        except urllib.error.HTTPError as e:
            raise _error_from(e) from None
        except (urllib.error.URLError, OSError) as e:
            raise GatewayError(
                None, "unreachable", f"gateway unreachable: {getattr(e, 'reason', e)}"
            ) from None

    def call(self, method: str, path: str, **kw: Any) -> dict[str, Any]:
        with self._request(method, path, **kw) as response:
            raw = response.read(MAX_JSON_BYTES + 1)
        try:
            data = json.loads(raw)
        except ValueError:
            raise GatewayError(response.status, "bad_answer", "gateway answered non-JSON") from None
        if len(raw) > MAX_JSON_BYTES or not isinstance(data, dict):
            raise GatewayError(response.status, "bad_answer", "gateway answer unusable")
        return data

    def download(self, path: str, dest: Path, max_bytes: int) -> int:
        """Stream a file to ``dest``; refuses more than ``max_bytes``."""
        written = 0
        with self._request("GET", path) as response, dest.open("wb") as out:
            while chunk := response.read(1 << 16):
                written += len(chunk)
                if written > max_bytes:
                    raise GatewayError(None, "too_large", f"download larger than {max_bytes} bytes")
                out.write(chunk)
        return written

    # --- the API -----------------------------------------------------------------------------
    def login_start(self) -> dict[str, Any]:
        return self.call("POST", "/v1/login/start", auth=False)

    def login_poll(self, login_id: str) -> dict[str, Any]:
        return self.call("POST", "/v1/login/poll", body={"login_id": login_id}, auth=False)

    def submit(self, archive: bytes) -> dict[str, Any]:
        return self.call("POST", "/v1/submissions", data=archive)

    def submissions(self) -> dict[str, Any]:
        return self.call("GET", "/v1/submissions")

    def submission(self, sid: str) -> dict[str, Any]:
        return self.call("GET", f"/v1/submissions/{quote(sid, safe='')}")

    def picks(self) -> dict[str, Any]:
        return self.call("GET", "/v1/picks")

    def pick(self, ids: list[str]) -> dict[str, Any]:
        return self.call("POST", "/v1/picks", body={"ids": ids})

    def train(self, archive: bytes, plan: dict[str, list[int]]) -> dict[str, Any]:
        params = {"plan": json.dumps(plan, separators=(",", ":"))}
        return self.call("POST", "/v1/train", data=archive, params=params)

    def train_run(self, tid: str) -> dict[str, Any]:
        return self.call("GET", f"/v1/train/{quote(tid, safe='')}")

    def train_results(self, tid: str, dest: Path, max_bytes: int) -> int:
        return self.download(f"/v1/train/{quote(tid, safe='')}/results", dest, max_bytes)


def _error_from(e: urllib.error.HTTPError) -> GatewayError:
    retry = retry_after_header(e.headers.get("Retry-After") if e.headers else None)
    code, message = "http_error", f"gateway answered HTTP {e.code}"
    try:
        err = json.loads(e.read(65536))["error"]
        code, message = str(err["code"]), str(err["message"])
    except (ValueError, KeyError, TypeError, OSError):
        pass
    return GatewayError(e.code, code, message, retry)


def cli_error(e: GatewayError) -> CliError:
    """Exit code for a gateway error: limits and rules -> LIMITED, setup -> NOT_READY,
    unknown ids -> NOT_FOUND, server trouble -> INFRA, a bad request -> USAGE."""
    when = f" (retry after {e.retry_after:.0f}s)" if e.retry_after else ""
    if e.code in LIMIT_CODES:
        return CliError(ExitCode.LIMITED, e.message)
    if e.code in SETUP_CODES or e.status == 401:
        fix = " Run: bench login" if e.status == 401 else ""
        return CliError(ExitCode.NOT_READY, e.message + fix)
    if e.status == 404:
        return CliError(ExitCode.NOT_FOUND, e.message)
    if e.code == "not_configured":
        return CliError(ExitCode.NOT_AVAILABLE, e.message)
    if e.status is None or e.status >= 500 or e.status == 429:
        return CliError(ExitCode.INFRA, e.message + when)
    return CliError(ExitCode.USAGE, e.message)
