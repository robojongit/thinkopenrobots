"""A small stand-in for the competition gateway, implementing the documented HTTP API closely
enough for `bench`'s tests: device-flow login (approved through ``POST /dev/approve``, as a
person would on github.com), submissions with the 3-per-24-hours and one-queued rules, status,
picks, remote training runs with downloadable episode folders, and a chat endpoint that can
pretend the shared model is waking up. The real server lives in the private ops repo; the tests
run against it too when ``INKLING_OPS_DIR`` points at that repo.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import secrets
import tarfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse


class State:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.logins: dict[str, dict[str, Any]] = {}
        self.keys: dict[str, str] = {}  # key -> login
        self.jobs: list[dict[str, Any]] = []
        self.picks: dict[str, list[str]] = {}
        self.window_open = False
        self.waking = 0  # model requests still to answer 503 "waking"
        self.auth_seen: list[str | None] = []
        self.uploads: list[bytes] = []


MAX_VIDEOS = 10
FAKE_MP4 = b"\x00\x00\x00\x18ftypisom\x00\x00\x02\x00isomiso2" + b"\x00" * 64


def check_video_plan(plan: Any) -> tuple[dict[str, list[int]], dict[str, list[int]]] | str:
    """(seeds, video) from either plan shape, or the reason it is refused."""
    if not isinstance(plan, dict) or "version" not in plan:
        return plan, {}
    if set(plan) != {"version", "seeds", "video"} or plan["version"] != 2:
        return "want version 2, seeds and video"
    seeds, video = plan["seeds"], plan["video"]
    if not isinstance(seeds, dict) or not isinstance(video, dict):
        return "seeds and video must be objects"
    for task, chosen in video.items():
        if not isinstance(chosen, list) or not chosen:
            return f"video of {task}: want a non-empty list"
        if any(s not in seeds.get(task, []) for s in chosen) or len(set(chosen)) != len(chosen):
            return f"video of {task}: not planned episodes"
    if sum(map(len, video.values())) > MAX_VIDEOS:
        return f"a remote run records at most {MAX_VIDEOS}"
    return seeds, video


def episode_results(plan: dict[str, list[int]], video: dict[str, list[int]] | None = None) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for task, seeds in plan.items():
            for seed in seeds:
                if seed in (video or {}).get(task, []):
                    info = tarfile.TarInfo(f"./{task}/seed_{seed}/video.mp4")
                    info.size = len(FAKE_MP4)
                    tar.addfile(info, io.BytesIO(FAKE_MP4))
                result = {
                    "task": task,
                    "success": seed % 2 == 0,
                    "tokens": 900 + seed % 100,
                    "calls": 2,
                    "seconds": 11.0,
                    "ended": "returned",
                }
                llm = {
                    "seq": 0,
                    "index": 0,
                    "reply": {
                        "content": "print('hi')",
                        "prompt_tokens": 500,
                        "completion_tokens": 400,
                    },
                }
                for name, data in [
                    ("result.json", json.dumps(result)),
                    ("transcript.jsonl", '{"kind": "llm", "seq": 0, "ok": true}\n'),
                    ("llm.jsonl", json.dumps(llm) + "\n"),
                ]:
                    raw = data.encode()
                    info = tarfile.TarInfo(f"./{task}/seed_{seed}/{name}")
                    info.size = len(raw)
                    tar.addfile(info, io.BytesIO(raw))
    return buf.getvalue()


class FakeGateway:
    def __init__(self) -> None:
        self.state = State()
        state = self.state

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:
                pass

            def _send(
                self,
                status: int,
                body: Any,
                headers: dict[str, str] | None = None,
                raw: bytes | None = None,
                ctype: str = "application/json",
            ) -> None:
                data = raw if raw is not None else json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(data)))
                for k, v in (headers or {}).items():
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(data)

            def _error(self, status: int, code: str, message: str, retry: int | None = None):
                headers = {"Retry-After": str(retry)} if retry else None
                self._send(status, {"error": {"code": code, "message": message}}, headers)

            def _body(self) -> bytes:
                return self.rfile.read(int(self.headers.get("Content-Length") or 0))

            def _account(self) -> str | None:
                auth = self.headers.get("Authorization")
                state.auth_seen.append(auth)
                key = (auth or "").removeprefix("Bearer ")
                login = state.keys.get(key)
                if login is None:
                    self._error(401, "unauthorized", "unknown gateway key: run bench login again")
                return login

            def _view(self, job: dict[str, Any]) -> dict[str, Any]:
                queued = [
                    j for j in state.jobs if j["kind"] == job["kind"] and j["state"] == "queued"
                ]
                view = {k: v for k, v in job.items() if k not in ("account", "kind", "plan_raw")}
                view["position"] = queued.index(job) + 1 if job in queued else None
                if job["state"] == "queued":
                    view["box"] = {"state": "waking", "eta_s": 240}
                return view

            def do_POST(self) -> None:  # noqa: N802
                url = urlparse(self.path)
                path, query = url.path, parse_qs(url.query)
                body = self._body()
                with state.lock:
                    if path == "/v1/login/start":
                        login_id = secrets.token_urlsafe(32)
                        code = f"{secrets.randbelow(9000) + 1000}-{secrets.randbelow(9000) + 1000}"
                        state.logins[login_id] = {"code": code, "login": None, "used": False}
                        return self._send(
                            200,
                            {
                                "login_id": login_id,
                                "user_code": code,
                                "verification_uri": "https://github.com/login/device",
                                "expires_in": 900,
                                "interval": 1,
                            },
                        )
                    if path == "/dev/approve":
                        data = json.loads(body)
                        for entry in state.logins.values():
                            if entry["code"] == data["user_code"]:
                                entry["login"] = data["login"]
                                return self._send(200, {"ok": True})
                        return self._error(404, "not_found", "unknown user code")
                    if path == "/v1/login/poll":
                        entry = state.logins.get(json.loads(body).get("login_id", ""))
                        if entry is None:
                            return self._error(
                                404, "unknown_login", "unknown login; run bench login"
                            )
                        if entry["used"]:
                            return self._error(
                                410, "login_used", "this login already produced a key"
                            )
                        if entry["login"] is None:
                            return self._send(200, {"status": "pending", "interval": 1})
                        entry["used"] = True
                        key = "ink_" + secrets.token_urlsafe(32)
                        state.keys = {k: v for k, v in state.keys.items() if v != entry["login"]}
                        state.keys[key] = entry["login"]
                        return self._send(
                            200, {"status": "ok", "account": entry["login"], "key": key}
                        )
                    if path == "/v1/chat/completions":
                        if self._account() is None:
                            return None
                        if state.waking:
                            state.waking -= 1
                            return self._error(
                                503, "waking", "waking the shared model, about 1 minute", 1
                            )
                        return self._send(
                            200,
                            {
                                "choices": [{"message": {"content": "hi"}}],
                                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                            },
                        )
                    account = self._account()
                    if account is None:
                        return None
                    if path in ("/v1/submissions", "/v1/train"):
                        return self._upload(path, account, body, query)
                    if path == "/v1/picks":
                        if not state.window_open:
                            return self._error(403, "window_closed", "final picks are not open now")
                        ids = json.loads(body)["ids"]
                        own = {
                            j["id"]
                            for j in state.jobs
                            if j["account"] == account and j["kind"] == "score"
                        }
                        bad = [i for i in ids if i not in own]
                        if bad:
                            return self._error(
                                404, "not_found", f"no submission {bad[0]!r} for your account"
                            )
                        state.picks[account] = ids
                        return self._send(200, self._picks(account))
                return self._error(404, "not_found", "no such route")

            def _upload(self, path: str, account: str, body: bytes, query: dict) -> None:
                try:
                    with tarfile.open(fileobj=io.BytesIO(gzip.decompress(body))) as tar:
                        names = {m.name for m in tar}
                except (OSError, tarfile.TarError):
                    return self._error(400, "invalid_archive", "not a tar.gz")
                if "agent/NOTES.md" not in names:
                    return self._error(400, "invalid_archive", "agent/NOTES.md is missing")
                state.uploads.append(body)
                kind = "score" if path == "/v1/submissions" else "train"
                mine = [j for j in state.jobs if j["account"] == account and j["kind"] == kind]
                if kind == "score":
                    if any(j["state"] == "queued" for j in mine):
                        return self._error(
                            409, "already_queued", "a submission is still waiting in the queue"
                        )
                    if len(mine) >= 3:
                        return self._error(
                            429, "quota_exceeded", "3 submissions in the last 24 hours", 3600
                        )
                elif len(mine) >= 10:
                    return self._error(
                        429, "quota_exceeded", "10 remote training runs in the last 24 hours", 3600
                    )
                job = {
                    "id": f"{kind[0]}{len(state.jobs) + 1:04d}",
                    "kind": kind,
                    "account": account,
                    "state": "queued",
                    "message": "waiting in the queue",
                    "created_at": "2026-12-01T10:00:00Z",
                    "sha256": hashlib.sha256(body).hexdigest(),
                    "score": None,
                    "ci_low": None,
                    "ci_high": None,
                    "tokens_bucket": None,
                    "commit_url": None,
                }
                if kind == "train":
                    parsed = check_video_plan(json.loads(query["plan"][0]))
                    if isinstance(parsed, str):
                        state.uploads.pop()
                        return self._error(400, "bad_plan", parsed)
                    plan, video = parsed
                    job.update(
                        plan=plan,
                        video=video,
                        videos=sum(map(len, video.values())),
                        episodes=sum(map(len, plan.values())),
                        results=None,
                        infra_incomplete=None,
                        polls=0,
                    )
                state.jobs.append(job)
                return self._send(202, self._view(job))

            def _picks(self, account: str) -> dict[str, Any]:
                picks = state.picks.get(account)
                done = [
                    j["id"] for j in state.jobs if j["account"] == account and j["state"] == "done"
                ]
                return {
                    "window": {
                        "start": "2026-12-27T00:00:00+00:00",
                        "end": "2026-12-29T00:00:00+00:00",
                        "open": state.window_open,
                    },
                    "picks": picks or done[::-1][:2],
                    "source": "explicit" if picks else "default",
                }

            def do_GET(self) -> None:  # noqa: N802
                path = urlparse(self.path).path
                with state.lock:
                    account = self._account()
                    if account is None:
                        return None
                    mine = [j for j in state.jobs if j["account"] == account]
                    if path == "/v1/submissions":
                        subs = [self._view(j) for j in mine if j["kind"] == "score"][::-1]
                        return self._send(
                            200,
                            {
                                "submissions": subs,
                                "limits": {"per_24h": 3, "used_24h": len(subs), "next_at": None},
                            },
                        )
                    if path == "/v1/picks":
                        return self._send(200, self._picks(account))
                    for job in mine:
                        prefix = "/v1/submissions/" if job["kind"] == "score" else "/v1/train/"
                        if path == prefix + job["id"]:
                            if job["kind"] == "train" and job["state"] != "done":
                                job["polls"] += 1  # queued, then running, then done
                                job["state"] = "running" if job["polls"] == 2 else job["state"]
                                if job["polls"] >= 3:
                                    job.update(
                                        state="done",
                                        message="finished",
                                        infra_incomplete=False,
                                        results=f"/v1/train/{job['id']}/results",
                                    )
                            return self._send(200, self._view(job))
                        if path == f"/v1/train/{job['id']}/results" and job["kind"] == "train":
                            if job["state"] != "done":
                                return self._error(409, "not_ready", "not done")
                            return self._send(
                                200,
                                None,
                                raw=episode_results(job["plan"], job["video"]),
                                ctype="application/gzip",
                            )
                return self._error(404, "not_found", "not found for your account")

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._server.server_address[1]}"

    def score_all(self) -> None:
        with self.state.lock:
            for job in self.state.jobs:
                if job["kind"] == "score" and job["state"] == "queued":
                    job.update(
                        state="done",
                        message="finished",
                        score=31.5,
                        ci_low=24.0,
                        ci_high=39.0,
                        tokens_bucket="10k-30k",
                        commit_url="https://github.com/robojongit/thinkopenrobots/commit/abc",
                    )

    def __enter__(self) -> FakeGateway:
        self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._server.shutdown()
        self._server.server_close()
