"""A local fake OpenAI-compatible ``/v1/chat/completions`` server for tests."""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

Reply = str | Callable[[dict[str, Any]], tuple]


class FakeModel:
    """Answers each request with the next scripted reply (the last one repeats).

    A reply is either the assistant content (a 200 with usage) or a callable taking the request
    body and returning ``(status, json_body)`` or ``(status, json_body, headers)``.
    Every request body is kept in ``requests``.
    """

    def __init__(
        self, replies: list[Reply], prompt_tokens: int = 100, completion_tokens: int = 50
    ) -> None:
        self.replies = replies
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.requests: list[dict[str, Any]] = []
        self.paths: list[str] = []
        model = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                model.requests.append(body)
                model.paths.append(self.path)
                status, payload, *extra = model._answer(body)
                data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
                self.send_response(status)
                for key, value in (extra[0] if extra else {}).items():
                    self.send_header(key, value)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args: Any) -> None:
                pass

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(
            target=self._server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True
        )

    def _answer(self, body: dict[str, Any]) -> tuple:
        reply = self.replies[min(len(self.requests) - 1, len(self.replies) - 1)]
        if callable(reply):
            return reply(body)
        return 200, {
            "id": "fake",
            "object": "chat.completion",
            "model": body.get("model"),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": reply, "reasoning": "thinking"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.prompt_tokens + self.completion_tokens,
            },
        }

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._server.server_address[1]}/v1"

    def __enter__(self) -> FakeModel:
        self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._server.shutdown()
        self._server.server_close()
