import json

from bench.exitcodes import ExitCode
from bench.infra import Probe, classify, probe, stop_reason, wait_until_ready
from bench.tests.scripted_server import ScriptedServer
from harness.tests.fake_model import FakeModel


def write_llm(folder, *errors: str) -> None:
    entries = [{"index": 0, "error": e} for e in errors] + [{"index": 1, "reply": {}}]
    (folder / "llm.jsonl").write_text("\n".join(json.dumps(e) for e in entries))


def test_classify_flags_unreachable_5xx_and_429_only(tmp_path):
    cases = {
        "model server returned HTTP 503": True,
        "model server returned HTTP 500": True,
        "model server returned HTTP 429": True,
        "model server unreachable: [Errno 111] Connection refused": True,
        "model server returned HTTP 400": False,
        "model server returned HTTP 401": False,
        "malformed model response": False,
    }
    for i, (error, infra) in enumerate(cases.items()):
        folder = tmp_path / str(i)
        folder.mkdir()
        write_llm(folder, error)
        assert (classify(folder) == error) is infra, error


def test_classify_reads_the_upstream_record(tmp_path):
    cases = [
        ({"kind": "http", "status": 503, "retry_after": 240.0}, True),
        ({"kind": "http", "status": 429, "retry_after": 7.0}, True),
        ({"kind": "unreachable", "status": None, "retry_after": None}, True),
        ({"kind": "timeout", "status": None, "retry_after": None}, True),
        ({"kind": "http", "status": 400, "retry_after": None}, False),
        ({"kind": "malformed", "status": 200, "retry_after": None}, False),
    ]
    for i, (upstream, infra) in enumerate(cases):
        folder = tmp_path / str(i)
        folder.mkdir()
        entry = {"index": 0, "error": f"case {i}", "upstream": upstream}
        (folder / "llm.jsonl").write_text(json.dumps(entry))
        assert (classify(folder) == f"case {i}") is infra, upstream


def test_classify_without_errors_or_log(tmp_path):
    assert classify(tmp_path) is None
    write_llm(tmp_path)
    assert classify(tmp_path) is None


def waiter(results: list[Probe], **kwargs):
    sleeps: list[float] = []
    logs: list[str] = []
    queue = iter(results)
    final = wait_until_ready(lambda: next(queue), sleep=sleeps.append, log=logs.append, **kwargs)
    return final, sleeps, logs


def test_429_waits_for_retry_after():
    final, sleeps, logs = waiter([Probe(429, 7.0), Probe(429, 3.0), Probe(200)])
    assert final.ok and sleeps == [7.0, 3.0]
    assert "429" in logs[0]


def test_503_with_retry_after_says_the_model_is_waking():
    final, sleeps, logs = waiter([Probe(503, 240.0), Probe(200)])
    assert final.ok and sleeps == [240.0]
    assert logs == ["waking the shared model, about 4 minutes"]


def test_unreachable_backs_off_and_gives_up_after_max_wait():
    final, sleeps, _ = waiter(
        [Probe(None, error="unreachable")] * 10, default_wait_s=5, max_wait_s=12
    )
    assert not final.ok and final.transient
    assert sleeps == [5, 10]


def test_client_errors_are_not_waited_for():
    final, sleeps, _ = waiter([Probe(401, error="HTTP 401")])
    assert final.status == 401 and not final.transient and sleeps == []


def test_probe_reads_status_and_retry_after_from_a_server():
    with ScriptedServer(
        [(503, {"Retry-After": "240"}), (429, {"Retry-After": "7"}), (200, {})]
    ) as s:
        assert probe(s.url, "m", None) == Probe(503, 240.0, "HTTP 503")
        assert probe(s.url, "m", None).retry_after == 7.0
        assert probe(s.url, "m", None).ok
        assert s.requests[0] == ("POST", "/v1/chat/completions")


def test_probe_unreachable():
    result = probe("http://127.0.0.1:9/v1", "m", None)
    assert result.status is None and result.transient and "unreachable" in result.error


ALLOWANCE_USED = {
    "code": "model_allowance_used",
    "message": "daily model allowance used",
    "tokens_used": 5_018_043,
    "tokens_per_day": 30_000_000,
    "resets_at": "2026-11-28T00:00:00Z",
}


def test_a_used_up_allowance_is_not_waited_for_and_says_when_it_is_back():
    used = Probe(429, 52238.0, "HTTP 429", "model_allowance_used", ALLOWANCE_USED)
    final, sleeps, logs = waiter([used, Probe(200)])
    assert final is used and final.stops and sleeps == [] and logs == []
    code, message = stop_reason(final)
    assert code == ExitCode.LIMITED == 8
    assert message.startswith("daily model allowance used: 5,018,043 of 30,000,000 tokens")
    assert "resets at 00:00 UTC (in " in message


def test_an_older_gateway_quota_is_still_a_limit_with_the_reset_time():
    old = Probe(429, 3600.0, "HTTP 429", "quota_exceeded", {"code": "quota_exceeded"})
    final, sleeps, _ = waiter([old])
    assert sleeps == [] and stop_reason(final)[0] == ExitCode.LIMITED
    assert "resets at" in stop_reason(final)[1] and "(in 1.0 h)" in stop_reason(final)[1]


def test_a_long_retry_after_without_a_limit_code_is_reported_not_waited_silently():
    final, sleeps, _ = waiter([Probe(429, 52238.0, "HTTP 429")])
    code, message = stop_reason(final)
    assert sleeps == [] and code == ExitCode.INFRA
    assert "asks to retry in 14.5 h" in message and "longer than bench waits" in message


def test_the_shared_models_idle_allowance_is_a_limit_not_waking():
    capped = {"code": "chat_allowance_used", "message": "the shared model's daily idle allowance"}
    final, sleeps, logs = waiter([Probe(503, 30000.0, "HTTP 503", capped["code"], capped)])
    assert sleeps == [] and logs == []
    code, message = stop_reason(final)
    assert code == ExitCode.LIMITED and message.startswith("the shared model's daily idle")


def test_a_503_that_is_not_waking_is_not_called_waking():
    final, sleeps, logs = waiter(
        [Probe(503, 30.0, "HTTP 503 unavailable", "unavailable"), Probe(200)]
    )
    assert final.ok and sleeps == [30.0] and "waking" not in logs[0]


def test_short_waits_are_capped():
    final, sleeps, _ = waiter([Probe(429, 900.0)], max_wait_s=10)
    assert final.stops and sleeps == []  # over the cap: reported, not waited
    final, sleeps, _ = waiter([Probe(429, 600.0), Probe(200)])
    assert final.ok and sleeps == [600.0]


def test_probe_reads_the_error_code_and_body():
    answer = lambda b: (429, {"error": ALLOWANCE_USED}, {"Retry-After": "52238"})  # noqa: E731
    with FakeModel([answer]) as model:
        result = probe(model.url, "m", None)
    assert result.status == 429 and result.retry_after == 52238.0
    assert result.code == "model_allowance_used" and result.body == ALLOWANCE_USED
    assert result.limited and result.stops
