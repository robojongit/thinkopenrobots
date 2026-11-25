import json

from bench.infra import Probe, classify, probe, wait_until_ready
from bench.tests.scripted_server import ScriptedServer


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
