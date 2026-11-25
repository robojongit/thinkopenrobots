import json
import socket
import time

import pytest

from harness.llm_relay import (
    FREE_FAILED_CALLS,
    Budget,
    LLMRelay,
    LogBudget,
    chat_url,
    estimate_prompt_tokens,
    request_seed,
    retry_after_seconds,
)
from harness.rpc import RpcError
from harness.tests.fake_model import FakeModel

MSGS = [{"role": "user", "content": "hi"}]


DEFAULT_BUDGET = Budget()


def relay_for(fake, budget=DEFAULT_BUDGET, wall=60.0, **kw) -> LLMRelay:
    return LLMRelay(fake.url, 7, budget, time.monotonic() + wall, **kw)


def test_forwards_with_forced_fields_and_logs():
    with FakeModel(["hello"], prompt_tokens=10, completion_tokens=5) as model:
        relay = relay_for(model, model="pinned-model", api_key="k")
        reply = relay.chat(MSGS, {"temperature": 1.0, "max_tokens": 100, "stop": "END"})
    assert reply == {
        "content": "hello",
        "reasoning": "thinking",
        "finish_reason": "stop",
        "prompt_tokens": 10,
        "completion_tokens": 5,
    }
    (req,) = model.requests
    assert model.paths == ["/v1/chat/completions"]
    assert req["model"] == "pinned-model"
    assert req["seed"] == request_seed(7, 0)
    assert req["max_tokens"] == 100 and req["stop"] == ["END"] and req["temperature"] == 1.0
    assert req["n"] == 1 and req["stream"] is False
    assert set(req) == {"model", "messages", "max_tokens", "seed", "n", "stream", "temperature",
                        "stop", "cache_salt"}  # fmt: skip
    assert relay.calls == 1 and relay.tokens == 15
    (entry,) = relay.log
    assert entry["index"] == 0 and entry["reply"]["content"] == "hello"
    assert "seed" not in entry["request"] and "model" not in entry["request"]
    assert "cache_salt" not in entry["request"]


def test_request_seeds_depend_on_episode_seed_and_call_index():
    seeds = {request_seed(s, i) for s in range(20) for i in range(10)}
    assert len(seeds) == 200
    assert request_seed(3, 4) == request_seed(3, 4)
    assert all(0 <= s < 2**31 for s in seeds)
    with FakeModel(["a"]) as model:
        relay = relay_for(model)
        for _ in range(3):
            relay.chat(MSGS, {})
    assert [r["seed"] for r in model.requests] == [request_seed(7, i) for i in range(3)]


def test_cache_salt_is_per_episode_and_can_be_turned_off():
    def strict(body):  # a server that rejects fields it does not know (OpenAI's own API does)
        known = {"model", "messages", "max_tokens", "seed", "n", "stream"}
        if set(body) - known:
            return 400, {"error": f"Unrecognized request argument: {sorted(set(body) - known)}"}
        return 200, {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}

    with FakeModel(["a"]) as model:
        first, second = relay_for(model), relay_for(model)
        first.chat(MSGS, {})
        first.chat(MSGS, {})
        second.chat(MSGS, {})
    salts = [r["cache_salt"] for r in model.requests]
    assert salts[0] == salts[1] != salts[2] and len(salts[0]) == 32
    with FakeModel([strict]) as model:
        with pytest.raises(RpcError):
            relay_for(model).chat(MSGS, {})
        assert relay_for(model, cache_salt=False).chat(MSGS, {})["content"] == "ok"


def test_chat_url():
    assert chat_url("http://h:1/v1") == "http://h:1/v1/chat/completions"
    assert chat_url("http://h:1/v1/chat/completions/") == "http://h:1/v1/chat/completions"


@pytest.mark.parametrize(
    "options",
    [
        {"model": "other-model"},
        {"seed": 1},
        {"n": 4},
        {"stream": True},
        {"logit_bias": {}},
        {"priority": -100},
        {"tools": []},
        {"temperature": "hot"},
        {"temperature": True},
        {"temperature": float("nan")},
        {"temperature": 5.0},
        {"max_tokens": 1.5},
        {"max_tokens": 0},
        {"stop": ["a", "b", "c", "d", "e"]},
        {"stop": [1]},
    ],
)
def test_options_outside_the_whitelist_are_rejected(options):
    with FakeModel(["a"]) as model:
        relay = relay_for(model)
        with pytest.raises(RpcError) as e:
            relay.chat(MSGS, options)
    assert e.value.code == "bad_request" and model.requests == []


@pytest.mark.parametrize(
    "messages",
    [
        [],
        "hi",
        [{"role": "user"}],
        [{"role": "tool", "content": "x"}],
        [{"role": ["user"], "content": "x"}],  # unhashable: was an internal error
        [{"role": {"r": 1}, "content": "x"}],
        [{"role": "user", "content": "x", "name": "n"}],
        [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}],
        [{"role": "user", "content": [{"type": "text", "text": 1}]}],
        [{"role": "user", "content": []}],
        [{"role": "user", "content": "x" * 1_000_001}],
        [{"role": "user", "content": "x"}] * 201,
    ],
)
def test_bad_messages_are_rejected(messages):
    with FakeModel(["a"]) as model:
        relay = relay_for(model)
        with pytest.raises(RpcError) as e:
            relay.chat(messages, {})
    assert e.value.code == "bad_request" and model.requests == []


def test_text_parts_are_accepted():
    with FakeModel(["a"]) as model:
        relay = relay_for(model)
        msgs = [{"role": "system", "content": "s"},
                {"role": "user", "content": [{"type": "text", "text": "t"}]}]  # fmt: skip
        relay.chat(msgs, {})
    assert model.requests[0]["messages"] == msgs


def test_call_budget():
    with FakeModel(["a"]) as model:
        relay = relay_for(model, Budget(calls=2))
        relay.chat(MSGS, {})
        relay.chat(MSGS, {})
        with pytest.raises(RpcError) as e:
            relay.chat(MSGS, {})
    assert e.value.code == "budget" and len(model.requests) == 2 and relay.limit_hit == "calls"


def test_token_budget_caps_max_tokens_then_refuses():
    bound = estimate_prompt_tokens(MSGS)
    with FakeModel(["a"], prompt_tokens=300, completion_tokens=400) as model:
        relay = relay_for(model, Budget(tokens=1000))
        relay.chat(MSGS, {"max_tokens": 20480})
        assert model.requests[0]["max_tokens"] == 1000 - bound
        relay.chat(MSGS, {})  # default max_tokens is capped too
        assert model.requests[1]["max_tokens"] == 300 - bound
        with pytest.raises(RpcError) as e:
            relay.chat(MSGS, {})
    assert e.value.code == "budget" and relay.limit_hit == "tokens"
    assert len(model.requests) == 2 and relay.tokens == 1400


def test_estimate_is_an_upper_bound_in_bytes():
    one = estimate_prompt_tokens([{"role": "user", "content": "abc"}])
    assert estimate_prompt_tokens([{"role": "user", "content": "abcd"}]) == one + 1
    assert estimate_prompt_tokens([{"role": "user", "content": "ab\u00e9"}]) == one + 1  # 2 bytes
    parts = [
        {"role": "user", "content": [{"type": "text", "text": "ab"}, {"type": "text", "text": "c"}]}
    ]
    assert estimate_prompt_tokens(parts) == one


def test_long_prompt_cannot_overshoot_the_budget():
    # The review's PoC: with 1 token left, a ~60k-token prompt used to be forwarded. Now the
    # prompt's upper bound must fit in what is left, and max_tokens fills at most the rest.
    def usage(body):  # ~4 characters per token, and the whole max_tokens generated
        chars = sum(len(m["content"]) for m in body["messages"])
        return 200, {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                     "usage": {"prompt_tokens": chars // 4,
                               "completion_tokens": body["max_tokens"]}}  # fmt: skip

    def ask(n):
        return relay.chat([{"role": "user", "content": "x" * n}], {})

    with FakeModel([usage]) as model:
        relay = relay_for(model, Budget(tokens=10_000))
        ask(7000)  # bound 7072: max_tokens 2928; charged 1750 + 2928
        assert relay.tokens == 4678
        with pytest.raises(RpcError) as e:  # bound 6072 > 5322 left (real: ~1500)
            ask(6000)
        assert e.value.code == "budget" and relay.limit_hit == "tokens"
        ask(5000)  # bound 5072: max_tokens 250
    assert [r["max_tokens"] for r in model.requests] == [2928, 250]
    assert relay.tokens == 4678 + 1250 + 250 <= 10_000


def test_max_tokens_is_capped_by_the_context_length():
    with FakeModel(["a"]) as model:
        relay = relay_for(model, Budget(tokens=10**6), context_length=4096)
        relay.chat(MSGS, {"max_tokens": 10**6})
        with pytest.raises(RpcError) as e:
            relay.chat([{"role": "user", "content": "x" * 4096}], {})
    assert model.requests[0]["max_tokens"] == 4096 - estimate_prompt_tokens(MSGS)
    assert e.value.code == "bad_request" and "context" in e.value.message
    assert len(model.requests) == 1 and relay.limit_hit is None
    assert relay_for(model).context_length == 65536


def test_default_budget_fits_the_base_request():
    # The final budget (10 calls, 66k tokens, 400 s): the base solution's one call, max_tokens
    # 20480 with a prompt bounded well above its real size, goes out uncapped. The budget is
    # larger than the model's context, so one prompt is limited by the context, not the budget.
    assert Budget() == Budget(calls=10, tokens=66_000, wall_s=400.0)
    with FakeModel(["a"]) as model:
        relay = relay_for(model)
        relay.chat([{"role": "user", "content": "x" * 30_000}], {"max_tokens": 20480})
        with pytest.raises(RpcError) as e:
            relay.chat([{"role": "user", "content": "x" * 65_536}], {})
    assert model.requests[0]["max_tokens"] == 20480
    assert relay.context_length < Budget().tokens
    assert e.value.code == "bad_request" and "context" in e.value.message


def test_wall_clock_budget():
    with FakeModel(["a"]) as model:
        relay = relay_for(model, wall=-1)
        with pytest.raises(RpcError) as e:
            relay.chat(MSGS, {})
    assert e.value.code == "budget" and relay.limit_hit == "wall" and model.requests == []


def test_slow_model_is_cut_off_at_the_deadline():
    def slow(body):
        time.sleep(3)
        return 200, {}

    with FakeModel([slow]) as model:
        relay = relay_for(model, wall=0.5)
        t0 = time.monotonic()
        with pytest.raises(RpcError) as e:
            relay.chat(MSGS, {})
    # The episode's time limit, not a model-server failure (bench must not retry it as infra).
    assert e.value.code == "budget" and time.monotonic() - t0 < 2.5
    assert relay.limit_hit == "wall" and relay.upstream_errors == 0 and relay.calls == 0
    assert relay.upstream_failures == [] and "upstream" not in relay.log[-1]
    assert "time limit" in relay.log[-1]["error"]


@pytest.mark.parametrize(
    "answer",
    [
        lambda b: (500, {"error": "boom"}),
        lambda b: (429, {"error": "busy"}),
        lambda b: (200, b"not json"),
        lambda b: (200, {"choices": []}),
        lambda b: (200, {"choices": [{"message": {"content": 5}}]}),
    ],
)
def test_upstream_failures_are_not_charged(answer):
    with FakeModel([answer]) as model:
        relay = relay_for(model)
        with pytest.raises(RpcError) as e:
            relay.chat(MSGS, {})
    assert e.value.code == "upstream"
    assert relay.calls == 0 and relay.tokens == 0 and relay.upstream_errors == 1
    assert "error" in relay.log[0]


def test_failed_requests_count_as_calls_after_a_free_few():
    # The review's PoC: 200 requests of ~1M characters, all failing, cost nothing.
    with FakeModel([lambda b: (400, {"error": "context length exceeded"})]) as model:
        relay = relay_for(model, Budget(calls=5))
        codes = []
        for _ in range(20):
            try:
                relay.chat([{"role": "user", "content": "secret " * 1000}], {})
            except RpcError as e:
                codes.append(e.code)
    assert codes == ["upstream"] * (FREE_FAILED_CALLS + 5) + ["budget"] * (15 - FREE_FAILED_CALLS)
    assert len(model.requests) == FREE_FAILED_CALLS + 5
    assert relay.calls == 5 and relay.limit_hit == "calls" and relay.tokens == 0
    assert all(e["request"]["messages"]["kept"] is False for e in relay.log)
    assert "secret" not in json.dumps(relay.log)


def test_upstream_status_and_retry_after_are_data():
    answers = [
        lambda b: (503, {"error": "waking"}, {"Retry-After": "240"}),
        lambda b: (429, {"error": "busy"}, {"Retry-After": "Wed, 21 Oct 2015 07:28:00 GMT"}),
        lambda b: (200, {"choices": []}),
    ]
    with FakeModel(answers) as model:
        relay = relay_for(model)
        errors = []
        for _ in answers:
            with pytest.raises(RpcError) as e:
                relay.chat(MSGS, {})
            errors.append(e.value)
    assert [e.data for e in errors] == [
        {"kind": "http", "status": 503, "retry_after": 240.0},
        {"kind": "http", "status": 429, "retry_after": 0.0},  # a date in the past
        {"kind": "malformed", "status": 200, "retry_after": None},
    ]
    assert errors[0].to_wire()["data"]["status"] == 503
    assert [entry["upstream"] for entry in relay.log] == [e.data for e in errors]
    assert [f["index"] for f in relay.upstream_failures] == [0, 0, 0]
    assert relay.upstream_failures[0] == {"index": 0, **errors[0].data}


def test_unreachable_model_server_is_data():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]  # closed again before the request: connection refused
    relay = LLMRelay(f"http://127.0.0.1:{port}/v1", 1, Budget(), time.monotonic() + 30)
    with pytest.raises(RpcError) as e:
        relay.chat(MSGS, {})
    assert e.value.data == {"kind": "unreachable", "status": None, "retry_after": None}


def test_retry_after_parsing():
    assert retry_after_seconds("7") == 7.0
    assert retry_after_seconds(" 0 ") == 0.0
    assert retry_after_seconds("Wed, 21 Oct 2015 07:28:00 GMT", now=1445412470.0) == 10.0
    assert retry_after_seconds("soon") is None and retry_after_seconds(None) is None


def test_missing_usage_is_charged_as_worst_case():
    def no_usage(body):
        return 200, {"choices": [{"message": {"content": "x"}, "finish_reason": "stop"}]}

    with FakeModel([no_usage]) as model:
        relay = relay_for(model)
        relay.chat([{"role": "user", "content": "y" * 1000}], {"max_tokens": 64})
    assert relay.completion_tokens == 64 and relay.prompt_tokens >= 250


def test_is_substring_of_reply():
    with FakeModel(["Here:\n```python\nopen_gripper()\nclose_gripper()\n```"]) as model:
        relay = relay_for(model)
        assert not relay.is_substring_of_reply("open_gripper()")
        relay.chat(MSGS, {})
    assert relay.is_substring_of_reply("open_gripper()\nclose_gripper()")
    assert relay.is_substring_of_reply("open_gripper()")
    assert not relay.is_substring_of_reply("open_gripper();close_gripper()")
    assert not relay.is_substring_of_reply("")
    assert not relay.is_substring_of_reply("  \n")
    assert not relay.is_substring_of_reply("thinking")  # reasoning text does not count


def test_log_budget_keeps_the_first_entries_then_only_counts():
    budget = LogBudget(100)
    assert budget.admit("events", {"a": "x" * 40})
    assert not budget.admit("events", {"a": "x" * 80})
    assert not budget.admit("runs", {"a": 1})  # small, but after the first miss: counted only
    assert budget.dropped == {"events": 1, "runs": 1}
    with FakeModel(["a"]) as model:
        relay = relay_for(model, log_budget=LogBudget(10))
        relay.chat(MSGS, {})
    assert relay.log == [] and relay.log_budget.dropped == {"model requests": 1}
