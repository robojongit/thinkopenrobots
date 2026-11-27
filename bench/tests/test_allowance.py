from datetime import UTC, datetime

from bench import allowance

NOW = datetime(2026, 11, 27, 9, 29, 22, tzinfo=UTC)


def test_the_allowance_from_a_newer_gateway():
    model = {
        "tokens_per_day": 30_000_000,
        "tokens_used": 5_018_043,
        "resets_at": "2026-11-28T00:00:00Z",
        "concurrent_requests": 4,
    }
    info = allowance.from_answer({"submissions": [], "model": model})
    assert info == model and allowance.left(info) == 24_981_957
    assert allowance.describe(info, NOW) == (
        "model tokens today: 5,018,043 of 30,000,000 used, resets at 00:00 UTC (in 14.5 h); "
        "at most 4 model requests at a time"
    )


def test_the_allowance_from_an_older_gateway_or_none():
    info = allowance.from_answer({"account": "a", "tokens_today": 7, "tokens_per_day": 5_000_000})
    assert info["tokens_used"] == 7 and info["tokens_per_day"] == 5_000_000
    assert info["concurrent_requests"] is None and info["resets_at"].endswith("T00:00:00Z")
    for answer in ({}, {"account": "a", "tokens_today": 0}, {"model": "x"}, None, []):
        assert allowance.from_answer(answer) is None
    broken = allowance.from_answer({"model": {"tokens_per_day": 10, "resets_at": "soon"}})
    assert broken["tokens_used"] is None and allowance.left(broken) is None
    assert "? of 10 used" in allowance.describe(broken)


def test_the_used_up_message_with_and_without_the_numbers():
    error = {
        "tokens_used": 5_018_043,
        "tokens_per_day": 5_000_000,
        "resets_at": "2026-11-28T00:00:00Z",
    }
    text = allowance.used_up(error, 52238, NOW)
    assert text.startswith(
        "daily model allowance used: 5,018,043 of 5,000,000 tokens, "
        "resets at 00:00 UTC (in 14.5 h)."
    )
    assert "same --out" in text and "does not help" in text
    old = allowance.used_up({"message": "daily model token limit (5000000) reached"}, 52238, NOW)
    assert old.startswith("daily model allowance used, resets at 00:00 UTC (in 14.5 h).")
    soon = allowance.resets_text("2026-11-27T09:50:00Z", NOW)
    assert soon == "09:50 UTC (in 21 min)"
    assert allowance.resets_text("2026-11-30T00:00:00Z", NOW).startswith("2026-11-30 00:00 UTC")
