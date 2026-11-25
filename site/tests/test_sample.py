"""The sample data files must match the schema the site and the publisher share."""

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

SITE = Path(__file__).resolve().parent.parent
SCHEMA = json.loads((SITE / "leaderboard.schema.json").read_text())
SAMPLES = ["leaderboard.sample.json", "leaderboard.sample-final.json"]


def load(name):
    return json.loads((SITE / name).read_text())


def test_schema_is_valid():
    Draft202012Validator.check_schema(SCHEMA)


@pytest.mark.parametrize("name", SAMPLES)
def test_sample_matches_schema(name):
    errors = list(Draft202012Validator(SCHEMA).iter_errors(load(name)))
    assert not errors, [e.message for e in errors]


@pytest.mark.parametrize("name", SAMPLES)
def test_sample_consistency(name):
    doc = load(name)
    entries = doc["entries"]
    ids = [e["id"] for e in entries]
    assert len(ids) == len(set(ids))
    assert sum(1 for e in entries if e.get("is_base")) == 1
    rows = entries + doc.get("final", {}).get("entries", [])
    for e in rows:
        assert e["ci_low"] <= e["score"] <= e["ci_high"]
    for f in doc.get("final", {}).get("entries", []):
        assert all(p["id"] in ids for p in f["picks"])


def test_sample_has_accounts_with_several_submissions():
    accounts = [e["account"] for e in load("leaderboard.sample.json")["entries"]]
    assert any(accounts.count(a) >= 3 for a in set(accounts))


def test_per_task_fields_are_rejected():
    doc = load("leaderboard.sample.json")
    doc["entries"][1]["per_task"] = {"cube_lifting": 0.9}
    assert list(Draft202012Validator(SCHEMA).iter_errors(doc))


def test_non_https_commit_url_is_rejected():
    doc = load("leaderboard.sample.json")
    doc["entries"][1]["commit_url"] = "javascript:alert(1)"
    assert list(Draft202012Validator(SCHEMA).iter_errors(doc))
