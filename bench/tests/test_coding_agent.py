"""The ``Coding agent:`` line of agent/NOTES.md: parsing, every refusal, and agreement with the
site's schema, the site's script and (when ``INKLING_OPS_DIR`` is set) the server's copy."""

from __future__ import annotations

import importlib.util
import json
import os
import re
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from bench import coding_agent
from bench.coding_agent import CodingAgentProblem, check_value, parse_notes

REPO = Path(__file__).resolve().parents[2]
OPS = os.environ.get("INKLING_OPS_DIR")
SCHEMA = json.loads((REPO / "site" / "leaderboard.schema.json").read_text())

GOOD = [
    "Claude Code · Fable 5",
    "Claude Code · Sonnet 5.5",
    "Codex",
    "Cursor",
    "Gemini CLI",
    "by hand",
    "reference harness (Claude Code port)",
    "aider + gpt_x/2-mini",
    "x" * 40,
    "7",
]
BAD = [
    "",
    "   ",
    "x" * 41,
    "<b>Codex</b>",
    "Codex & co",
    'say "hi"',
    "it's",
    "a,b",
    "Codex\x00",
    "Co\tdex",
    "Codex\u2028",
    "Cod\u0435x",  # Cyrillic e
    "Codex \u2022 5",  # bullet, not the middle dot
    "Codex\u200b",
    "javascript:alert(1)",
    "a;b",
    "#1",
    "50%",
]
NOTES = [
    b"Coding agent: Codex\n",
    b"# Notes\r\n\r\nCoding agent:\tCursor \r\nmore\r\n",
    "\ufeffCoding agent: Claude Code · Fable 5".encode(),
    b"Coding agent:by hand",
    b"intro\n- **Coding agent:** Codex\nCoding agent: Codex\n",
    b"",
    b"Notes only\n",
    b"coding agent: Codex\n",
    b" Coding agent: Codex\n",
    b"- Coding agent: Codex\n",
    b"Coding agent: Codex\nCoding agent: Codex\n",
    b"Coding agent: Codex\rCoding agent: Cursor\n",
    b"Coding agent: <script>\n",
    b"Coding agent: " + b"y" * 41 + b"\n",
    b"Coding agent: \xff\n",
    b"Coding agent: Codex\x0bCursor\n",
    b"Coding agent:\n",
]


def outcome(module, notes: bytes) -> tuple[str, str]:
    try:
        return "ok", module.parse_notes(notes)
    except module.CodingAgentProblem as e:
        return "refused", e.message


def test_good_values_are_kept_as_written():
    for value in GOOD:
        assert check_value(value) == value
        assert parse_notes(f"# Notes\n\nCoding agent: {value}\n\nfree text\n") == value
    assert check_value("  Codex \t") == "Codex"


@pytest.mark.parametrize("value", BAD)
def test_bad_values_are_refused_with_the_line_to_add(value):
    with pytest.raises(CodingAgentProblem) as e:
        parse_notes(f"Coding agent: {value}\n")
    assert "Coding agent: Claude Code · Fable 5" in e.value.message
    assert "\n" not in e.value.message


def test_each_refusal_says_what_is_wrong():
    cases = {
        b"Notes\n": 'no "Coding agent:" line',
        b"Coding agent: a\nCoding agent: b\n": "2 lines starting with",
        b"Coding agent: \n": "is empty",
        b"Coding agent: " + b"z" * 41: "41 characters; the limit is 40",
        b"Coding agent: <x>": "not allowed ('<' '>')",
        b"\xfe": "not UTF-8",
    }
    for notes, message in cases.items():
        with pytest.raises(CodingAgentProblem, match=re.escape(message)):
            parse_notes(notes)


def test_line_rules():
    assert parse_notes(NOTES[1]) == "Cursor"  # CRLF, tab after the colon, trailing space
    assert parse_notes(NOTES[2]) == "Claude Code · Fable 5"  # byte order mark
    assert parse_notes(NOTES[4]) == "Codex"  # the old bullet does not count as a line
    for notes in NOTES[7:10]:  # other case, indented, bulleted: not the line
        with pytest.raises(CodingAgentProblem, match="no "):
            parse_notes(notes)


def test_schema_accepts_exactly_the_values_bench_accepts():
    entry = SCHEMA["$defs"]["entry"]
    assert "coding_agent" not in entry["required"]
    assert entry["properties"]["coding_agent"] == {"$ref": "#/$defs/coding_agent"}
    validator = Draft202012Validator(SCHEMA["$defs"]["coding_agent"])
    for value in GOOD + BAD + [" Codex", "Codex "]:
        try:
            ok = check_value(value) == value
        except CodingAgentProblem:
            ok = False
        assert validator.is_valid(value) == ok, value


def test_site_script_uses_the_same_rule():
    app = (REPO / "site" / "app.js").read_text()
    js = re.search(r"var CODING_AGENT = /(.*)/;", app)[1]
    assert js == r"^[A-Za-z0-9 ._+\-\/()\u00b7]{1,40}$"
    assert coding_agent.VALUE_RE.pattern == r"[A-Za-z0-9 ._+\-/()\u00b7]{1,40}"


@pytest.mark.skipif(not OPS, reason="set INKLING_OPS_DIR to the ops repo to compare")
def test_server_rule_is_the_same():
    path = Path(OPS) / "gateway" / "coding_agent.py"
    spec = importlib.util.spec_from_file_location("ops_coding_agent", path)
    server = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(server)
    for name in ("PREFIX", "MAX_CHARS", "ALLOWED_TEXT", "EXAMPLE", "HOW"):
        assert getattr(server, name) == getattr(coding_agent, name), name
    assert server.VALUE_RE.pattern == coding_agent.VALUE_RE.pattern
    corpus = NOTES + [f"Coding agent: {v}\n".encode() for v in GOOD + BAD]
    for notes in corpus:
        assert outcome(server, notes) == outcome(coding_agent, notes), notes
