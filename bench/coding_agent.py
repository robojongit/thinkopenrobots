"""The coding agent a submission names in ``agent/NOTES.md``, shown on the leaderboard.

The rule, the same in `bench submit` and on the server: ``agent/NOTES.md`` has exactly one line
that starts with ``Coding agent:`` (at the very start of the line, this capitalisation), for
example::

    Coding agent: Claude Code · Fable 5

The value is the rest of the line with spaces and tabs trimmed from both ends: 1 to 40
characters, only ASCII letters and digits, spaces and ``. _ + - / ( ) ·`` (the middle dot is
U+00B7). Nothing else: no commas, quotes, ``< > &``, control characters or other Unicode. Lines
are split on ``\\n`` only (a ``\\r`` before it is dropped), and a byte order mark at the start of
the file is ignored. `bench train --remote` needs no such line.

Stdlib only: the server's copy of these rules is checked against this file by tests in both
repositories.
"""

from __future__ import annotations

import re

PREFIX = "Coding agent:"
MAX_CHARS = 40
ALLOWED_TEXT = "letters, digits, spaces and . _ + - / ( ) ·"
VALUE_RE = re.compile(r"[A-Za-z0-9 ._+\-/()\u00b7]{1,40}")
EXAMPLE = "Coding agent: Claude Code · Fable 5"
HOW = (
    f'add one line "{EXAMPLE}" near the top of agent/NOTES.md, naming the coding agent that '
    'built this agent (or "Coding agent: by hand"); it is shown on the leaderboard'
)


class CodingAgentProblem(ValueError):
    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


def check_value(value: str) -> str:
    """The trimmed value, or CodingAgentProblem saying what is wrong with it."""
    if not isinstance(value, str):
        raise CodingAgentProblem("the coding agent must be text")
    value = value.strip(" \t")
    if not value:
        raise CodingAgentProblem(f'the "{PREFIX}" line in agent/NOTES.md is empty: {HOW}')
    if len(value) > MAX_CHARS:
        raise CodingAgentProblem(
            f'the "{PREFIX}" value in agent/NOTES.md has {len(value)} characters; the limit is '
            f'{MAX_CHARS}: shorten it to a name, for example "{EXAMPLE}"'
        )
    if not VALUE_RE.fullmatch(value):
        bad = sorted({c for c in value if not VALUE_RE.fullmatch(c)})
        shown = " ".join(repr(c) for c in bad[:10])
        raise CodingAgentProblem(
            f'the "{PREFIX}" value in agent/NOTES.md has characters that are not allowed '
            f'({shown}); use only {ALLOWED_TEXT}, for example "{EXAMPLE}"'
        )
    return value


def parse_notes(notes: bytes | str) -> str:
    """The coding agent named in the content of ``agent/NOTES.md``, or CodingAgentProblem."""
    if isinstance(notes, bytes):
        try:
            notes = notes.decode("utf-8")
        except UnicodeDecodeError:
            raise CodingAgentProblem("agent/NOTES.md is not UTF-8 text") from None
    notes = notes.removeprefix("\ufeff")
    lines = [line.removesuffix("\r") for line in notes.split("\n")]
    found = [line for line in lines if line.startswith(PREFIX)]
    if not found:
        raise CodingAgentProblem(f'agent/NOTES.md has no "{PREFIX}" line: {HOW}')
    if len(found) > 1:
        raise CodingAgentProblem(
            f'agent/NOTES.md has {len(found)} lines starting with "{PREFIX}"; keep exactly one, '
            f'for example "{EXAMPLE}"'
        )
    return check_value(found[0][len(PREFIX) :])
