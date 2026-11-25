"""The participant docs must match the code they describe, and must not leak internal details."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pytest

from bench.cli import build_parser
from bench.coding_agent import EXAMPLE, parse_notes
from bench.exitcodes import ExitCode
from harness.llm_relay import Budget
from harness.tasks import TASKS

ROOT = Path(__file__).resolve().parent.parent
DOCS = {name: (ROOT / name).read_text() for name in ("AGENTS.md", "README.md")}
RULES = (ROOT / "site" / "rules.html").read_text()

# Organiser-only details that must not appear in public docs.
DENYLIST = [
    r"\bAWS\b",
    r"vLLM box",
    r"Secrets Manager",
    r"\bspot\b",
]


def subparsers() -> dict[str, argparse.ArgumentParser]:
    parser = build_parser()
    (action,) = (a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
    return dict(action.choices)


def code_segments(text: str) -> list[str]:
    """Fenced block lines and inline code spans, each as one string."""
    segments = []
    for block in re.findall(r"```[^\n]*\n(.*?)```", text, re.S):
        segments += block.splitlines()
    outside = re.sub(r"```[^\n]*\n.*?```", "", text, flags=re.S)
    segments += re.findall(r"`([^`\n]+)`", outside)
    return segments


def bench_commands(text: str) -> list[tuple[str, list[str]]]:
    """(subcommand, flags) for every `bench <subcommand> ...` in a code segment."""
    found = []
    for seg in code_segments(text):
        seg = re.split(r"\s#\s|&&|\||;|>", seg)[0]
        m = re.search(r"(?:^|\s|/)bench\s+([a-z][a-z-]*)\b(.*)$", seg)
        if m and (m[0].lstrip().startswith("bench") or "uv run bench" in seg or "./bench" in seg):
            found.append((m[1], re.findall(r"(?<![\w-])(--[a-z][\w-]*)", m[2])))
    return found


@pytest.mark.parametrize("name", DOCS)
def test_bench_subcommands_and_flags_exist(name):
    subs = subparsers()
    commands = bench_commands(DOCS[name])
    assert commands, f"{name} mentions no bench command"
    for sub, flags in commands:
        assert sub in subs, f"{name}: unknown subcommand `bench {sub}`"
        known = {o for a in subs[sub]._actions for o in a.option_strings}
        for flag in flags:
            assert flag in known, f"{name}: `bench {sub}` has no {flag}"


@pytest.mark.parametrize("name", DOCS)
def test_every_flag_mentioned_exists_somewhere(name):
    known = {o for p in subparsers().values() for a in p._actions for o in a.option_strings}
    for seg in code_segments(DOCS[name].split("## Development")[0]):
        if re.match(r"\s*(git|uv sync|uv python|curl)\b", seg):  # other tools' flags
            continue
        for flag in re.findall(r"(?<![\w-])(--[a-z][\w*-]*)", seg):
            if flag == "--recurse-submodules":  # a git flag, mentioned as something not to use
                continue
            if flag.endswith("*"):
                assert any(k.startswith(flag[:-1]) for k in known), f"{name}: {flag}"
            else:
                assert flag in known, f"{name}: unknown flag {flag}"


def test_every_subcommand_is_documented():
    for name in ("AGENTS.md", "README.md"):
        mentioned = {sub for sub, _ in bench_commands(DOCS[name])}
        assert set(subparsers()) <= mentioned, f"{name} misses {set(subparsers()) - mentioned}"


def test_exit_code_table_matches():
    rows = re.findall(r"^\| (\d+) \| (\w+) \|", DOCS["AGENTS.md"], re.M)
    assert {(int(n), name) for n, name in rows} == {(int(c), c.name.lower()) for c in ExitCode}


def site_prompt() -> str:
    """The starter prompt as site/app.js builds it from site/config.js."""
    config = (ROOT / "site" / "config.js").read_text()
    repo = re.search(r'repoUrl:\s*"([^"]+)"', config)[1]
    app = (ROOT / "site" / "app.js").read_text()
    body = re.search(r"function promptText\(\) \{(.*?)\n  \}", app, re.S)[1]
    expr = re.search(r"return\s+(.*?);", body, re.S)[1]
    parts = []
    for term in re.split(r'\+\s*(?=["r])', expr):
        term = term.strip()
        literal = bytes(term[1:-1], "utf-8").decode("unicode_escape")
        parts.append(repo if term == "repo" else literal)
    return "".join(parts)


def test_readme_prompt_equals_site_prompt():
    prompt = site_prompt()
    assert "AGENTS.md" in prompt
    blocks = re.findall(r"```\n(.*?)```", DOCS["README.md"], re.S)
    assert prompt in [b.rstrip("\n") for b in blocks]


def test_task_table_matches_tasks_py():
    rows = {}
    for line in DOCS["README.md"].splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 5 and cells[0].strip("`") in TASKS:
            names = lambda cell: set(re.findall(r"`(\w+)`", cell))  # noqa: E731
            rows[cells[0].strip("`")] = (names(cells[2]), names(cells[3]))
    assert set(rows) == set(TASKS)
    for task, (reads, motions) in rows.items():
        assert reads == set(TASKS[task].reads), task
        assert motions == set(TASKS[task].motions), task


@pytest.mark.parametrize("name", DOCS)
def test_no_internal_words(name):
    for pattern in DENYLIST:
        assert not re.search(pattern, DOCS[name], re.I if pattern[0] != "\\" else 0), (
            f"{name}: {pattern}"
        )
    for pattern in DENYLIST:
        assert not re.search(pattern, RULES), f"rules.html: {pattern}"


def count_budget_numbers(text: str) -> int:
    tokens = Budget().tokens
    return len(re.findall(rf"{tokens // 1000}k|{tokens:,}|{tokens}", text))


@pytest.mark.parametrize("name", ["AGENTS.md", "README.md", "rules.html"])
def test_docs_state_the_budget_defaults(name):
    text = RULES if name == "rules.html" else DOCS[name]
    default = Budget()
    assert count_budget_numbers(text) == 1, name
    assert re.search(rf"\b{default.calls} (model )?calls\b", text), name
    assert re.search(rf"\b{default.wall_s:g} s(econds)?\b", text), name


def test_rules_guide_is_filled():
    assert "placeholder" not in RULES and "To be written" not in RULES
    for guide in ("measure", "failures", "changes", "compare", "pitfalls"):
        assert f'id="guide-{guide}"' in RULES


@pytest.mark.parametrize("name", DOCS)
def test_python_examples_parse(name):
    for block in re.findall(r"^```python\n(.*?)^```$", DOCS[name], re.S | re.M):
        compile(block, name, "exec")


def test_docs_state_the_coding_agent_line():
    assert (
        "coding agent that built it, in <code>agent/NOTES.md</code>; it is shown on the " in RULES
    )
    for name, text in DOCS.items():
        assert "Disclose AI help" not in text, name
        flat = " ".join(text.split())
        assert f"`{EXAMPLE}`" in flat and "shown on the leaderboard" in flat, name
        for line in re.findall(r"`(Coding agent: [^`<]+)`", text):  # not the <name> template
            assert parse_notes(line + "\n"), (name, line)
    assert "Disclose AI help" not in RULES
