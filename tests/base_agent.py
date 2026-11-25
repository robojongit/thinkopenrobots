"""Helpers for the base-solution tests: load ``base/agent.py`` and run it in an episode."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

from harness.episode import run_episode
from harness.llm_relay import Budget

ROOT = Path(__file__).resolve().parent.parent
BASE_DIR = ROOT / "base"
AGENT_DIR = ROOT / "agent"


def load_agent(folder: Path = BASE_DIR) -> ModuleType:
    """Import ``<folder>/agent.py`` under a private module name."""
    spec = importlib.util.spec_from_file_location(
        f"_inkling_{folder.name}_agent", folder / "agent.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def completion(content: str, prompt_tokens: int = 100, completion_tokens: int = 50) -> dict:
    """An OpenAI-style chat completion body with ``content`` as the reply."""
    return {
        "id": "scripted",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content, "reasoning": None},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


class Episode:
    def __init__(self, result: dict[str, Any], out: Path) -> None:
        self.result = result
        self.out = out

    def json(self, name: str) -> Any:
        return json.loads((self.out / name).read_text())

    @property
    def log(self) -> str:
        return (self.out / "agent.log").read_text()


def run_base(
    task: str,
    seed: int,
    model_url: str,
    out: Path,
    *,
    make_env: Any = None,
    budget: Budget | None = None,
    agent_dir: Path = BASE_DIR,
) -> Episode:
    result = run_episode(
        task, seed, agent_dir, model_url, budget or Budget(), out, make_env=make_env
    )
    return Episode(result, out)
