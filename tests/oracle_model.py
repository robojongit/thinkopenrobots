"""A scripted stand-in for the model that answers CaP-X's prompts with the tasks' oracle code.

It recognises the task from the prompt text (CaP-X's task prompt starts the user message) and
replies with that task's privileged oracle code as bare code, the format CaP-X's task prompts ask
for ("do not write it in code fences"). Any other request (e.g. bench's readiness probe) gets an
empty reply. Needs CaP-X (the ``sim`` dependency group).

    python -m tests.oracle_model     # serve until interrupted; prints the base URL
"""

from __future__ import annotations

import sys
import time
from typing import Any

from harness.capx_env import capx_root, import_capx
from harness.tasks import TASKS
from harness.tests.fake_model import FakeModel
from harness.tests.reference import oracle_code
from tests.base_agent import completion


def task_prompt(task: str) -> str:
    """The task prompt CaP-X uses for ``task``'s privileged config (the config's ``prompt`` if
    set, else the env class's ``prompt``; ``capx/envs/tasks/base.py:110``)."""
    import_capx()
    from capx.envs.configs.instantiate import locate
    from capx.envs.configs.loader import DictLoader

    env = DictLoader.load([str(capx_root() / TASKS[task].config)])["env"]
    return env["cfg"].get("prompt") or locate(env["_target_"]).prompt


def user_text(body: dict[str, Any]) -> str:
    content = body["messages"][-1]["content"]
    if isinstance(content, str):
        return content
    return "".join(part.get("text", "") for part in content)


class OracleModel(FakeModel):
    """``FakeModel`` whose reply to a task's prompt is that task's oracle code."""

    def __init__(self, tasks: list[str] | None = None) -> None:
        self.prompts = {task: task_prompt(task) for task in tasks or TASKS}
        self.codes = {task: oracle_code(task) for task in self.prompts}
        self.answered: list[str | None] = []
        super().__init__([self._reply])

    def task_of(self, body: dict[str, Any]) -> str | None:
        text = user_text(body)
        found = [t for t, p in self.prompts.items() if text.startswith(p + "\nAPIs:\n")]
        return found[0] if len(found) == 1 else None

    def _reply(self, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        task = self.task_of(body)
        self.answered.append(task)
        return 200, completion(self.codes[task] if task else "")


def main() -> int:
    with OracleModel() as model:
        print(model.url, flush=True)
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            return 0


if __name__ == "__main__":
    sys.exit(main())
