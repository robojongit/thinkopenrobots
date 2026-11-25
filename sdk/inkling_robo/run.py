"""Agent entry point: ``python -m inkling_robo.run <agent_dir> [--socket PATH]``.

Loads ``<agent_dir>/agent.py`` and calls ``solve(task, robot, llm)``. Exit status 0 means
``solve`` returned; 1 means it raised.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import traceback
from pathlib import Path

from .client import LLM, Connection, Robot, Task, line_buffered_output

DEFAULT_SOCKET = "/run/sim.sock"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m inkling_robo.run")
    parser.add_argument("agent_dir", type=Path)
    parser.add_argument("--socket", default=os.environ.get("INKLING_ROBO_SOCKET", DEFAULT_SOCKET))
    args = parser.parse_args(argv)
    line_buffered_output()

    agent_dir = args.agent_dir.resolve()
    sys.path.insert(0, str(agent_dir))
    try:
        conn = Connection(args.socket)
        info = conn.call("hello")
        task = Task(instruction=info["instruction"], api_docs=info["api_docs"])
        robot = Robot(conn, info["read_functions"])
        llm = LLM(conn)
        spec = importlib.util.spec_from_file_location("agent", agent_dir / "agent.py")
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load {agent_dir / 'agent.py'}")
        module = importlib.util.module_from_spec(spec)
        sys.modules["agent"] = module
        spec.loader.exec_module(module)
        module.solve(task, robot, llm)
    except SystemExit as e:
        return 0 if e.code in (0, None) else 1
    except BaseException:
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
