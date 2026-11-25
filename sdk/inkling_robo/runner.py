"""Code-runner entry point: ``python -m inkling_robo.runner [--socket PATH]``.

Authenticates with the run's token (``INKLING_ROBO_TOKEN``, removed from the environment before
any other code runs), fetches the model-written code from the sim server, puts every robot API
function of the task (reads and motion) in scope, and executes the code. Nothing else is loaded:
no participant-written code runs in the runner. Exit status 0 means the code ran without
raising; a traceback goes to stderr otherwise.
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from typing import Any

from .client import Connection, api_function, line_buffered_output

DEFAULT_SOCKET = "/run/sim.sock"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m inkling_robo.runner")
    parser.add_argument("--socket", default=os.environ.get("INKLING_ROBO_SOCKET", DEFAULT_SOCKET))
    args = parser.parse_args(argv)
    line_buffered_output()

    token = os.environ.pop("INKLING_ROBO_TOKEN", "")
    conn = Connection(args.socket)
    info = conn.call("runner.hello", {"token": token})
    api = {name: api_function(conn, name) for name in info["functions"]}

    namespace: dict[str, Any] = {"__name__": "__main__", **api}
    try:
        # Same filename as CaP-X's exec(), so tracebacks read the same to the model.
        exec(compile(info["code"], "<string>", "exec"), namespace, namespace)
    except SystemExit as e:
        return 0 if e.code in (0, None) else 1
    except BaseException:
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
