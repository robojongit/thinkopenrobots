"""Run CaP-X's reference solutions (the privileged oracles) through the harness.

Each task's oracle code is read from its CaP-X privileged oracle config, served by a scripted
fake model inside one ```python block, and run by a minimal agent with ``run_episode`` in
plain-process mode: the code reaches the robot only through ``robot.run``, so it passes the
"substring of a model reply" check like any model-written code.

    python -m harness.tests.reference [--tasks a,b] [--seeds 1-10] [--native] [--out DIR]

``--native`` also runs CaP-X's own launch path (``capx/envs/launch.py`` with
``--use-oracle-code``, one worker) on the same seeds and prints both success rates. CaP-X numbers
its trials 1..N and resets trial ``k`` with ``seed=k``, so native seeds are always 1..N; the
harness seeds must then be 1..N too.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

from harness.capx_env import capx_root, import_capx
from harness.episode import run_episode
from harness.llm_relay import Budget
from harness.tasks import TASKS
from harness.tests.fake_model import FakeModel

ORACLE_DIR = "env_configs/human_oracle_code/robosuite"
ORACLE_CONFIGS = {
    "cube_lifting": "franka_robosuite_cube_lifting_privileged_oracle.yaml",
    "cube_stack": "franka_robosuite_cube_stack_privileged_oracle.yaml",
    "cube_restack": "franka_robosuite_cube_restack_privileged_oracle.yaml",
    "nut_assembly": "franka_robosuite_nut_assembly_privileged_oracle.yaml",
    "spill_wipe": "franka_robosuite_spill_wipe_privileged_oracle.yaml",
    "two_arm_lift": "franka_robosuite_two_arm_lift_privileged_oracle.yaml",
    "two_arm_handover": "two_arm_handover_privileged_oracle.yaml",
}
BUDGET = Budget(calls=1, tokens=1_000_000, wall_s=900.0)

AGENT = """
import re


def solve(task, robot, llm):
    prompt = task.instruction + "\\nAPIs:\\n" + task.api_docs
    reply = llm.chat([{"role": "user", "content": prompt}])
    code = re.search(r"```python\\n(.*?)```", reply, re.S).group(1)
    result = robot.run(code)
    print("ok:", result.ok)
    print(result.stdout[-4000:])
    print(result.stderr[-4000:])
"""


def oracle_config(task: str) -> Path:
    return capx_root() / ORACLE_DIR / ORACLE_CONFIGS[task]


def oracle_code(task: str) -> str:
    """The oracle code CaP-X runs for ``task``: the config's ``oracle_code`` if it sets one,
    else the ``oracle_code`` attribute of the config's env class. Also checks that the oracle
    config uses the same low-level env and API as the task's privileged config."""
    import_capx()
    from capx.envs.configs.instantiate import locate
    from capx.envs.configs.loader import DictLoader

    env = DictLoader.load([str(oracle_config(task))])["env"]
    cfg = env["cfg"]
    spec = TASKS[task]
    if list(cfg["apis"]) != [spec.api]:
        raise RuntimeError(f"{task}: oracle config uses APIs {cfg['apis']}, not {spec.api}")
    task_cfg = DictLoader.load([str(capx_root() / spec.config)])["env"]
    low = cfg["low_level"]
    low = low if isinstance(low, str) else low["_target_"]
    if low != task_cfg["cfg"]["low_level"] and not low.endswith(_LOW_LEVEL_CLASSES.get(task, "?")):
        raise RuntimeError(f"{task}: oracle low-level env {low} differs from the task's")
    if env["_target_"] != task_cfg["_target_"]:
        raise RuntimeError(f"{task}: oracle env class {env['_target_']} differs from the task's")
    code = cfg.get("oracle_code") or locate(env["_target_"]).oracle_code
    if not code or not code.strip():
        raise RuntimeError(f"{task}: no oracle code")
    return code


# The handover oracle config names its low-level env by class instead of registry name.
_LOW_LEVEL_CLASSES = {"two_arm_handover": "robosuite_handover.RobosuiteHandoverEnv"}


def reply_for(code: str) -> str:
    return f"Reference solution.\n```python\n{code.strip()}\n```\n"


def write_agent(folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "agent.py").write_text(AGENT)
    return folder


def run_reference(
    task: str, seed: int, out: Path, *, video: bool = False, code: str | None = None
) -> dict[str, Any]:
    """One harness episode of ``task``'s oracle on ``seed``. Returns the episode result plus
    ``build_and_run_s`` (wall clock including the env build)."""
    reply = reply_for(code if code is not None else oracle_code(task))
    agent = write_agent(out / "agent")
    started = time.monotonic()
    with FakeModel([reply]) as model:
        result = run_episode(task, seed, agent, model.url, BUDGET, out / "episode", video)
    result["build_and_run_s"] = round(time.monotonic() - started, 1)
    return result


def run_native(task: str, trials: int, out: Path) -> dict[int, bool]:
    """CaP-X's own launch path, one worker, trials 1..``trials`` (the oracle configs set
    ``use_oracle_code: true``): seed -> task completed."""
    cmd = [
        sys.executable,
        "capx/envs/launch.py",
        "--config-path",
        f"{ORACLE_DIR}/{ORACLE_CONFIGS[task]}",
        "--total-trials",
        str(trials),
        "--num-workers",
        "1",
        "--output-dir",
        str(out / task),
    ]
    out.mkdir(parents=True, exist_ok=True)
    log = out / f"{task}.log"
    with open(log, "w") as f:
        subprocess.run(cmd, cwd=capx_root(), stdout=f, stderr=subprocess.STDOUT, check=False)
    found: dict[int, bool] = {}
    # CaP-X inserts the model name: <out>/<model>/<task>/trial_NN_..._taskcompleted_X
    for d in out.glob(f"*/{task}/trial_*_taskcompleted_*"):
        m = re.match(r"trial_(\d+)_.*_taskcompleted_(\d)$", d.name)
        if m:
            found[int(m[1])] = m[2] == "1"
    return found


def parse_seeds(text: str) -> list[int]:
    seeds: list[int] = []
    for part in text.split(","):
        lo, _, hi = part.partition("-")
        seeds.extend(range(int(lo), int(hi or lo) + 1))
    return seeds


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m harness.tests.reference")
    parser.add_argument("--tasks", default=",".join(TASKS))
    parser.add_argument("--seeds", default="1-10", help="e.g. 1-10 or 3,5,8")
    parser.add_argument("--native", action="store_true", help="also run CaP-X's launch path")
    parser.add_argument("--out", type=Path, default=None, help="keep episode folders here")
    parser.add_argument("--json", type=Path, default=None, help="write all results here")
    args = parser.parse_args(argv)
    tasks = args.tasks.split(",")
    seeds = parse_seeds(args.seeds)
    if args.native and seeds != list(range(1, len(seeds) + 1)):
        parser.error("--native needs seeds 1..N (CaP-X seeds trial k with k)")

    from harness.ik_server import ensure_ik_server

    ensure_ik_server()
    out = args.out or Path(tempfile.mkdtemp(prefix="inkling-reference-"))
    report: dict[str, Any] = {}
    for task in tasks:
        rows = []
        for seed in seeds:
            try:
                r = run_reference(task, seed, out / task / str(seed))
                rows.append(
                    {
                        "seed": seed,
                        "completed": json.loads(
                            (out / task / str(seed) / "episode" / "result.json").read_text()
                        )["task_completed"],
                        "seconds": r["seconds"],
                        "build_and_run_s": r["build_and_run_s"],
                        "ended": r["ended"],
                    }
                )
            except Exception as e:
                traceback.print_exc()
                rows.append({"seed": seed, "completed": False, "error": repr(e)})
            print(task, rows[-1], flush=True)
        native = run_native(task, len(seeds), out / "native") if args.native else {}
        report[task] = {"harness": rows, "native": {str(k): v for k, v in sorted(native.items())}}
        h = sum(r["completed"] for r in rows)
        n = sum(native.values())
        print(
            f"== {task}: harness {h}/{len(rows)}"
            + (f", native {n}/{len(native)}" if args.native else ""),
            flush=True,
        )
    if args.json:
        args.json.write_text(json.dumps(report, indent=1))
    print(f"episode folders: {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    sys.exit(main())
