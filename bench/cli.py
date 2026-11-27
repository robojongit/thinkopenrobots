"""The `bench` command line: argument parsing and dispatch. Logic lives in the other modules."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, NoReturn

from harness.llm_relay import Budget
from harness.sandbox import agent_image, check_docker
from harness.tasks import TASKS

from . import doctor, remote, train
from .compare import compare_runs, format_comparison
from .config import Config, load_config
from .exitcodes import CliError, ExitCode, table
from .splits import listed_seeds, parse_seed_list, train_seeds
from .view import DEFAULT_LIMIT, format_view, view_run

EPILOG = "exit codes:\n" + table()


class Parser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        raise CliError(ExitCode.USAGE, f"{message}\n{self.format_usage().strip()}")


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--json", action="store_true", help="print one JSON document on stdout")
    parser = Parser(
        prog="bench",
        description="Run, inspect and compare robot-agent episodes. Never prompts.",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True, parser_class=Parser)

    def command(name: str, help: str) -> argparse.ArgumentParser:
        return sub.add_parser(
            name, help=help, parents=[common], epilog=EPILOG,
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )  # fmt: skip

    p = command("train", "run the training split with the local simulator")
    p.add_argument("--tasks", help=f"comma-separated subset of: {','.join(TASKS)}")
    seeds = p.add_mutually_exclusive_group()
    seeds.add_argument(
        "--seeds", type=int, metavar="N", help="first N training seeds per task (default: all)"
    )
    seeds.add_argument(
        "--seed-list",
        metavar="LIST",
        help='these seeds for every task instead of the training split, e.g. "1-50" or "3,7,10-12"',
    )
    p.add_argument("--agent", type=Path, default=Path("agent"), help="agent folder (default agent)")
    p.add_argument("--model-url", help="OpenAI-compatible base URL (default: from config)")
    p.add_argument("--model", help="model name sent to the server (default: from config)")
    p.add_argument("--workers", type=int, default=2, help="parallel episodes (default 2)")
    p.add_argument(
        "--video",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="write video.mp4 in each episode folder (about 4x slower); with --remote, for at"
        f" most {remote.REMOTE_MAX_VIDEOS} episodes of the run",
    )
    p.add_argument("--out", type=Path, help="run folder (default runs/<timestamp>); resumes it")
    p.add_argument(
        "--verbose",
        action="store_true",
        help="show the simulator's and libraries' own output (by default it goes to sim.log in"
        " the run folder)",
    )
    p.add_argument(
        "--remote",
        action="store_true",
        help="simulate on the shared box: up to 50 episodes per run, training seeds only (default"
        f" --seeds {remote.REMOTE_DEFAULT_SEEDS}); the default on native Windows",
    )
    default = Budget()
    p.add_argument(
        "--budget-calls", type=int, metavar="N",
        help=f"model calls per episode (default {default.calls})",
    )  # fmt: skip
    p.add_argument(
        "--budget-tokens", type=int, metavar="N",
        help=f"prompt + completion tokens per episode (default {default.tokens})",
    )  # fmt: skip
    p.add_argument(
        "--budget-wall-s", type=float, metavar="S",
        help=f"wall-clock seconds per episode (default {default.wall_s:g})",
    )  # fmt: skip
    p.add_argument(
        "--sandbox",
        choices=["process", "container"],
        default="process",
        help="run the agent as plain processes (default) or in containers, as scoring does",
    )
    p.add_argument(
        "--runtime",
        choices=["runsc", "runc"],
        default="runsc",
        help="container runtime: runsc (gVisor, default, as scoring) or runc",
    )

    p = command("view", "show readable traces of episodes in a run")
    p.add_argument("run", type=Path)
    p.add_argument("--failures", action="store_true", help="only episodes that did not succeed")
    p.add_argument("--task", choices=list(TASKS))
    p.add_argument("--seed", type=int)
    p.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help="full traces to show")
    p.add_argument("--full", action="store_true", help="do not truncate replies, code or output")

    p = command("compare", "paired comparison of two runs on shared (task, seed) pairs")
    p.add_argument("run_a", type=Path)
    p.add_argument("run_b", type=Path)

    p = command("doctor", "check the setup and print exact fixes")
    p.add_argument("--agent", type=Path, default=Path("agent"))
    p.add_argument("--model-url")

    p = command("login", "log in with GitHub (device code, no permissions) and save the key")
    p.add_argument(
        "--no-wait",
        action="store_true",
        help="print the code and return; run bench login again to finish after approval",
    )

    p = command("submit", "upload agent/ for scoring on the public test (3 per 24 hours)")
    p.add_argument("--agent", type=Path, default=Path("agent"), help="agent folder (default agent)")

    p = command("status", "your submissions: queue position, public scores (or one id)")
    p.add_argument("id", nargs="?", help="a submission (s...) or remote run (t...) id")

    p = command("pick", "choose 1-2 final submissions in the final window (no ids: show picks)")
    p.add_argument("ids", nargs="*", metavar="ID")
    return parser


Result = tuple[dict[str, Any], str, ExitCode]


def get_config() -> Config:
    try:
        return load_config()
    except ValueError as e:
        raise CliError(
            ExitCode.NOT_READY, f"{e} (check INKLING_GATEWAY_URL or the config file)"
        ) from None


def cmd_train(args: argparse.Namespace) -> Result:
    remote_run = args.remote or doctor.is_native_windows()
    if remote_run and not args.remote:
        print("bench: native Windows: simulating on the shared box (--remote)", file=sys.stderr)
    tasks = args.tasks.split(",") if args.tasks else None
    unknown = [t for t in tasks or [] if t not in TASKS]
    if unknown:
        raise CliError(ExitCode.USAGE, f"unknown tasks {unknown}; choose from {list(TASKS)}")
    if args.seeds is not None and not 1 <= args.seeds <= 30:
        raise CliError(ExitCode.USAGE, "--seeds must be between 1 and 30")
    if args.workers < 1:
        raise CliError(ExitCode.USAGE, "--workers must be at least 1")
    if args.seed_list is not None:
        try:
            seed_list = parse_seed_list(args.seed_list)
        except ValueError:
            raise CliError(
                ExitCode.USAGE,
                f'bad --seed-list {args.seed_list!r}; use e.g. "1-50" or "3,7,10-12"',
            ) from None
    limits = {
        "calls": args.budget_calls,
        "tokens": args.budget_tokens,
        "wall_s": args.budget_wall_s,
    }
    if any(v is not None and v <= 0 for v in limits.values()):
        raise CliError(ExitCode.USAGE, "--budget-* values must be positive")
    budget = Budget(**{k: v for k, v in limits.items() if v is not None})
    if remote_run:
        return train_on_box(args, tasks, seed_list if args.seed_list is not None else None)

    config = get_config()
    url = config.resolve_model_url(args.model_url)
    if not url:
        raise CliError(
            ExitCode.NOT_READY,
            "no model URL: pass --model-url (for example http://localhost:11434/v1) or set "
            "INKLING_MODEL_URL",
        )
    for check in (doctor.check_agent(args.agent), doctor.check_simulator()):
        if check.status == doctor.FAIL:
            raise CliError(ExitCode.NOT_READY, f"{check.detail}. Fix: {check.fix}")
    if args.sandbox == "container":
        try:
            check_docker(args.runtime, agent_image())
        except RuntimeError as e:
            raise CliError(ExitCode.NOT_READY, f"--sandbox container: {e}") from None
    opts = train.TrainOptions(
        seeds=train_seeds(tasks, args.seeds)
        if args.seed_list is None
        else listed_seeds(tasks, seed_list),
        agent_dir=args.agent,
        model_url=url,
        model=args.model or config.model,
        api_key=config.api_key_for(url),
        out=args.out or Path("runs") / datetime.now().strftime("%Y%m%d-%H%M%S"),
        workers=args.workers,
        budget=budget,
        video=args.video,
        sandbox=args.sandbox,
        runtime=args.runtime,
        verbose=args.verbose,
        gateway_url=config.gateway_url if config.is_gateway_url(url) else None,
    )
    report = train.run_train(opts)
    return report, train.format_report(report), train.exit_code(report)


def train_on_box(args: argparse.Namespace, tasks: list[str] | None, seed_list) -> Result:
    local_only = {
        "--model-url": args.model_url,
        "--model": args.model,
        "--budget-calls": args.budget_calls,
        "--budget-tokens": args.budget_tokens,
        "--budget-wall-s": args.budget_wall_s,
        "--sandbox container": args.sandbox == "container" or None,
    }
    used = [flag for flag, value in local_only.items() if value is not None]
    if used:
        raise CliError(
            ExitCode.USAGE,
            f"{', '.join(used)} cannot be used with --remote: the shared box runs the fixed "
            "model and budget in containers",
        )
    check = doctor.check_agent(args.agent)
    if check.status == doctor.FAIL:
        raise CliError(ExitCode.NOT_READY, f"{check.detail}. Fix: {check.fix}")
    if seed_list is not None:
        seeds = listed_seeds(tasks, seed_list)
        split = train_seeds(tasks)
        outside = sorted({s for t, ss in seeds.items() for s in ss if s not in split[t]})
        if outside:
            raise CliError(
                ExitCode.USAGE,
                f"--remote runs training seeds only; not training seeds: {outside[:10]}",
            )
    else:
        seeds = train_seeds(tasks, args.seeds or remote.REMOTE_DEFAULT_SEEDS)
    remote.remote_plan(seeds)
    out = args.out or Path("runs") / datetime.now().strftime("%Y%m%d-%H%M%S")
    report = remote.train_remote(get_config(), args.agent, seeds, out, video=args.video)
    text = train.format_report(report)
    if "videos" in report:
        text += "\n" + remote.format_videos(report)
    return report, text, train.exit_code(report)


def cmd_view(args: argparse.Namespace) -> Result:
    view = view_run(
        args.run,
        failures=args.failures,
        task=args.task,
        seed=args.seed,
        limit=args.limit,
        full=args.full,
    )
    return view, format_view(view), ExitCode.OK


def cmd_compare(args: argparse.Namespace) -> Result:
    for run in (args.run_a, args.run_b):
        if not run.is_dir():
            raise CliError(ExitCode.NOT_FOUND, f"no such run folder: {run}")
    comparison = compare_runs(args.run_a, args.run_b)
    return comparison, format_comparison(comparison), ExitCode.OK


def cmd_doctor(args: argparse.Namespace) -> Result:
    config = get_config()
    checks = doctor.run_checks(args.agent, config.resolve_model_url(args.model_url), config)
    failed = [c.name for c in checks if c.status == doctor.FAIL]
    payload = {"ok": not failed, "failed": failed, "checks": doctor.as_dicts(checks)}
    text = doctor.format_checks(checks)
    if failed:
        print(f"bench: doctor: failed checks: {', '.join(failed)}", file=sys.stderr)
    return payload, text, ExitCode.NOT_READY if failed else ExitCode.OK


def cmd_login(args: argparse.Namespace) -> Result:
    return remote.login(get_config(), no_wait=args.no_wait)


def cmd_submit(args: argparse.Namespace) -> Result:
    return remote.submit(get_config(), args.agent)


def cmd_status(args: argparse.Namespace) -> Result:
    return remote.status(get_config(), args.id)


def cmd_pick(args: argparse.Namespace) -> Result:
    if len(args.ids) > 2:
        raise CliError(ExitCode.USAGE, "pick at most 2 submissions")
    return remote.pick(get_config(), args.ids)


COMMANDS: dict[str, Callable[[argparse.Namespace], Result]] = {
    "train": cmd_train,
    "view": cmd_view,
    "compare": cmd_compare,
    "doctor": cmd_doctor,
    "login": cmd_login,
    "submit": cmd_submit,
    "status": cmd_status,
    "pick": cmd_pick,
}


def main(argv: Sequence[str] | None = None) -> int:
    as_json = "--json" in (sys.argv[1:] if argv is None else argv)
    try:
        args = build_parser().parse_args(argv)
        payload, text, code = COMMANDS[args.command](args)
        print(json.dumps(payload, indent=1) if args.json else text)
        return int(code)
    except CliError as e:
        print(f"bench: error: {e.message}", file=sys.stderr)
        if as_json:
            error = {"code": int(e.code), "name": e.code.name.lower(), "message": e.message}
            print(json.dumps({"error": error}, indent=1))
        return int(e.code)
    except KeyboardInterrupt:
        print("bench: interrupted; rerun with the same --out to resume", file=sys.stderr)
        return int(ExitCode.INTERRUPTED)
