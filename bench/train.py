"""`bench train`: run the training split and write the run folder, score.json and results.tsv."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import multiprocessing
import shutil
import traceback
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any

from harness.episode import EnvFactory, run_episode
from harness.llm_relay import Budget
from harness.sandbox import make_sandbox

from .exitcodes import CliError, ExitCode
from .infra import Probe, classify, probe, say, wait_until_ready
from .runs import CRASH, INFRA, SCORED, episode_dir, list_episodes, read_json
from .score import compute_score

RESULTS_COLUMNS = (
    "run_id finished overall ci_low ci_high episodes planned infra_failed crashed model agent_sha"
    " tokens_mean seconds_mean"
).split()


@dataclass(frozen=True)
class TrainOptions:
    seeds: dict[str, list[int]]
    agent_dir: Path
    model_url: str
    model: str
    api_key: str | None
    out: Path
    workers: int = 2
    video: bool = False
    retries: int = 2
    default_wait_s: float = 10.0
    max_wait_s: float = 900.0
    results_tsv: Path = Path("results.tsv")
    budget: Budget = Budget()
    """Per-episode budget; the default is ``harness.llm_relay.Budget``'s."""
    sandbox: str = "process"
    """"process" (plain subprocesses) or "container" (Docker, on ``runtime``)."""
    runtime: str = "runsc"


@dataclass(frozen=True)
class Job:
    """One episode to run, picklable so it can go to a worker process."""

    task: str
    seed: int
    options: TrainOptions
    make_env: EnvFactory | None

    @property
    def out(self) -> Path:
        return episode_dir(self.options.out, self.task, self.seed)


def agent_fingerprint(agent_dir: Path) -> str:
    """Hash of every .py file under the agent folder (notes and memory files don't count)."""
    digest = hashlib.sha256()
    for path in sorted(agent_dir.rglob("*.py")):
        if "__pycache__" not in path.parts:
            digest.update(str(path.relative_to(agent_dir)).encode() + b"\0" + path.read_bytes())
    return digest.hexdigest()[:12]


def wait_for_server(opts: TrainOptions) -> Probe:
    return wait_until_ready(
        partial(probe, opts.model_url, opts.model, opts.api_key),
        default_wait_s=opts.default_wait_s,
        max_wait_s=opts.max_wait_s,
    )


def run_job(job: Job) -> dict[str, Any]:
    """Run one episode, retrying after model-server trouble. Returns task, seed, status, detail."""
    opts = job.options
    summary = {"task": job.task, "seed": job.seed, "status": SCORED, "detail": ""}
    for attempt in range(opts.retries + 1):
        shutil.rmtree(job.out, ignore_errors=True)
        try:
            result = run_episode(
                job.task,
                job.seed,
                opts.agent_dir,
                opts.model_url,
                opts.budget,
                job.out,
                opts.video,
                model=opts.model,
                api_key=opts.api_key,
                make_env=job.make_env,
                sandbox=make_sandbox(opts.sandbox, opts.runtime),
            )
        except Exception as e:  # a harness bug or a missing simulator, not a task failure
            detail = f"{type(e).__name__}: {e}"
            job.out.mkdir(parents=True, exist_ok=True)
            (job.out / "crash.json").write_text(
                json.dumps({"error": detail, "traceback": traceback.format_exc()}, indent=1)
            )
            return {**summary, "status": CRASH, "detail": detail}
        problem = None if result["success"] else classify(job.out)
        if problem is None:
            return {**summary, "detail": "success" if result["success"] else "failed"}
        (job.out / "infra.json").write_text(
            json.dumps({"detail": problem, "attempts": attempt + 1}, indent=1)
        )
        if attempt == opts.retries:
            break
        if not wait_for_server(opts).ok:
            break
    return {**summary, "status": INFRA, "detail": problem}


def check_server(opts: TrainOptions) -> None:
    """Fail early, and distinctly, if the model server is down or rejects our requests."""
    ready = wait_for_server(opts)
    if ready.ok:
        return
    if ready.transient:
        raise CliError(ExitCode.INFRA, f"model server not available ({ready.error or 'busy'})")
    raise CliError(
        ExitCode.NOT_READY,
        f"model server rejected a test request ({ready.error}); check --model-url, --model and "
        "the key",
    )


def prepare_run(opts: TrainOptions, where: str = "local") -> None:
    """Create the run folder, or check that a resumed run has the same agent, model, budget and
    place (local or the shared box)."""
    meta_path = opts.out / "run.json"
    meta = {
        "agent_sha": agent_fingerprint(opts.agent_dir),
        "model": opts.model,
        "budget": dataclasses.asdict(opts.budget),
        "where": where,
    }
    old = read_json(meta_path)
    if old and {k: old.get(k, "local" if k == "where" else None) for k in meta} != meta:
        raise CliError(
            ExitCode.USAGE,
            f"{opts.out} was started with a different agent, model, budget or place (local or "
            "--remote); resuming would mix results. Use a new --out",
        )
    opts.out.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(
        json.dumps({**meta, "seeds": opts.seeds, "started": datetime.now().isoformat()}, indent=1)
    )


def run_pool(jobs: list[Job], workers: int) -> None:
    """Run the jobs ``workers`` at a time; stop starting new ones once the model server has
    defeated an episode's retries."""
    queue = iter(jobs)
    pool = ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn"))
    running: set[Future] = set()
    finished, stopped = 0, False
    try:
        while True:
            while not stopped and len(running) < workers and (job := next(queue, None)):
                running.add(pool.submit(run_job, job))
            if not running:
                return
            done, running = wait(running, return_when=FIRST_COMPLETED)
            for future in done:
                row = future.result()
                finished += 1
                say(f"[{finished}/{len(jobs)}] {row['task']} seed {row['seed']}: {row['detail']}")
                if row["status"] == INFRA and not stopped:
                    stopped = True
                    say("model server still failing after retries; not starting more episodes")
    finally:
        pool.shutdown(wait=True, cancel_futures=True)


def write_results_row(opts: TrainOptions, score: dict[str, Any]) -> None:
    """Add this run's row to results.tsv, replacing an earlier row of the same run."""
    path = opts.results_tsv
    rows = []
    if path.exists():
        rows = [
            ln for ln in path.read_text().splitlines()[1:] if ln.split("\t")[0] != score["run_id"]
        ]
    row = [
        score["run_id"],
        score["finished"],
        score["overall"],
        *score["ci95"],
        score["episodes"],
        score["planned"],
        score["infra_failed"],
        score["crashed"],
        score["model"],
        score["agent_sha"],
        score["tokens"]["mean"],
        score["seconds"]["mean"],
    ]
    rows.append("\t".join(map(str, row)))
    path.write_text("\n".join(["\t".join(RESULTS_COLUMNS), *rows]) + "\n")


def run_train(opts: TrainOptions, make_env: EnvFactory | None = None) -> dict[str, Any]:
    """Run (or resume) the training run and return the contents of score.json plus problems."""
    prepare_run(opts)
    planned = [(t, s) for t, seeds in opts.seeds.items() for s in seeds]
    done = {(e.task, e.seed) for e in list_episodes(opts.out) if e.status == SCORED}
    jobs = [Job(t, s, opts, make_env) for t, s in planned if (t, s) not in done]
    if jobs:
        check_server(opts)
        if make_env is None:  # the real simulator: one IK server, owned here, for all workers
            from harness.ik_server import ensure_ik_server

            ensure_ik_server()
    run_pool(jobs, opts.workers)
    return finish_run(opts, planned)


def finish_run(opts: TrainOptions, planned: list[tuple[str, int]]) -> dict[str, Any]:
    """Score the run folder's planned episodes: write score.json and the results.tsv row, and
    return the report. Shared by local runs and runs downloaded from the shared box."""
    episodes = [e for e in list_episodes(opts.out) if (e.task, e.seed) in planned]
    results = [e.result for e in episodes if e.status == SCORED]
    problems = {
        kind: [
            {"task": e.task, "seed": e.seed, **read_json(e.path / f"{kind}.json", {})}
            for e in episodes
            if e.status == kind
        ]
        for kind in (INFRA, CRASH)
    }
    score = {
        "run_id": opts.out.name,
        **compute_score(results),
        "planned": len(planned),
        "complete": len(results) == len(planned),
        "infra_failed": len(problems[INFRA]),
        "crashed": len(problems[CRASH]),
        "model": opts.model,
        "agent_sha": agent_fingerprint(opts.agent_dir),
        "budget": dataclasses.asdict(opts.budget),
        "finished": datetime.now().isoformat(timespec="seconds"),
    }
    (opts.out / "score.json").write_text(json.dumps(score, indent=1))
    write_results_row(opts, score)
    return {**score, "infra": problems[INFRA], "crash": problems[CRASH]}


def exit_code(report: dict[str, Any]) -> ExitCode:
    if report["infra_failed"] or not (report["complete"] or report["crashed"]):
        return ExitCode.INFRA
    return ExitCode.CRASHED if report["crashed"] else ExitCode.OK


def format_report(report: dict[str, Any]) -> str:
    low, high = report["ci95"]
    lines = [
        f"run {report['run_id']}: {report['episodes']}/{report['planned']} episodes scored",
        f"overall {report['overall']:.3f} (95% interval {low:.3f} to {high:.3f})",
        f"{'task':<18}{'rate':>7}{'wins':>6}{'n':>4}   95% interval",
    ]
    for task, t in report["per_task"].items():
        lines.append(
            f"{task:<18}{t['rate']:>7.3f}{t['successes']:>6}{t['n']:>4}"
            f"   {t['ci95'][0]:.3f} to {t['ci95'][1]:.3f}"
        )
    tokens, seconds = report["tokens"], report["seconds"]
    lines.append(
        f"tokens mean {tokens['mean']:.0f}, p90 {tokens['p90']:.0f}; "
        f"seconds mean {seconds['mean']:.1f}, p90 {seconds['p90']:.1f}"
    )
    lines.append("ended: " + ", ".join(f"{k}={v}" for k, v in sorted(report["ended"].items())))
    if report["infra"]:
        lines.append(
            f"INFRA: {report['infra_failed']} episodes not scored because of the model server "
            f"(first: {report['infra'][0]['detail']}); these are not task failures. "
            "Rerun with the same --out to finish them."
        )
    if report["crash"]:
        lines.append(
            f"CRASHED: {report['crashed']} episodes crashed in the harness "
            f"(first: {report['crash'][0]['error']}); see crash.json in the episode folder."
        )
    return "\n".join(lines)
