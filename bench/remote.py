"""The commands that talk to the competition's gateway: login, submit, status, pick and
``train --remote``. Each returns ``(payload, text, exit code)`` like the other commands."""

from __future__ import annotations

import json
import math
import shutil
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from harness.llm_relay import DEFAULT_MODEL

from .archive import ArchiveProblem, build
from .config import Config, write_private
from .exitcodes import CliError, ExitCode
from .gateway import Gateway, GatewayError, cli_error
from .infra import say
from .runs import SCORED, list_episodes, read_json
from .train import TrainOptions, finish_run, prepare_run

POLL_S = 10.0
MAX_WAIT_S = 6 * 3600.0
REMOTE_MAX_EPISODES = 50
REMOTE_DEFAULT_SEEDS = 7  # 7 tasks x 7 seeds = 49 episodes, one remote run
RESULTS_MAX_BYTES = 2 * 1024**3
FINISHED = ("done", "timed_out", "infra_error")

Result = tuple[dict[str, Any], str, ExitCode]


def sleep(seconds: float) -> None:  # tests replace this
    time.sleep(seconds)


def now() -> float:  # tests replace this
    return time.time()


def gateway_for(config: Config, need_key: bool = True) -> Gateway:
    if not config.gateway_url:
        raise CliError(
            ExitCode.NOT_AVAILABLE,
            "no gateway configured: the competition's server is not announced yet (or set "
            "INKLING_GATEWAY_URL)",
        )
    key = config.read_key()
    if need_key and not key:
        raise CliError(ExitCode.NOT_READY, "not logged in. Fix: bench login")
    return Gateway(config.gateway_url, key)


def _minutes(seconds: float) -> str:
    m = max(1, math.ceil(seconds / 60))
    return f"{m} minute{'s' * (m != 1)}"


def _box_note(item: dict[str, Any]) -> str | None:
    box = item.get("box") or {}
    if box.get("state") in ("asleep", "waking"):
        return f"waking the shared model, about {_minutes(box.get('eta_s') or 300)}"
    return None


# --- login ------------------------------------------------------------------------------------
def login(config: Config, no_wait: bool = False) -> Result:
    gw = gateway_for(config, need_key=False)
    pending_path = config.state_dir / "login.json"
    pending = read_json(pending_path) or {}
    if not (
        pending.get("gateway") == config.gateway_url and pending.get("expires_at", 0) > now() + 10
    ):
        try:
            start = gw.login_start()
        except GatewayError as e:
            raise cli_error(e) from None
        pending = {
            "gateway": config.gateway_url,
            "login_id": start["login_id"],
            "user_code": start["user_code"],
            "verification_uri": start["verification_uri"],
            "interval": int(start.get("interval", 5)),
            "expires_at": now() + int(start["expires_in"]),
        }
        write_private(pending_path, json.dumps(pending))
    left = pending["expires_at"] - now()
    say(
        f"To log in, open {pending['verification_uri']} and enter the code "
        f"{pending['user_code']} (valid for {_minutes(left)}). No permissions are requested."
    )
    if no_wait:
        payload = {
            "status": "pending",
            "user_code": pending["user_code"],
            "verification_uri": pending["verification_uri"],
            "expires_in": int(left),
        }
        text = "waiting for approval; run bench login again to finish once the code is approved"
        return payload, text, ExitCode.OK
    interval = pending["interval"]
    while now() < pending["expires_at"]:
        sleep(interval)
        try:
            answer = gw.login_poll(pending["login_id"])
        except GatewayError as e:
            if e.status in (403, 404, 410):
                pending_path.unlink(missing_ok=True)
                raise CliError(ExitCode.NOT_READY, f"{e.message}") from None
            if e.status is None or e.status >= 500 or e.status == 429:
                interval = min(interval * 2, 60)
                continue
            raise cli_error(e) from None
        if answer.get("status") == "ok":
            config.save_key(answer["key"])
            pending_path.unlink(missing_ok=True)
            payload = {
                "status": "ok",
                "account": answer["account"],
                "key_path": str(config.key_path),
            }
            return (
                payload,
                f"logged in as {answer['account']}; key saved to {config.key_path}",
                ExitCode.OK,
            )
        interval = int(answer.get("interval", interval))
    pending_path.unlink(missing_ok=True)
    raise CliError(
        ExitCode.NOT_READY, "the code expired before it was approved; run bench login again"
    )


# --- submissions -------------------------------------------------------------------------------
def _archive(agent_dir: Path):
    try:
        return build(agent_dir)
    except ArchiveProblem as e:
        raise CliError(
            ExitCode.NOT_READY, "agent/ cannot be uploaded: " + "; ".join(e.problems)
        ) from None


def submit(config: Config, agent_dir: Path) -> Result:
    gw = gateway_for(config)
    archive = _archive(agent_dir)
    try:
        sub = gw.submit(archive.data)
    except GatewayError as e:
        raise cli_error(e) from None
    lines = [
        f"submitted {sub['id']}: {sub['state']}, position {sub.get('position') or '-'} in the queue"
    ]
    if note := _box_note(sub):
        lines.append(note)
    lines.append(
        f"the public score arrives when it is scored; check with: bench status {sub['id']}"
    )
    return sub, "\n".join(lines), ExitCode.OK


def _score_text(s: dict[str, Any]) -> str:
    if s.get("score") is None:
        return ""
    return (
        f"{s['score']:.1f} (95% {s['ci_low']:.1f}-{s['ci_high']:.1f}) tokens {s['tokens_bucket']}"
    )


def _row(s: dict[str, Any]) -> str:
    detail = _score_text(s)
    if s["state"] == "queued":
        detail = f"position {s.get('position')}" + (f"; {n}" if (n := _box_note(s)) else "")
    elif s["state"] not in ("done", "running"):
        detail = s.get("message", "")
    return f"{s['id']:<8}{s['state']:<12}{s['created_at'][:16].replace('T', ' '):<18}{detail}"


def status(config: Config, ident: str | None) -> Result:
    gw = gateway_for(config)
    try:
        if ident and ident.startswith("t"):
            run = gw.train_run(ident)
            return (
                run,
                _row(run) + (f"\nresults: {run['results']}" if run.get("results") else ""),
                ExitCode.OK,
            )
        if ident:
            sub = gw.submission(ident)
            text = _row(sub) + (f"\nrecord: {sub['commit_url']}" if sub.get("commit_url") else "")
            return sub, text, ExitCode.OK
        data = gw.submissions()
    except GatewayError as e:
        raise cli_error(e) from None
    limits = data["limits"]
    lines = [_row(s) for s in data["submissions"]] or ["no submissions yet"]
    lines.append(
        f"{limits['used_24h']} of {limits['per_24h']} submissions in the last 24 hours"
        + (f"; next allowed at {limits['next_at']}" if limits.get("next_at") else "")
    )
    return data, "\n".join(lines), ExitCode.OK


def pick(config: Config, ids: list[str]) -> Result:
    gw = gateway_for(config)
    try:
        data = gw.pick(ids) if ids else gw.picks()
    except GatewayError as e:
        raise cli_error(e) from None
    w = data["window"]
    window = (
        f"window {w['start']} to {w['end']} ({'open' if w['open'] else 'closed'})"
        if w.get("start")
        else "window not announced"
    )
    text = f"final picks ({data['source']}): {', '.join(data['picks']) or 'none'}; {window}"
    return data, text, ExitCode.OK


# --- remote training runs ------------------------------------------------------------------------
def remote_plan(seeds: dict[str, list[int]]) -> int:
    total = sum(len(v) for v in seeds.values())
    if total > REMOTE_MAX_EPISODES:
        raise CliError(
            ExitCode.USAGE,
            f"{total} episodes; a remote run has at most {REMOTE_MAX_EPISODES}. Use --seeds "
            f"{REMOTE_DEFAULT_SEEDS} (7 tasks x {REMOTE_DEFAULT_SEEDS}) or fewer --tasks",
        )
    return total


def _wait(gw: Gateway, run_id: str) -> dict[str, Any]:
    started, last, failures = now(), None, 0
    while True:
        try:
            run = gw.train_run(run_id)
            failures = 0
        except GatewayError as e:
            if e.status is None or e.status >= 500 or e.status == 429:
                failures += 1
                if failures > 30:
                    raise cli_error(e) from None
                sleep(POLL_S)
                continue
            raise cli_error(e) from None
        if run["state"] in FINISHED:
            return run
        note = (
            f"{run_id}: queued, position {run.get('position')}"
            if run["state"] == "queued"
            else f"{run_id}: running on the shared box"
        )
        box = _box_note(run) if run["state"] == "queued" else None
        if (note, box) != last:
            say(note)
            if box:
                say(box)
            last = (note, box)
        if now() - started > MAX_WAIT_S:
            raise CliError(
                ExitCode.INFRA,
                f"{run_id} is still {run['state']}; rerun the same "
                "command (same --out) to keep waiting",
            )
        sleep(POLL_S)


def extract_results(archive: Path, out: Path, plan: dict[str, list[int]]) -> int:
    """Unpack downloaded episode folders into ``out``, accepting only regular files and folders
    under ``<task>/seed_<seed>/`` of planned episodes. Returns the number of files."""
    allowed = {f"{t}/seed_{s}" for t, seeds in plan.items() for s in seeds}
    incoming = out / ".incoming"
    shutil.rmtree(incoming, ignore_errors=True)
    files, total = 0, 0
    try:
        with tarfile.open(archive, "r:gz") as tar:
            for m in tar:
                name = m.name.removeprefix("./")
                if name in ("", "."):
                    continue
                parts = name.split("/")
                if (
                    m.name.startswith("/")
                    or "\\" in name
                    or ":" in name
                    or any(p in ("", ".", "..") for p in parts)
                ):
                    raise CliError(ExitCode.INFRA, f"refused path in the results: {m.name!r}")
                if not (m.isreg() or m.isdir()):
                    raise CliError(ExitCode.INFRA, f"refused member in the results: {m.name!r}")
                if len(parts) == 1 and m.isdir() and parts[0] in plan:
                    continue
                if "/".join(parts[:2]) not in allowed:
                    raise CliError(ExitCode.INFRA, f"unexpected path in the results: {m.name!r}")
                target = incoming.joinpath(*parts)
                if m.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                total += m.size
                if total > RESULTS_MAX_BYTES:
                    raise CliError(ExitCode.INFRA, "results larger than expected")
                target.parent.mkdir(parents=True, exist_ok=True)
                src = tar.extractfile(m)
                with target.open("wb") as f:
                    shutil.copyfileobj(src, f)
                files += 1
        for episode in sorted(allowed):
            task, seed = episode.split("/")
            if (incoming / task / seed).is_dir():
                dest = out / task / seed
                shutil.rmtree(dest, ignore_errors=True)
                dest.parent.mkdir(parents=True, exist_ok=True)
                (incoming / task / seed).rename(dest)
    except (tarfile.TarError, EOFError) as e:
        raise CliError(ExitCode.INFRA, f"downloaded results are damaged ({e}); rerun") from None
    finally:
        shutil.rmtree(incoming, ignore_errors=True)
    return files


def train_remote(config: Config, agent_dir: Path, seeds: dict[str, list[int]], out: Path) -> dict:
    """Run the plan on the shared box, download the episode folders into ``out`` and score them
    exactly as a local run. Rerunning with the same ``out`` resumes: it waits for a run already
    submitted, or submits only the episodes still missing."""
    gw = gateway_for(config)
    opts = TrainOptions(
        seeds=seeds,
        agent_dir=agent_dir,
        model_url="remote",
        model=DEFAULT_MODEL,
        api_key=None,
        out=out,
    )
    remote_plan(seeds)
    prepare_run(opts, where="remote")
    planned = [(t, s) for t, ss in seeds.items() for s in ss]
    state_path = out / "remote.json"
    state = read_json(state_path) or {}
    if state.get("downloaded", True):
        done = {(e.task, e.seed) for e in list_episodes(out) if e.status == SCORED}
        missing = {t: [s for s in ss if (t, s) not in done] for t, ss in seeds.items()}
        missing = {t: ss for t, ss in missing.items() if ss}
        if missing:
            archive = _archive(agent_dir)
            try:
                run = gw.train(archive.data, missing)
            except GatewayError as e:
                raise cli_error(e) from None
            state = {
                "id": run["id"],
                "plan": missing,
                "downloaded": False,
                "submitted": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            state_path.write_text(json.dumps(state, indent=1))
            say(f"remote run {run['id']}: {sum(len(v) for v in missing.values())} episodes queued")
    if not state.get("downloaded", True):
        run = _wait(gw, state["id"])
        if run["state"] != "done":
            state["downloaded"] = True  # nothing to download: a rerun submits again
            state["ended"] = run["state"]
            state_path.write_text(json.dumps(state, indent=1))
            raise CliError(
                ExitCode.INFRA,
                f"remote run {state['id']} ended as {run['state']}: "
                f"{run.get('message', '')}. Rerun with the same --out to try again",
            )
        say(f"{state['id']}: done; downloading the episode folders")
        tmp = out / ".results.tar.gz"
        try:
            gw.train_results(state["id"], tmp, RESULTS_MAX_BYTES)
            extract_results(tmp, out, state["plan"])
        except GatewayError as e:
            raise cli_error(e) from None
        finally:
            tmp.unlink(missing_ok=True)
        state["downloaded"] = True
        state_path.write_text(json.dumps(state, indent=1))
    return finish_run(opts, planned)
