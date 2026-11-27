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
from harness.tasks import TASKS

from . import allowance
from .archive import ArchiveProblem, build
from .coding_agent import CodingAgentProblem, parse_notes
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
REMOTE_MAX_VIDEOS = 10  # episodes of one remote run recorded with --video
REMOTE_VIDEO_MAX_BYTES = 8 * 1024**2  # one video; the shared box drops larger ones
RESULTS_MAX_BYTES = 2 * 1024**3  # the episode folders, videos not counted
VIDEO, VIDEO_NOTE = "video.mp4", "video_note.txt"
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


def coding_agent_of(agent_dir: Path) -> str:
    """The coding agent named in ``agent_dir/NOTES.md`` (CodingAgentProblem if not valid)."""
    notes = agent_dir / "NOTES.md"
    if notes.is_symlink() or not notes.is_file():
        raise CodingAgentProblem("agent/NOTES.md is missing")
    return parse_notes(notes.read_bytes())


def _submission(agent_dir: Path):
    """The archive and the coding agent to submit; every problem with either at once."""
    problems: list[str] = []
    archive = label = None
    try:
        archive = build(agent_dir)
    except ArchiveProblem as e:
        problems += e.problems
    try:
        label = coding_agent_of(agent_dir)
    except CodingAgentProblem as e:
        if not any("NOTES.md" in p for p in problems):  # missing or empty: already said
            problems.append(e.message)
    if problems or archive is None or label is None:
        raise CliError(ExitCode.NOT_READY, "agent/ cannot be submitted: " + "; ".join(problems))
    return archive, label


def submit(config: Config, agent_dir: Path) -> Result:
    gw = gateway_for(config)
    archive, label = _submission(agent_dir)
    say(f"submitting agent/ built by coding agent: {label}")
    try:
        sub = gw.submit(archive.data)
    except GatewayError as e:
        raise cli_error(e) from None
    position = sub.get("position") or "-"
    lines = [
        f"submitted {sub['id']}: {sub['state']}, position {position} in the queue",
        f"coding agent: {sub.get('coding_agent') or label} (shown on the leaderboard)",
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
            text = _row(run)
            if run.get("videos"):
                video = run.get("video") or {}
                video = {t: video[t] for t in TASKS if t in video}  # task-table order
                text += f"\nvideos: {run['videos']} ({episode_list(video)})"
            if run.get("results"):
                text += f"\nresults: {run['results']}"
            return run, text, ExitCode.OK
        if ident:
            sub = gw.submission(ident)
            text = _row(sub)
            if sub.get("coding_agent"):
                text += f"\ncoding agent: {sub['coding_agent']}"
            if sub.get("commit_url"):
                text += f"\nrecord: {sub['commit_url']}"
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
    model = allowance.from_answer(data)
    if model is None:  # older gateways: only /v1/me has the token counts
        try:
            model = allowance.from_answer(gw.me())
        except GatewayError:
            model = None
    if model is not None:
        data = {**data, "model": model}
        lines.append(allowance.describe(model))
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
def choose_videos(
    seeds: dict[str, list[int]], cap: int = REMOTE_MAX_VIDEOS
) -> dict[str, list[int]]:
    """The episodes a remote ``--video`` run records: at most ``cap``, spread across the tasks in
    plan order (each task's first seed, then each task's second seed, ...)."""
    chosen: dict[str, list[int]] = {}
    for i in range(max(map(len, seeds.values()), default=0)):
        for task, ss in seeds.items():
            if i < len(ss) and sum(map(len, chosen.values())) < cap:
                chosen.setdefault(task, []).append(ss[i])
    return {t: chosen[t] for t in seeds if t in chosen}


def episode_list(plan: dict[str, list[int]]) -> str:
    return "; ".join(
        f"{t} seed{'s' * (len(ss) > 1)} {', '.join(map(str, ss))}" for t, ss in plan.items()
    )


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


def extract_results(
    archive: Path, out: Path, plan: dict[str, list[int]], video: dict[str, list[int]] | None = None
) -> int:
    """Unpack downloaded episode folders into ``out``, accepting only regular files and folders
    under ``<task>/seed_<seed>/`` of planned episodes, and a ``video.mp4`` (at most
    ``REMOTE_VIDEO_MAX_BYTES``) only in an episode ``video`` asked for. Returns the number of
    files."""
    allowed = {f"{t}/seed_{s}" for t, seeds in plan.items() for s in seeds}
    filmed = {f"{t}/seed_{s}" for t, seeds in (video or {}).items() for s in seeds}
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
                if parts[-1] == VIDEO and len(parts) == 3:
                    if "/".join(parts[:2]) not in filmed or m.size > REMOTE_VIDEO_MAX_BYTES:
                        raise CliError(
                            ExitCode.INFRA, f"unexpected video in the results: {m.name!r}"
                        )
                else:
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


def videos_of(out: Path, plan: dict[str, list[int]]) -> tuple[list[str], list[str]]:
    """(paths of the videos in ``out`` for the planned episodes, "task seed N: why not" for
    those that have a note instead)."""
    paths, notes = [], []
    for task, seeds in plan.items():
        for seed in seeds:
            folder = out / task / f"seed_{seed}"
            if (folder / VIDEO).is_file():
                paths.append(str(folder / VIDEO))
            elif (folder / VIDEO_NOTE).is_file():
                why = (folder / VIDEO_NOTE).read_text(errors="replace").strip().splitlines()
                notes.append(f"{task} seed {seed}: {why[-1][:300] if why else 'no video'}")
    return paths, notes


def train_remote(
    config: Config,
    agent_dir: Path,
    seeds: dict[str, list[int]],
    out: Path,
    video: bool = False,
) -> dict:
    """Run the plan on the shared box, download the episode folders into ``out`` and score them
    exactly as a local run. Rerunning with the same ``out`` resumes: it waits for a run already
    submitted, or submits only the episodes still missing. With ``video``, up to
    ``REMOTE_MAX_VIDEOS`` episodes (``choose_videos``) come back with ``video.mp4``."""
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
            filmed = choose_videos(missing) if video else {}
            if filmed:
                n, total = sum(map(len, filmed.values())), sum(map(len, missing.values()))
                say(f"video for {n} of {total} episodes: {episode_list(filmed)}")
            body = {"version": 2, "seeds": missing, "video": filmed} if filmed else missing
            try:
                run = gw.train(archive.data, body)
            except GatewayError as e:
                if filmed and e.code == "bad_plan":
                    raise CliError(
                        ExitCode.NOT_AVAILABLE,
                        f"the shared box refused the video request ({e.message}); "
                        "run again without --video",
                    ) from None
                raise cli_error(e) from None
            state = {
                "id": run["id"],
                "plan": missing,
                "video": filmed,
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
            extract_results(tmp, out, state["plan"], state.get("video"))
        except GatewayError as e:
            raise cli_error(e) from None
        finally:
            tmp.unlink(missing_ok=True)
        state["downloaded"] = True
        state_path.write_text(json.dumps(state, indent=1))
    report = finish_run(opts, planned)
    if video or state.get("video"):
        report["videos"], report["video_notes"] = videos_of(out, seeds)
    return report


def format_videos(report: dict[str, Any]) -> str:
    """Where a remote run's videos are, and why the missing ones are missing."""
    lines = [f"videos ({len(report['videos'])}):"] + [f"  {p}" for p in report["videos"]]
    lines += [f"  no video for {note}" for note in report["video_notes"]]
    return "\n".join(lines)
