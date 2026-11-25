import json

import pytest

from bench.cli import main
from bench.exitcodes import CliError, ExitCode
from bench.runs import list_episodes
from bench.tests.test_train import options
from bench.train import run_train
from bench.view import view_run
from harness.episode import run_episode
from harness.llm_relay import Budget
from harness.tests.conftest import CHAT_AND_RUN, LIFT_REPLY
from harness.tests.fake_env import FakeEnv
from harness.tests.fake_model import FakeModel

BAD_REPLY = "I will try.\n```python\nraise ValueError('boom from model code')\n```\n" + "x" * 3000


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    """Real episodes: cube_lifting wins, cube_stack runs code that raises and fails."""
    tmp = tmp_path_factory.mktemp("view")
    agent = tmp / "agent"
    agent.mkdir()
    (agent / "agent.py").write_text(CHAT_AND_RUN)

    def reply(body):
        text = LIFT_REPLY if "[fake cube_lifting]" in json.dumps(body) else BAD_REPLY
        return 200, {
            "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }

    with FakeModel([reply]) as model:
        opts = options(tmp, agent, model.url, seeds={"cube_lifting": [1], "cube_stack": [2, 3]})
        run_train(opts, FakeEnv)
    return opts.out


def test_failures_only_show_failed_episodes_with_the_error(run):
    view = view_run(run, failures=True)
    assert view["matched"] == 2 and {t["seed"] for t in view["traces"]} == {2, 3}
    trace = view["traces"][0]
    assert trace["outcome"] == "failed" and trace["task"] == "cube_stack"
    kinds = [e["type"] for e in trace["events"]]
    assert kinds == ["reply", "run"]
    run_event = trace["events"][1]
    assert run_event["ok"] is False and "ValueError: boom from model code" in run_event["stderr"]
    assert "raise ValueError" in run_event["code"]


def test_long_replies_are_truncated_unless_full(run):
    short = view_run(run, task="cube_stack", seed=2)["traces"][0]["events"][0]["content"]
    assert "chars omitted" in short and len(short) < 1200
    full = view_run(run, task="cube_stack", seed=2, full=True)["traces"][0]["events"][0]["content"]
    assert full == BAD_REPLY


def test_filters_and_limit(run):
    assert view_run(run)["matched"] == 3
    assert [t["seed"] for t in view_run(run, task="cube_lifting")["traces"]] == [1]
    assert view_run(run, seed=3)["traces"][0]["task"] == "cube_stack"
    limited = view_run(run, limit=1)
    assert len(limited["traces"]) == 1 and len(limited["not_shown"]) == 2
    assert limited["not_shown"][0]["line"].startswith("cube_stack seed 2: failed")


def test_no_failures_is_an_empty_ok_view(run):
    view = view_run(run, failures=True, task="cube_lifting")
    assert view["matched"] == 0 and view["traces"] == []


def test_unknown_run_or_filter_is_not_found(run, tmp_path):
    for bad in (lambda: view_run(tmp_path / "nope"), lambda: view_run(run, seed=999)):
        with pytest.raises(CliError) as e:
            bad()
        assert e.value.code == ExitCode.NOT_FOUND


def test_infra_and_crash_episodes_are_labelled(run, tmp_path):
    folder = run / "cube_stack" / "seed_3"
    (folder / "infra.json").write_text(json.dumps({"detail": "model server returned HTTP 503"}))
    try:
        by_seed = {t["seed"]: t for t in view_run(run, task="cube_stack")["traces"]}
        assert by_seed[3]["outcome"] == "infra"
        assert by_seed[3]["problem"] == "model server returned HTTP 503"
    finally:
        (folder / "infra.json").unlink()


def test_cli_text_is_plain_and_compact(run, capsys):
    assert main(["view", str(run), "--failures", "--task", "cube_stack", "--seed", "2"]) == 0
    out = capsys.readouterr().out
    assert "=== cube_stack seed 2: FAILED" in out
    assert "[model reply 0]" in out and "[run 0] FAILED" in out
    assert "ValueError: boom from model code" in out
    assert len(out) < 4000


def test_cli_json_is_valid(run, capsys):
    assert main(["view", str(run), "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["matched"] == 3 and data["traces"][0]["task"] == "cube_lifting"
    assert len(list_episodes(run)) == 3
    assert main(["view", str(run) + "-gone", "--json"]) == ExitCode.NOT_FOUND
    assert json.loads(capsys.readouterr().out)["error"]["name"] == "not_found"


def test_replies_pair_with_their_entries_after_refused_calls(make_agent, tmp_path):
    # Refused calls (bad_request, budget) log a transcript event but no llm.jsonl entry, and a
    # failed upstream request logs both: each reply must still show against its own entry.
    agent = make_agent(
        """
        from inkling_robo import InklingError

        def solve(task, robot, llm):
            for options in [{}, {"logprobs": True}, {}, {}, {}]:
                try:
                    llm.chat([{"role": "user", "content": "hi"}], **options)
                except InklingError as e:
                    print(e.code)
        """
    )
    rejected = lambda body: (400, {"error": "bad"})  # noqa: E731
    run = tmp_path / "run"
    out = run / "cube_lifting" / "seed_1"
    with FakeModel(["first", rejected, "third"]) as model:
        budget = Budget(calls=2, wall_s=30)
        run_episode("cube_lifting", 1, agent, model.url, budget, out, make_env=FakeEnv)
    events = view_run(run)["traces"][0]["events"]
    assert [(e["type"], e.get("content")) for e in events] == [
        ("reply", "first"),
        ("model_error", None),
        ("model_error", None),
        ("reply", "third"),
        ("model_error", None),
    ]
    assert "logprobs" in events[1]["error"] and "HTTP 400" in events[2]["error"]
    assert "budget" in events[4]["error"]


def test_video_path_and_note_in_text_and_json(write_run, capsys):
    run = write_run("vids", {"cube_lifting": {1: True, 2: False, 3: False}})
    (run / "cube_lifting" / "seed_1" / "video.mp4").write_bytes(b"\0\0\0\x18ftypisom")
    (run / "cube_lifting" / "seed_2" / "video_note.txt").write_text(
        "no video: rendering could not start (RuntimeError: no EGL)\n"
    )
    view = view_run(run, limit=2)
    one, two = view["traces"]
    assert one["video"] == str(run / "cube_lifting" / "seed_1" / "video.mp4")
    assert one["video_note"] is None
    assert two["video"] is None and "no EGL" in two["video_note"]
    assert view["not_shown"][0]["line"].endswith("ended=returned calls=3 tokens=1003")
    code = main(["view", str(run), "--limit", "2"])
    text = capsys.readouterr().out
    assert code == 0 and f"video: {run / 'cube_lifting' / 'seed_1' / 'video.mp4'}" in text
    assert "video note: no video: rendering could not start (RuntimeError: no EGL)" in text
    code = main(["view", str(run), "--seed", "1", "--json"])
    trace = json.loads(capsys.readouterr().out)["traces"][0]
    assert code == 0 and trace["video"].endswith("seed_1/video.mp4")
    (run / "cube_lifting" / "seed_3" / "video.mp4").write_bytes(b"x")
    assert view_run(run, limit=1)["not_shown"][1]["line"].endswith("seed_3/video.mp4")
