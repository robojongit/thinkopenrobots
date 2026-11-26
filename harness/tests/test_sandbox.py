import subprocess
import sys

from harness import sandbox
from harness.sandbox import ContainerSandbox, ProcessSandbox


def test_a_late_kill_spares_processes_started_after_the_episode(tmp_path):
    # A thread that outlives its episode (e.g. one stuck in an env call) may kill its runner
    # later, while the same process runs something else: that kill must not sweep the new
    # process up as an orphan of the old sandbox.
    box = ProcessSandbox(tmp_path)
    runner = box.start_runner(tmp_path / "nothing.sock", "token")
    runner.wait(timeout=30)
    box.close()
    other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        runner.kill()
        assert other.poll() is None
    finally:
        other.kill()
        other.wait()


def test_container_flags(tmp_path, monkeypatch):
    # What every container gets, independent of Docker being here: the security flags, exactly
    # one socket, read-only mounts otherwise, and the run token passed by name only.
    monkeypatch.setattr(sandbox, "check_docker", lambda runtime, image: None)
    box = ContainerSandbox(tmp_path / "work", runtime="runsc", image="img", cpuset="2-3")
    sock = tmp_path / "r0.sock"
    argv = box.create_argv("inkling-x-r0", [(sock, "/run/sim.sock")], ["-m", "m"], ["TOKEN"])
    text = " ".join(argv)
    for flags in [
        "--rm --pull never",
        f"--label inkling.episode={box.episode} --runtime runsc",
        "--network none --ipc none --read-only --cap-drop ALL",
        "--security-opt no-new-privileges --user 65534:65534",
        "--tmpfs /tmp:rw,nosuid,nodev,noexec,size=512m",
        "--cpus 1.0 --memory 2g --memory-swap 2g --pids-limit 256 --ulimit nproc=56",
        "--ulimit core=0 --ulimit msgqueue=0 --log-driver none --cpuset-cpus 2-3",
        "--env TOKEN --mount",
        "--workdir /tmp --entrypoint python img -I -B -m m",
    ]:
        assert flags in text
    mounts = [argv[i + 1] for i, a in enumerate(argv) if a == "--mount"]
    assert mounts == [f"type=bind,src={sock},dst=/run/sim.sock"]
    assert "-v" not in argv and "--volume" not in argv and "--privileged" not in argv
    agent = box.create_argv("a", [(sock, "/run/sim.sock"), (tmp_path, "/agent")], [], [])
    assert f"type=bind,src={tmp_path},dst=/agent,readonly" in agent
    # runc: the pids cgroup bounds the processes; RLIMIT_NPROC would count all containers' uid
    runc = ContainerSandbox(tmp_path / "w2", runtime="runc", image="img")
    assert "nproc=" not in " ".join(runc.create_argv("b", [], [], []))


def test_runner_container_mounts_only_its_socket(tmp_path, monkeypatch):
    monkeypatch.setattr(sandbox, "check_docker", lambda runtime, image: None)
    created = []
    monkeypatch.setattr(sandbox, "_Container", lambda name, create, *a: created.append(create))
    box = ContainerSandbox(tmp_path / "work", runtime="runc", image="img")
    sock = tmp_path / "r0.sock"
    sock.touch()
    monkeypatch.setattr(box, "_share_socket", lambda path: None)
    box.start_runner(sock, "0" * 32)
    (argv,) = created
    mounts = [argv[i + 1] for i, a in enumerate(argv) if a == "--mount"]
    assert mounts == [f"type=bind,src={sock},dst=/run/sim.sock"]
    assert argv[-4:] == ["-m", "inkling_robo.runner", "--socket", "/run/sim.sock"]
    assert list((tmp_path / "work").iterdir()) == []  # no per-run copies of anything


def test_job_labels_and_cgroup_parent_are_passed_as_given(tmp_path, monkeypatch):
    monkeypatch.setattr(sandbox, "check_docker", lambda runtime, image: None)
    box = ContainerSandbox(
        tmp_path / "work",
        runtime="runsc",
        image="img",
        labels={"inkling.job": "s0012-1", "inkling.kind": "score"},
        cgroup_parent="inkling-s0012_1.slice",
    )
    argv = box.create_argv("inkling-x-r0", [], ["-m", "m"], [])
    labels = [argv[i + 1] for i, a in enumerate(argv) if a == "--label"]
    assert labels == [f"inkling.episode={box.episode}", "inkling.job=s0012-1", "inkling.kind=score"]
    (parent,) = [argv[i + 1] for i, a in enumerate(argv) if a == "--cgroup-parent"]
    assert parent == "inkling-s0012_1.slice"
    assert argv.index("--cgroup-parent") < argv.index("img")  # docker options, not the command
    plain = ContainerSandbox(tmp_path / "w2", runtime="runc", image="img")
    plain_argv = plain.create_argv("b", [], [], [])
    assert "--cgroup-parent" not in plain_argv
    assert [a for a in plain_argv if a.startswith("inkling.")] == [
        f"inkling.episode={plain.episode}"
    ]
    cgroupfs = ContainerSandbox(
        tmp_path / "w3", runtime="runc", image="img", cgroup_parent="/inkling/job-1"
    )
    assert "/inkling/job-1" in cgroupfs.create_argv("c", [], [], [])


BAD_LABELS = [
    {"inkling.episode": "x"},  # the sandbox's own label
    {"Inkling.Job": "x"},
    {"inkling job": "x"},
    {"inkling.job": ""},
    {"inkling.job": "a=b"},
    {"inkling.job": "x y"},
    {"inkling.job": "x\n--privileged"},
    {"inkling.job": "-x"},
    {"inkling.job": "x" * 64},
    {"": "x"},
    {"inkling..job": "x"},
    {"inkling.job": 3},
    {f"k{i}": "v" for i in range(9)},
    ["inkling.job=x"],
]
BAD_PARENTS = [
    "inkling-s0012-1.scope",  # Docker's systemd driver wants a slice
    "slice",
    ".slice",
    "-x.slice",
    "a--b.slice",
    "a b.slice",
    "x.slice\n",
    "--privileged",
    "/a/../b",
    "/a/./b",
    "relative/path",
    "/",
    "/a//b",
    "x" * 201 + ".slice",
    "",
    3,
]


def test_bad_labels_and_cgroup_parents_are_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(sandbox, "check_docker", lambda runtime, image: None)
    for bad in BAD_LABELS:
        try:
            ContainerSandbox(tmp_path / "w", image="img", labels=bad)
        except ValueError:
            continue
        raise AssertionError(f"labels {bad!r} accepted")
    for bad in BAD_PARENTS:
        try:
            ContainerSandbox(tmp_path / "w", image="img", cgroup_parent=bad)
        except ValueError:
            continue
        raise AssertionError(f"cgroup_parent {bad!r} accepted")


def test_make_sandbox_threads_the_options_to_containers_only(tmp_path, monkeypatch):
    monkeypatch.setattr(sandbox, "check_docker", lambda runtime, image: None)
    factory = sandbox.make_sandbox(
        "container",
        "runc",
        labels={"inkling.job": "p0001-1"},
        cgroup_parent="inkling-p0001_1.slice",
    )
    box = factory(tmp_path / "work")
    assert box.labels == {"inkling.job": "p0001-1"}
    assert box.cgroup_parent == "inkling-p0001_1.slice" and box.runtime == "runc"
    assert sandbox.make_sandbox("process") is ProcessSandbox
    for kwargs in ({"labels": {"inkling.job": "x"}}, {"cgroup_parent": "x.slice"}):
        try:
            sandbox.make_sandbox("process", **kwargs)
        except ValueError:
            continue
        raise AssertionError(f"process sandbox took {kwargs}")
    try:
        sandbox.make_sandbox("container", labels={"inkling.episode": "x"})
    except ValueError:
        pass
    else:
        raise AssertionError("bad label accepted by make_sandbox")
