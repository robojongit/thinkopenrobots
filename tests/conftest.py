import pytest


@pytest.fixture(scope="session")
def ik_server():
    """The real simulator's prerequisites: skip if CaP-X is not installed; reuse a running
    PyRoKi IK server or start one for the session."""
    try:
        from harness.capx_env import import_capx

        import_capx()
        import robosuite  # noqa: F401
    except Exception as e:  # missing submodule or sim group
        pytest.skip(f"simulator not installed: {e}")
    from harness.ik_server import ensure_ik_server, stop_ik_server

    ensure_ik_server()
    yield
    stop_ik_server()
