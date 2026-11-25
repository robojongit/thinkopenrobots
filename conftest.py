"""Repo-wide test isolation: no test may reach the real gateway or read the person's key."""

import pytest


@pytest.fixture(autouse=True)
def _no_real_gateway(monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory):
    import bench.config

    home = tmp_path_factory.mktemp("isolated-config")
    for name in (
        "INKLING_GATEWAY_URL",
        "INKLING_GATEWAY_HOST",
        "INKLING_MODEL_URL",
        "INKLING_MODEL",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("INKLING_CONFIG", str(home / "config.json"))
    monkeypatch.setenv("INKLING_KEY_PATH", str(home / "key"))
    monkeypatch.setattr(bench.config, "DEFAULT_GATEWAY_URL", None)
