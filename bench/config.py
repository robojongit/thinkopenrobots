"""Where `bench` finds the gateway, the key and the model URL.

Precedence for each setting: command-line flag, environment variable, user config file, default.
The user config file is JSON at ``$INKLING_CONFIG`` or ``<config dir>/inkling-robo/config.json``
(config dir: ``%APPDATA%`` on Windows, else ``$XDG_CONFIG_HOME`` or ``~/.config``).

    {"gateway_url": "https://...", "key_path": "...", "model_url": "...", "model": "..."}

Environment variables: INKLING_GATEWAY_URL (or INKLING_GATEWAY_HOST for ``https://<host>``),
INKLING_KEY_PATH, INKLING_MODEL_URL, INKLING_MODEL. The gateway is HTTPS; plain HTTP is accepted
only for a server on this machine (127.0.0.1, ::1, localhost).

The key file holds the gateway key that `bench login` writes (permissions 0600). It is only ever
sent to the gateway's own origin (scheme, host and port), never to a model URL the person
passes themselves.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from harness.llm_relay import DEFAULT_MODEL

DEFAULT_GATEWAY_URL: str | None = "https://api.thinkopenrobots.com"
"""The competition's gateway. Set at launch, once the domain exists."""
LOOPBACK = ("127.0.0.1", "::1", "localhost")


def config_dir(env: Mapping[str, str] = os.environ) -> Path:
    if sys.platform == "win32" and env.get("APPDATA"):
        base = Path(env["APPDATA"])
    else:
        base = Path(env.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    return base / "inkling-robo"


def origin(url: str) -> tuple[str, str, int] | None:
    """(scheme, host, port) of an http(s) URL, or None."""
    try:
        u = urlparse(url)
        port = u.port or (443 if u.scheme == "https" else 80)
    except ValueError:
        return None
    if u.scheme not in ("http", "https") or not u.hostname:
        return None
    return u.scheme, u.hostname.lower(), port


def check_gateway_url(url: str) -> str:
    """The gateway base URL without a trailing slash; ValueError if it is not HTTPS (or HTTP on
    this machine) or carries a path, query or credentials."""
    o = origin(url)
    u = urlparse(url)
    if o is None or u.username or u.password or u.query or u.fragment or u.path not in ("", "/"):
        raise ValueError(f"gateway URL must look like https://host[:port], not {url!r}")
    if o[0] == "http" and o[1] not in LOOPBACK:
        raise ValueError(f"gateway URL must be https:// (http only for this machine): {url!r}")
    return url.rstrip("/")


@dataclass(frozen=True)
class Config:
    gateway_url: str | None
    key_path: Path
    model_url: str | None
    model: str
    state_dir: Path

    @property
    def gateway_host(self) -> str | None:
        return urlparse(self.gateway_url).hostname if self.gateway_url else None

    def resolve_model_url(self, override: str | None = None) -> str | None:
        """The URL episodes use: the flag, else the configured URL, else the gateway."""
        if override or self.model_url:
            return override or self.model_url
        return f"{self.gateway_url}/v1" if self.gateway_url else None

    def api_key_for(self, url: str) -> str | None:
        """The gateway key, but only for URLs on the gateway's origin."""
        if not self.gateway_url or origin(url) != origin(self.gateway_url):
            return None
        return self.read_key()

    def read_key(self) -> str | None:
        try:
            return self.key_path.read_text().strip() or None
        except OSError:
            return None

    def save_key(self, key: str) -> None:
        """Write the key readable by this user only (0600), replacing any earlier one."""
        write_private(self.key_path, key + "\n")


def write_private(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with contextlib.suppress(OSError):
        path.parent.chmod(0o700)
    tmp = path.with_name(path.name + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, text.encode())
    finally:
        os.close(fd)
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)


def load_config(env: Mapping[str, str] = os.environ) -> Config:
    path = Path(env.get("INKLING_CONFIG") or config_dir(env) / "config.json")
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        data = {}
    if not isinstance(data, dict):
        data = {}

    def setting(env_name: str, key: str) -> str | None:
        return env.get(env_name) or data.get(key) or None

    gateway = setting("INKLING_GATEWAY_URL", "gateway_url")
    host = setting("INKLING_GATEWAY_HOST", "gateway_host")
    if not gateway and host:
        gateway = f"https://{host}"
    gateway = gateway or DEFAULT_GATEWAY_URL
    return Config(
        gateway_url=check_gateway_url(gateway) if gateway else None,
        key_path=Path(setting("INKLING_KEY_PATH", "key_path") or config_dir(env) / "key"),
        model_url=setting("INKLING_MODEL_URL", "model_url"),
        model=setting("INKLING_MODEL", "model") or DEFAULT_MODEL,
        state_dir=config_dir(env),
    )
