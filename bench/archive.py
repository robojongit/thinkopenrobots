"""The ``agent/`` archive that `bench submit` and `bench train --remote` upload.

The server accepts only regular files under ``agent/``: no links or special files, at most 300
files and 5 MB, only ``.py``, ``.md``, ``.json``, ``.yaml`` and ``.yml`` files as UTF-8 text, with
``agent/agent.py`` and ``agent/NOTES.md`` present. These checks run here first so a problem is
reported before anything is sent. Python caches and OS clutter (``__pycache__``, ``*.pyc``,
``.DS_Store``, ``Thumbs.db``) are left out silently.

The archive is deterministic: the same files give the same bytes (sorted, fixed owner, mode and
times), so its SHA-256 identifies the agent.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import os
import stat
import tarfile
import unicodedata
from dataclasses import dataclass
from pathlib import Path

ALLOWED_SUFFIXES = (".py", ".md", ".json", ".yaml", ".yml")
MAX_BYTES = 5 * 1024 * 1024
MAX_FILES = 300
MAX_PATH_BYTES = 255
SKIPPED_DIRS = {"__pycache__"}
SKIPPED_FILES = {".DS_Store", "Thumbs.db", "desktop.ini"}
SKIPPED_SUFFIXES = (".pyc", ".pyo")
REQUIRED = {
    "agent/agent.py": "agent/agent.py is missing: it must define solve(task, robot, llm)",
    "agent/NOTES.md": "agent/NOTES.md is missing: name the model and coding agent used and add "
    "short notes",
}


class ArchiveProblem(ValueError):
    def __init__(self, problems: list[str]) -> None:
        super().__init__("; ".join(problems))
        self.problems = problems


@dataclass(frozen=True)
class AgentArchive:
    data: bytes
    files: int
    bytes: int

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.data).hexdigest()


def collect(agent_dir: Path) -> dict[str, bytes]:
    """Files of the agent folder as ``agent/<relative path>`` -> content, checked against the
    server's rules. Raises ArchiveProblem listing everything wrong."""
    if not agent_dir.is_dir():
        raise ArchiveProblem([f"{agent_dir} is not a folder"])
    problems: list[str] = []
    files: dict[str, bytes] = {}
    total = 0
    folded: dict[str, str] = {}
    for root, dirs, names in os.walk(agent_dir, followlinks=False):
        root_path = Path(root)
        for d in list(dirs):
            p = root_path / d
            if d in SKIPPED_DIRS:
                dirs.remove(d)
            elif p.is_symlink():
                dirs.remove(d)
                problems.append(f"{_name(agent_dir, p)} is a symbolic link; links are not allowed")
        for n in names:
            p = root_path / n
            name = _name(agent_dir, p)
            if n in SKIPPED_FILES or n.endswith(SKIPPED_SUFFIXES):
                continue
            mode = os.lstat(p).st_mode
            if stat.S_ISLNK(mode):
                problems.append(f"{name} is a symbolic link; links are not allowed")
                continue
            if not stat.S_ISREG(mode):
                problems.append(f"{name} is a special file; only regular files are allowed")
                continue
            if not n.endswith(ALLOWED_SUFFIXES):
                problems.append(
                    f"{name}: file type not allowed (only {', '.join(ALLOWED_SUFFIXES)})"
                )
                continue
            if "\\" in name or ":" in name or not name.isprintable():
                problems.append(f"{name!r}: unusual characters in the file name")
                continue
            if len(name.encode()) > MAX_PATH_BYTES:
                problems.append(f"{name}: path too long")
                continue
            content = p.read_bytes()
            try:
                content.decode("utf-8")
            except UnicodeDecodeError:
                problems.append(f"{name}: not UTF-8 text")
                continue
            name = unicodedata.normalize("NFC", name)
            if name.casefold() in folded:
                problems.append(f"{name} and {folded[name.casefold()]} differ only in letter case")
            folded[name.casefold()] = name
            files[name] = content
            total += len(content)
    if len(files) > MAX_FILES:
        problems.append(f"agent/ has {len(files)} files; the limit is {MAX_FILES}")
    if total > MAX_BYTES:
        problems.append(f"agent/ holds {total} bytes; the limit is {MAX_BYTES} (5 MB)")
    for path, message in REQUIRED.items():
        if path not in files:
            problems.append(message)
        elif not files[path].strip():
            problems.append(f"{path} is empty")
    if problems:
        raise ArchiveProblem(problems)
    return files


def _name(agent_dir: Path, path: Path) -> str:
    return "agent/" + path.relative_to(agent_dir).as_posix()


def pack(files: dict[str, bytes]) -> bytes:
    """Deterministic tar.gz of ``files`` (the same form the server stores)."""
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0, compresslevel=9) as gz:
        with tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as tar:
            for path in sorted(files):
                info = tarfile.TarInfo(path)
                info.size = len(files[path])
                info.mode = 0o644
                info.mtime = 0
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                tar.addfile(info, io.BytesIO(files[path]))
    return buf.getvalue()


def build(agent_dir: Path) -> AgentArchive:
    files = collect(agent_dir)
    data = pack(files)
    if len(data) > MAX_BYTES:
        raise ArchiveProblem([f"the archive is {len(data)} bytes; the limit is {MAX_BYTES}"])
    return AgentArchive(data, len(files), sum(len(v) for v in files.values()))
