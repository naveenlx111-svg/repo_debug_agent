"""Find the source files worth looking at in a repository."""

from __future__ import annotations

import logging
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from repo_debug_agent.languages import Language, detect_language

log = logging.getLogger(__name__)

# Never descend into these, even if git tracks them.
SKIP_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        ".nox",
        "node_modules",
        "venv",
        ".venv",
        "env",
        "site-packages",
        "vendor",
        "dist",
        "build",
        "target",
        "out",
        "coverage",
        ".next",
        ".nuxt",
        ".gradle",
        ".idea",
        ".vscode",
        ".agent_index",
        ".debug_agent",
        "debug_reports",
    }
)

_TEST_DIRS = frozenset({"test", "tests", "__tests__", "spec", "specs", "testing"})
_TEST_FILE = re.compile(
    r"""^(
        test_.*\.py | .*_test\.(py|go|rb|rs) | conftest\.py
      | .*\.(test|spec)\.[cm]?[jt]sx? | .*Tests?\.(java|kt|cs) | .*_spec\.rb
    )$""",
    re.VERBOSE,
)


@dataclass(frozen=True)
class SourceFile:
    path: Path  # absolute
    rel: str  # POSIX path relative to the repo root
    language: Language
    is_test: bool


def is_test_path(rel: str) -> bool:
    *dirs, name = rel.split("/")
    return any(d in _TEST_DIRS for d in dirs) or bool(_TEST_FILE.match(name))


def discover(repo: Path, max_bytes: int = 256_000) -> list[SourceFile]:
    """All readable UTF-8 source files in `repo`, sorted by path.

    Uses `git ls-files` when the repo is under git (so .gitignore is honoured),
    otherwise walks the tree.
    """
    repo = repo.resolve()
    rels = _git_files(repo)
    if rels is None:
        rels = _walk(repo)

    found = []
    for rel in sorted(set(rels)):
        if any(part in SKIP_DIRS for part in rel.split("/")[:-1]):
            continue
        language = detect_language(rel)
        if language is None or ".min." in rel:
            continue
        path = repo / rel
        if not _is_text_file(path, max_bytes):
            continue
        found.append(SourceFile(path, rel, language, is_test_path(rel)))
    return found


def _git_files(repo: Path) -> list[str] | None:
    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(repo),
                "ls-files",
                "-z",
                "--cached",
                "--others",
                "--exclude-standard",
            ],
            capture_output=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return [p for p in result.stdout.decode("utf-8", "surrogateescape").split("\0") if p]


def _walk(repo: Path) -> list[str]:
    rels = []
    for root, dirs, files in os.walk(repo):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        base = Path(root).relative_to(repo)
        rels.extend((base / f).as_posix() for f in files)
    return rels


def _is_text_file(path: Path, max_bytes: int) -> bool:
    try:
        if not path.is_file() or path.stat().st_size > max_bytes:
            return False
        data = path.read_bytes()
    except OSError as e:
        log.warning("cannot read %s: %s", path, e)
        return False
    if b"\0" in data[:8192]:
        return False
    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        log.info("skipping non-UTF-8 file %s", path)
        return False
    return True
