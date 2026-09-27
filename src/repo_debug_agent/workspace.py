"""Where edits happen: never in the user's repo until they ask for it.

Edits go to an in-memory overlay. When a test command needs a real tree, the
repo is copied to a temporary sandbox (dependency folders like node_modules
and .venv are symlinked, not copied) and edits are mirrored there. The user's
files change only through `apply()`, which refuses to overwrite anything that
changed on disk during the run.
"""

from __future__ import annotations

import difflib
import logging
import os
import shutil
import tempfile
from pathlib import Path

log = logging.getLogger(__name__)

# Heavy dependency folders tests may need: linked into the sandbox, not copied.
LINK_DIRS = frozenset({"node_modules", "venv", ".venv", "env", "vendor"})
# Never copied.
SKIP_COPY = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".tox",
        ".nox",
        ".agent_index",
        ".debug_agent",
        "debug_reports",
    }
)


class ApplyConflict(RuntimeError):
    pass


def _read(path: Path) -> tuple[str, str]:
    """(text with '\\n' line endings, original newline style)."""
    text = path.read_bytes().decode("utf-8")
    newline = "\r\n" if "\r\n" in text else "\n"
    return text.replace("\r\n", "\n"), newline


def _write(path: Path, text: str, newline: str) -> None:
    path.write_bytes(text.replace("\n", newline).encode("utf-8"))


class Workspace:
    def __init__(self, repo: Path):
        self.repo = repo.resolve()
        self.sandbox: Path | None = None
        self._tmp: str | None = None
        self._overlay: dict[str, str] = {}
        self._originals: dict[str, str] = {}
        self._newlines: dict[str, str] = {}

    # ------------------------------------------------------------------ files

    def read(self, rel: str) -> str:
        if rel in self._overlay:
            return self._overlay[rel]
        text, newline = _read(self.repo / rel)
        self._newlines.setdefault(rel, newline)
        return text

    def write(self, rel: str, text: str) -> None:
        if rel not in self._originals:
            self._originals[rel] = self.read(rel)
        self._overlay[rel] = text
        if self.sandbox is not None:
            _write(self.sandbox / rel, text, self._newlines.get(rel, "\n"))

    def changed_files(self) -> list[str]:
        return sorted(rel for rel, text in self._overlay.items() if text != self._originals[rel])

    def original(self, rel: str) -> str:
        return self._originals[rel] if rel in self._originals else self.read(rel)

    # ------------------------------------------------------------------ sandbox

    def materialize(self) -> Path:
        """Copy the repo to a temp dir (once) and return its path."""
        if self.sandbox is not None:
            return self.sandbox
        self._tmp = tempfile.mkdtemp(prefix="repo-debug-agent-")
        sandbox = Path(self._tmp) / self.repo.name
        links: list[tuple[Path, Path]] = []

        def ignore(directory: str, names: list[str]) -> set[str]:
            skipped = set()
            for name in names:
                src = Path(directory) / name
                if name in SKIP_COPY:
                    skipped.add(name)
                elif name in LINK_DIRS and src.is_dir() and not src.is_symlink():
                    skipped.add(name)
                    links.append((src, sandbox / src.relative_to(self.repo)))
            return skipped

        shutil.copytree(self.repo, sandbox, symlinks=True, ignore=ignore)
        for src, dst in links:
            try:
                os.symlink(src, dst, target_is_directory=True)
            except OSError as e:  # e.g. Windows without symlink privilege
                log.warning(
                    "could not link %s into sandbox (%s); tests needing it may fail", src, e
                )
        for rel, text in self._overlay.items():
            _write(sandbox / rel, text, self._newlines.get(rel, "\n"))
        self.sandbox = sandbox
        return sandbox

    def refresh(self, rels: list[str]) -> None:
        """Rewrite sandbox copies from the agent's view of the files (undoing anything
        a script run in the sandbox may have written to them)."""
        if self.sandbox is None:
            return
        for rel in set(rels) | set(self._overlay):
            text = self.read(rel)
            _write(self.sandbox / rel, text, self._newlines.get(rel, "\n"))

    def cleanup(self) -> None:
        if self._tmp:
            shutil.rmtree(self._tmp, ignore_errors=True)
            self._tmp = None
            self.sandbox = None

    # ------------------------------------------------------------------ results

    def diff(self, rels: list[str] | None = None) -> str:
        """Unified diff (git-apply compatible) of changed files against the originals."""
        out: list[str] = []
        for rel in rels if rels is not None else self.changed_files():
            before = self._originals.get(rel)
            after = self._overlay.get(rel)
            if before is None or after is None or before == after:
                continue
            out.extend(unified_diff(before, after, rel))
        return "".join(out)

    def apply(self) -> list[str]:
        """Write changed files back to the repo. Raises ApplyConflict before touching anything
        if any of them was modified on disk since the run started."""
        changed = self.changed_files()
        conflicts = [rel for rel in changed if _read(self.repo / rel)[0] != self._originals[rel]]
        if conflicts:
            raise ApplyConflict(
                "these files changed on disk during the run; not applying: " + ", ".join(conflicts)
            )
        for rel in changed:
            _write(self.repo / rel, self._overlay[rel], self._newlines.get(rel, "\n"))
        return changed


def unified_diff(before: str, after: str, rel: str) -> list[str]:
    lines = []
    for line in difflib.unified_diff(
        before.splitlines(keepends=True),
        after.splitlines(keepends=True),
        fromfile=f"a/{rel}",
        tofile=f"b/{rel}",
    ):
        lines.append(line if line.endswith("\n") else line + "\n\\ No newline at end of file\n")
    return lines
