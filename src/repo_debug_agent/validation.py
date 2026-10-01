"""Checks a candidate edit must pass: syntax, static sanity, and the project's tests."""

from __future__ import annotations

import ast
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

# --------------------------------------------------------------------------- static checks

# pyflakes findings that indicate real defects (not style), keyed by message class name.
SERIOUS_PYFLAKES = frozenset(
    {
        "UndefinedName",
        "UndefinedLocal",
        "UndefinedExport",
        "DuplicateArgument",
        "ReturnOutsideFunction",
        "YieldOutsideFunction",
        "ContinueOutsideLoop",
        "BreakOutsideLoop",
        "IsLiteral",
        "MultiValueRepeatedKeyLiteral",
        "TooManyExpressionsInStarredAssignment",
        "TwoStarredExpressions",
        "AssertTuple",
        "IfTuple",
        "InvalidPrintSyntax",
        "PercentFormatMissingArgument",
        "PercentFormatExtraNamedArguments",
        "PercentFormatPositionalCountMismatch",
        "StringDotFormatMissingArgument",
        "StringDotFormatExtraPositionalArguments",
        "ForwardAnnotationSyntaxError",
    }
)


@dataclass(frozen=True)
class StaticFinding:
    line: int
    kind: str
    message: str


# warnings.catch_warnings() mutates global state; reviews run in threads.
_WARNINGS_LOCK = threading.Lock()


def _parse(source: str, filename: str) -> tuple[ast.Module, list[StaticFinding]]:
    """ast.parse, also returning compile-time warnings (e.g. invalid escape sequences)
    as findings instead of letting them print to the terminal."""
    with _WARNINGS_LOCK, warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tree = ast.parse(source, filename)
    found = [
        StaticFinding(w.lineno or 1, "SyntaxWarning", str(w.message))
        for w in caught
        if issubclass(w.category, (SyntaxWarning, DeprecationWarning))  # 3.12+ / older
    ]
    return tree, found


def python_findings(source: str, filename: str = "<file>") -> list[StaticFinding]:
    """Syntax errors, compile-time warnings and serious pyflakes findings for Python source."""
    try:
        tree, found = _parse(source, filename)
    except SyntaxError as e:
        return [StaticFinding(e.lineno or 1, "SyntaxError", f"syntax error: {e.msg}")]
    try:
        from pyflakes.checker import Checker
    except ImportError:
        return found
    checker = Checker(tree, filename=filename)
    return found + [
        StaticFinding(m.lineno, type(m).__name__, m.message % m.message_args)
        for m in checker.messages
        if type(m).__name__ in SERIOUS_PYFLAKES
    ]


@dataclass(frozen=True)
class CheckResult:
    ok: bool | None  # None: no checker could judge this file
    message: str = ""


# External syntax checkers. "{file}" is a temp copy that keeps the original file name.
_CHECKERS: dict[str, list[str]] = {
    "javascript": ["node", "--check", "{file}"],
    "typescript": ["node", "--check", "{file}"],  # node >= 23 strips erasable TS syntax
    "go": ["gofmt", "-e", "{file}"],
    "java": ["javac", "-proc:none", "-d", "{tmp}", "{file}"],
    "c": ["gcc", "-fsyntax-only", "-I{srcdir}", "{file}"],
    "cpp": ["g++", "-fsyntax-only", "-I{srcdir}", "{file}"],
    "ruby": ["ruby", "-c", "{file}"],
    "php": ["php", "-l", "{file}"],
}


def _run_checker(language: str, source: str, original_path: Path) -> CheckResult:
    command = _CHECKERS.get(language)
    if command is None or shutil.which(command[0]) is None:
        return CheckResult(None, f"no syntax checker available for {language}")
    with tempfile.TemporaryDirectory(prefix="rda-check-") as tmp:
        file = Path(tmp) / original_path.name
        file.write_text(source, encoding="utf-8")
        args = [a.format(file=file, tmp=tmp, srcdir=original_path.parent) for a in command]
        try:
            proc = subprocess.run(args, capture_output=True, text=True, timeout=60)
        except (OSError, subprocess.TimeoutExpired) as e:
            return CheckResult(None, f"{command[0]} failed to run: {e}")
    output = (proc.stderr or proc.stdout).replace(str(file), original_path.name).strip()
    return CheckResult(proc.returncode == 0, output[-2000:])


def check_edit(language: str, path: Path, before: str, after: str) -> CheckResult:
    """Is `after` at least as valid as `before`?

    Only regressions count: a checker that already rejects the original file
    (missing headers, JSX, unresolved imports, ...) can't judge the edit, so the
    result is "unknown" rather than a false rejection.
    """
    if language == "python":
        return _check_python(before, after, path.name)
    result_after = _run_checker(language, after, path)
    if result_after.ok is not False:
        return result_after
    if _run_checker(language, before, path).ok is not True:
        return CheckResult(None, "syntax checker rejects the original file too; cannot judge")
    return result_after


def _check_python(before: str, after: str, filename: str) -> CheckResult:
    try:
        _parse(after, filename)
    except SyntaxError as e:
        line = (e.text or "").rstrip()
        return CheckResult(False, f"SyntaxError: {e.msg} (line {e.lineno}): {line}")
    old = {(f.kind, f.message) for f in python_findings(before, filename)}
    new = [f for f in python_findings(after, filename) if (f.kind, f.message) not in old]
    if new:
        details = "\n".join(f"line {f.line}: {f.message}" for f in new)
        return CheckResult(False, f"the change introduces new problems:\n{details}")
    return CheckResult(True)


# --------------------------------------------------------------------------- tests

_FAILURE_PATTERNS = [
    re.compile(r"(\d+) (?:failed|errors?)\b"),  # pytest "2 failed, 1 error"; jest "Tests: 2 failed"
    re.compile(r"FAILED \((?:failures=(\d+))?(?:, )?(?:errors=(\d+))?"),  # unittest
    re.compile(r"(\d+) failing\b"),  # mocha
    re.compile(r"test result: FAILED\. \d+ passed; (\d+) failed"),  # cargo
]
_GO_FAIL = re.compile(r"^--- FAIL:", re.MULTILINE)


def count_failures(output: str) -> int | None:
    """Best-effort number of failing tests from common runners' summaries."""
    tail = output[-5000:]
    for pattern in _FAILURE_PATTERNS:
        matches = list(pattern.finditer(tail))
        if matches:
            # Only the last summary line counts (earlier matches may be log noise).
            last_line = tail[: matches[-1].end()].rsplit("\n", 1)[-1]
            total = 0
            for m in pattern.finditer(last_line):
                total += sum(int(g) for g in m.groups() if g)
            return total
    go = len(_GO_FAIL.findall(output))
    return go or None


@dataclass
class TestRun:
    __test__ = False  # not a pytest test class

    command: str
    exit_code: int | None  # None: timed out
    failures: int | None
    output: str
    seconds: float

    @property
    def passed(self) -> bool:
        return self.exit_code == 0

    @property
    def no_tests(self) -> bool:
        return self.exit_code == 5 and "no tests ran" in self.output  # pytest's "nothing collected"

    def summary(self) -> str:
        if self.exit_code is None:
            return f"timed out after {self.seconds:.0f}s"
        if self.passed:
            return f"passed ({self.seconds:.1f}s)"
        count = f"{self.failures} failing" if self.failures is not None else "failing"
        return f"{count} (exit {self.exit_code}, {self.seconds:.1f}s)"

    def excerpt(self, limit: int = 3000) -> str:
        out = self.output.strip()
        return out if len(out) <= limit else "...\n" + out[-limit:]


def run_tests(command: str, cwd: Path, timeout: float) -> TestRun:
    """Run the user's test command in `cwd`, preferring code in `cwd` over installed copies."""
    env = dict(os.environ)
    paths = [str(cwd)] + ([str(cwd / "src")] if (cwd / "src").is_dir() else [])
    env["PYTHONPATH"] = os.pathsep.join(paths + [p for p in [env.get("PYTHONPATH")] if p])
    env["PYTHONDONTWRITEBYTECODE"] = "1"

    started = time.monotonic()
    proc = subprocess.Popen(
        command,
        shell=True,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        start_new_session=sys.platform != "win32",
    )
    try:
        output, _ = proc.communicate(timeout=timeout)
        code: int | None = proc.returncode
    except subprocess.TimeoutExpired:
        _kill_tree(proc)
        output, _ = proc.communicate()
        code = None
    seconds = time.monotonic() - started
    output = normalize_output(output or "", cwd)
    failures = 0 if code == 0 else count_failures(output)
    return TestRun(command, code, failures, output, seconds)


_DURATION = re.compile(r" in \d+(?:\.\d+)?s\b")
_PYTEST_TMP = re.compile(r"/tmp/pytest-of-[^/\s]+/pytest-\d+/")
_ADDRESS = re.compile(r"\b0x[0-9a-fA-F]{6,}\b")


def normalize_output(text: str, cwd: Path) -> str:
    """Drop what changes from run to run (sandbox paths, timings, temp dirs, object
    addresses). It's noise to the model, and it would make identical failures look new
    to the review cache."""
    root = str(cwd)
    text = text.replace(root + os.sep, "").replace(root, ".")
    text = _DURATION.sub("", text)
    text = _PYTEST_TMP.sub("<tmp>/", text)
    return _ADDRESS.sub("0x…", text)


def _kill_tree(proc: subprocess.Popen) -> None:
    if sys.platform != "win32":
        try:
            os.killpg(proc.pid, signal.SIGKILL)
            return
        except ProcessLookupError:
            return
    proc.kill()


@dataclass(frozen=True)
class TestVerdict:
    __test__ = False  # not a pytest test class

    acceptable: bool
    improved: bool
    reason: str


def judge(baseline: TestRun, after: TestRun) -> TestVerdict:
    """Accept an edit if it doesn't make the test suite worse."""
    if after.exit_code is None:
        return TestVerdict(False, False, "the test suite timed out after the change")
    if after.passed:
        return TestVerdict(True, not baseline.passed, "tests pass")
    if baseline.passed:
        return TestVerdict(False, False, "tests passed before the change and fail after it")
    if after.failures is not None and baseline.failures is not None:
        if after.failures > baseline.failures:
            return TestVerdict(
                False, False, f"failing tests went from {baseline.failures} to {after.failures}"
            )
        improved = after.failures < baseline.failures
        return TestVerdict(
            True, improved, f"failing tests: {baseline.failures} -> {after.failures}"
        )
    return TestVerdict(True, False, "suite was already failing; could not compare failure counts")
