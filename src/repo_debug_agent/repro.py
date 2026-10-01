"""Reproduce a reported bug with a model-written script before fixing it.

A claimed bug has to be demonstrated. The script must fail against the current
code, either inside the target file or on an assertion. A script that passes is
evidence against the report. Once a fix is proposed, the same script has to pass
for the fix to be accepted, which gives a behavioural check even in repos that
have no test suite. Python only, for now.
"""

from __future__ import annotations

import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from repo_debug_agent import prompts
from repo_debug_agent.llm import ChatModel, LLMError, LLMUnavailableError, Message, pick_code_block
from repo_debug_agent.models import Chunk, Issue
from repo_debug_agent.validation import TestRun, run_tests
from repo_debug_agent.workspace import Workspace

REPRO_TIMEOUT = 30.0
REPRO_MAX_TOKENS = 2048
VET_MAX_TOKENS = 512
_VET_VERDICT = re.compile(r"VERDICT\W*\s*(INVALID|VALID)", re.IGNORECASE)
_DECLINE = re.compile(r"^\W*NOT_REPRODUCIBLE\W*\s*(.*)", re.IGNORECASE | re.MULTILINE)


class ReproStatus(str, Enum):
    REPRODUCED = "reproduced"  # fails now, in the target code or on an assertion
    NOT_REPRODUCED = "not_reproduced"  # the script passed: the report looks wrong
    UNAVAILABLE = "unavailable"  # declined (not scriptable) or the script was broken
    # The script failed, but a check found its input or expectation unrealistic. Not evidence
    # either way: the bug isn't dismissed, but a fix for it can't count as verified.
    REJECTED = "rejected"


@dataclass
class Reproduction:
    status: ReproStatus
    script: str = ""
    output: str = ""  # output of the failing run, for the fix prompt
    detail: str = ""
    calls: int = 0


def module_name(rel: str) -> str | None:
    """Import path for a repo-relative Python file ("src/pkg/mod.py" -> "pkg.mod")."""
    parts = rel.removesuffix(".py").split("/")
    if parts and parts[0] == "src":
        parts = parts[1:]
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    if not parts or not all(p.isidentifier() for p in parts):
        return None
    return ".".join(parts)


def import_line(chunk: Chunk) -> str | None:
    module = module_name(chunk.file)
    if module is None:
        return None
    if chunk.name == "<module>":
        return f"import {module}"
    return f"from {module} import {chunk.name.split('.')[0]}"


_MOCKS = re.compile(
    r"unittest\.mock|from\s+unittest\s+import\s+mock|\bimport\s+mock\b|\bfrom\s+mock\s+import"
    r"|\bMagicMock\b|\bcreate_autospec\b|\bmock\.patch\b|\bmonkeypatch\b"
)


def uses_mocks(script: str) -> bool:
    return _MOCKS.search(script) is not None


def find_python(repo: Path) -> str:
    """The interpreter most likely to have the project's dependencies installed."""
    for venv in (".venv", "venv", "env"):
        for rel in ("bin/python", "Scripts/python.exe"):
            candidate = repo / venv / rel
            if candidate.exists():
                return str(candidate)
    return shutil.which("python3") or shutil.which("python") or sys.executable


def classify(run: TestRun, rel: str) -> str:
    """ "pass", "relevant" (failed in the target file or on an assertion), or "broken"."""
    if run.exit_code == 0:
        return "pass"
    if run.exit_code is None:
        return "broken"  # timed out
    in_target = re.search(r'File "(?:[^"]*[/\\])?' + re.escape(rel) + r'", line \d+', run.output)
    if in_target or re.search(r"^AssertionError\b", run.output, re.MULTILINE):
        return "relevant"
    return "broken"


class Reproducer:
    def __init__(self, llm: ChatModel, model: str, workspace: Workspace, python: str | None = None):
        self.llm = llm
        self.model = model
        self.workspace = workspace
        self.python = python or find_python(workspace.repo)

    def run(self, script: str, target: str) -> TestRun:
        sandbox = self.workspace.materialize()
        with tempfile.TemporaryDirectory(prefix="rda-repro-") as tmp:
            path = Path(tmp) / "reproduce_bug.py"
            path.write_text(script, encoding="utf-8")
            args = [self.python, str(path)]
            command = subprocess.list2cmdline(args) if sys.platform == "win32" else shlex.join(args)
            try:
                run = run_tests(command, sandbox, REPRO_TIMEOUT)
            finally:
                self.workspace.refresh([target])
        run.output = run.output.replace(str(path), "reproduce_bug.py")
        return run

    def reproduce(
        self, issue: Issue, chunk: Chunk, excerpt: str, related: str = ""
    ) -> Reproduction:
        """Get a failing script from the model (one retry, with feedback) and check that it
        fails for the right reason. `related` shows how the code is really called."""
        imports = import_line(chunk)
        if chunk.language != "python" or imports is None:
            return Reproduction(ReproStatus.UNAVAILABLE, detail="not an importable Python module")

        prompt = prompts.repro_prompt(issue, chunk, excerpt, imports, related)
        messages: list[Message] = [
            {"role": "system", "content": prompts.REPRO_SYSTEM},
            {"role": "user", "content": prompt},
        ]
        passed_once = False
        calls = 0
        for attempt in (1, 2):
            try:
                reply = self.llm.complete(messages, model=self.model, max_tokens=REPRO_MAX_TOKENS)
                calls += 1
            except LLMUnavailableError:
                raise
            except LLMError as e:
                return Reproduction(ReproStatus.UNAVAILABLE, detail=f"LLM error: {e}", calls=calls)

            script = pick_code_block(reply)
            if script is None:
                declined = _DECLINE.search(reply)
                reason = declined.group(1).strip() if declined else "no script in the reply"
                # Declining after our own script passed means: on reflection, the code is fine.
                status = ReproStatus.NOT_REPRODUCED if passed_once else ReproStatus.UNAVAILABLE
                return Reproduction(status, detail=reason, calls=calls)

            if uses_mocks(script):
                # Mocks can make anything fail; they demonstrate the mock, not the code.
                outcome, feedback = "mocked", prompts.repro_feedback_mocks()
            else:
                run = self.run(script, chunk.file)
                outcome = classify(run, chunk.file)
                if outcome == "relevant":
                    valid, why = self.vet(issue, chunk, script, run.excerpt(1500), related)
                    calls += 1
                    if valid:
                        return Reproduction(
                            ReproStatus.REPRODUCED, script, run.excerpt(2000), why, calls
                        )
                    outcome, rejected_output = "rejected", run.excerpt(2000)
                    feedback = prompts.repro_feedback_rejected(why)
                else:
                    feedback = (
                        prompts.repro_feedback_passed()
                        if outcome == "pass"
                        else prompts.repro_feedback_broken(run.excerpt(1500))
                    )

            if attempt == 2:
                if outcome == "rejected":
                    detail = f"the reproduction script was rejected: {why}"
                    return Reproduction(
                        ReproStatus.REJECTED, script, rejected_output, detail, calls
                    )
                if outcome == "pass":
                    detail = "the reproduction script passed against the current code"
                    return Reproduction(
                        ReproStatus.NOT_REPRODUCED, script, detail=detail, calls=calls
                    )
                detail = {
                    "mocked": "the reproduction script relied on mocks",
                    "broken": "the reproduction script could not run",
                }[outcome]
                return Reproduction(ReproStatus.UNAVAILABLE, script, detail=detail, calls=calls)

            passed_once = outcome == "pass"
            messages = messages[:2] + [
                {"role": "assistant", "content": reply},
                {"role": "user", "content": feedback},
            ]
        raise AssertionError("unreachable")

    def vet(
        self, issue: Issue, chunk: Chunk, script: str, output: str, related: str
    ) -> tuple[bool, str]:
        """A second opinion, from a fresh context: does this failing script show a real bug,
        or one it manufactured (impossible input, wrong expectation)? Returns (valid, why)."""
        messages: list[Message] = [
            {"role": "system", "content": prompts.VET_SYSTEM},
            {"role": "user", "content": prompts.vet_prompt(issue, chunk, script, output, related)},
        ]
        try:
            reply = self.llm.complete(messages, model=self.model, max_tokens=VET_MAX_TOKENS)
        except LLMUnavailableError:
            raise
        except LLMError as e:
            return True, f"fails as reported (the script check itself failed: {e})"
        verdict = _VET_VERDICT.search(reply)
        reasons = " ".join(
            m.group(0).strip() for m in re.finditer(r"^\W*(INPUT|EXPECTATION)\W.*$", reply, re.M)
        )
        if verdict and verdict.group(1).upper() == "INVALID":
            return False, reasons or "the check judged the input or expectation unrealistic"
        return True, "fails as reported" + (f" ({reasons})" if reasons else "")
