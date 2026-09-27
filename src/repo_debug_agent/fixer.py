"""The fix loop for one issue: propose -> splice -> check -> test -> accept, or retry with feedback.

The model returns a complete replacement for one chunk (a function, method,
class or block of module code), which is spliced in by line range. That is far
more reliable than asking a model to reproduce the original code exactly for a
search-and-replace. When a candidate fails a check, the next attempt sees the
failure (syntax error, new undefined name, test output) instead of blindly
re-asking the same question.
"""

from __future__ import annotations

import ast
import logging
import re
import textwrap
from dataclasses import dataclass, replace

from repo_debug_agent import prompts
from repo_debug_agent.analyzer import locate, relevant_test_output
from repo_debug_agent.chunker import chunk_source
from repo_debug_agent.crawler import SourceFile
from repo_debug_agent.llm import (
    ChatModel,
    LLMError,
    LLMUnavailableError,
    Message,
    pick_code_block,
    strip_reasoning,
)
from repo_debug_agent.models import Attempt, Chunk, FixResult, FixStatus, Issue
from repo_debug_agent.retrieval import ContextRetriever
from repo_debug_agent.retrieval.symbols import SymbolIndex
from repo_debug_agent.validation import TestRun, TestVerdict, check_edit, judge, run_tests
from repo_debug_agent.workspace import Workspace, unified_diff

log = logging.getLogger(__name__)

FIX_MAX_TOKENS = 4096
MAX_REWRITE_LINES = (
    300  # beyond this, asking a model to reproduce the whole unit is too error-prone
)
WHOLE_FILE_LINES = 400  # smaller files go into the prompt whole
EXCERPT_RADIUS = 80


class TestHarness:
    """Runs the user's test command in the sandbox and ratchets the baseline as fixes land."""

    __test__ = False  # not a pytest test class

    def __init__(self, command: str, workspace: Workspace, timeout: float):
        self.command = command
        self.workspace = workspace
        self.timeout = timeout
        self.baseline: TestRun | None = None

    def run(self) -> TestRun:
        return run_tests(self.command, self.workspace.materialize(), self.timeout)

    def start(self) -> TestRun:
        self.baseline = self.run()
        return self.baseline

    def evaluate(self) -> tuple[TestVerdict, TestRun]:
        assert self.baseline is not None, "start() first"
        after = self.run()
        verdict = judge(self.baseline, after)
        if verdict.acceptable:
            self.baseline = after
        return verdict, after

    def output_for(self, rel: str) -> str | None:
        if self.baseline is None or self.baseline.passed:
            return None
        return relevant_test_output(self.baseline.output, rel)


@dataclass
class FixReply:
    code: str | None = None
    explanation: str = ""
    analysis: str = ""
    dismissed: bool = False
    error: str | None = None


_ANALYSIS = re.compile(r"ANALYSIS\s*:\s*(.+?)(?=^\W*VERDICT\b|^```|\Z)", re.DOTALL | re.MULTILINE)
_VERDICT = re.compile(r"VERDICT\W*\s*(NOT[_ ]A[_ ]BUG|BUG)", re.IGNORECASE)
_LEGACY_DISMISS = re.compile(r"\s*\**NOT_A_BUG\**\s*:?\s*(.*)", re.IGNORECASE | re.DOTALL)
# An opened fence the model never closed: small models often go straight to EXPLANATION.
_UNCLOSED_FENCE = re.compile(
    r"^```[ \t]*[\w+#.-]*[^\n]*\n(.*?)(?=^\W*EXPLANATION\s*:|\Z)", re.DOTALL | re.MULTILINE
)


def parse_fix_reply(text: str) -> FixReply:
    """Parse ANALYSIS / VERDICT / code block / EXPLANATION (each part optional-ish)."""
    text = strip_reasoning(text)
    analysis = m.group(1).strip() if (m := _ANALYSIS.search(text)) else ""
    explanation = m.group(1).strip() if (m := re.search(r"EXPLANATION\s*:\s*(.+)", text)) else ""

    verdict = _VERDICT.search(text)
    if verdict and verdict.group(1).upper().startswith("NOT"):
        return FixReply(
            dismissed=True,
            analysis=analysis,
            explanation=analysis or "the fixer judged the report a false positive",
        )
    if not verdict and (m := _LEGACY_DISMISS.match(text)):
        reason = m.group(1).strip().split("\n")[0]
        return FixReply(dismissed=True, explanation=reason or "the report is not a real bug")

    code = pick_code_block(text)
    if code is None and (m := _UNCLOSED_FENCE.search(text)) and m.group(1).strip():
        code = m.group(1)
    if code is None:
        return FixReply(analysis=analysis, error="there was no fenced code block in your answer")
    return FixReply(code=code, explanation=explanation or analysis, analysis=analysis)


def _indent_of(code: str) -> str:
    for line in code.split("\n"):
        if line.strip():
            return line[: len(line) - len(line.lstrip())]
    return ""


def _python_definition(code: str, name: str) -> tuple[str | None, str | None]:
    """Make sure `code` is the definition of `name`; extract it if it's nested in a class."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code, None  # the whole-file check reports this with proper context
    defs = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    if any(isinstance(n, defs) and n.name == name for n in tree.body):
        return code, None
    for node in ast.walk(tree):  # e.g. the model returned the whole enclosing class
        if isinstance(node, defs) and node.name == name:
            start = min([node.lineno, *(d.lineno for d in node.decorator_list)])
            return textwrap.dedent("\n".join(code.split("\n")[start - 1 : node.end_lineno])), None
    return None, f"the code block does not define `{name}`"


def fit_replacement(code: str, chunk: Chunk) -> tuple[str | None, str | None]:
    """Normalize a model's code block to drop into `chunk`'s line range. Returns (code, error)."""
    code = textwrap.dedent(code.strip("\n")).rstrip()
    if not code.strip():
        return None, "the code block is empty"
    if chunk.is_definition:
        if chunk.language == "python":
            code, error = _python_definition(code, chunk.short_name)
            if error:
                return None, error
        elif not re.search(rf"\b{re.escape(chunk.short_name)}\b", code):
            return None, f"the code block does not contain the definition of `{chunk.short_name}`"
    indent = _indent_of(chunk.code)
    return "\n".join(indent + line if line.strip() else "" for line in code.split("\n")), None


def splice(source: str, chunk: Chunk, replacement: str) -> str:
    lines = source.split("\n")
    return "\n".join(
        lines[: chunk.start_line - 1] + replacement.split("\n") + lines[chunk.end_line :]
    )


def whole_class(chunk: Chunk, source: str) -> Chunk:
    """Class chunks cover only the header; a fix needs the whole class."""
    if chunk.kind != "class" or chunk.body_end <= chunk.end_line:
        return chunk
    lines = source.split("\n")
    code = "\n".join(lines[chunk.start_line - 1 : chunk.body_end])
    return replace(chunk, end_line=chunk.body_end, code=code)


def file_excerpt(source: str, chunk: Chunk) -> tuple[str, bool]:
    """The file around `chunk` with the chunk itself replaced by a marker. (text, is_whole_file)"""
    lines = source.split("\n")
    comment = "#" if chunk.language in ("python", "ruby") else "//"
    marker = (
        f"{_indent_of(chunk.code)}{comment} >>> {prompts.describe_target(chunk)} goes here "
        "(the code to fix, shown above) <<<"
    )
    before, after = lines[: chunk.start_line - 1], lines[chunk.end_line :]
    if len(lines) <= WHOLE_FILE_LINES:
        return "\n".join(before + [marker] + after).strip("\n"), True

    lo = max(0, len(before) - EXCERPT_RADIUS)
    head = before[:20] + [f"{comment} ... (lines 21-{lo} omitted)"] if lo > 20 else before[:lo]
    tail = after[:EXCERPT_RADIUS]
    if len(after) > EXCERPT_RADIUS:
        tail.append(f"{comment} ... ({len(after) - EXCERPT_RADIUS} more lines)")
    return "\n".join(head + before[lo:] + [marker] + tail), False


def _with_feedback(base: list[Message], reply: str, feedback: str) -> list[Message]:
    """The original request plus only the latest failed exchange (keeps small contexts small)."""
    return base + [
        {"role": "assistant", "content": reply},
        {"role": "user", "content": feedback},
    ]


class Fixer:
    def __init__(
        self,
        llm: ChatModel,
        model: str,
        workspace: Workspace,
        symbols: SymbolIndex,
        retriever: ContextRetriever,
        tests: TestHarness | None,
        max_attempts: int = 3,
        base_temperature: float = 0.1,
    ):
        self.llm = llm
        self.model = model
        self.workspace = workspace
        self.symbols = symbols
        self.retriever = retriever
        self.tests = tests
        self.max_attempts = max_attempts
        self.base_temperature = base_temperature

    def fix(self, issue: Issue, sf: SourceFile) -> FixResult:
        before = self.workspace.read(issue.file)
        chunks = chunk_source(issue.file, before, sf.language)
        # Earlier fixes may have shifted lines; the chunk key survives that.
        chunk = next((c for c in chunks if c.key == issue.chunk_key), None)
        chunk = chunk or locate(chunks, issue.line, issue.symbol)
        if chunk is None:
            return FixResult(
                issue, FixStatus.SKIPPED, "could not locate the code this issue refers to"
            )
        chunk = whole_class(chunk, before)
        if chunk.span > MAX_REWRITE_LINES:
            return FixResult(
                issue,
                FixStatus.SKIPPED,
                f"`{chunk.name}` is {chunk.span} lines; too large to rewrite reliably",
            )

        excerpt, whole_file = file_excerpt(before, chunk)
        related = self.retriever.render(
            chunk,
            query=f"{issue.description}\n{chunk.name}\n{chunk.code[:400]}",
            exclude_files={issue.file} if whole_file else set(),
        )
        test_output = self.tests.output_for(issue.file) if self.tests else None
        base: list[Message] = [
            {"role": "system", "content": prompts.FIX_SYSTEM.format(language=sf.language.name)},
            {
                "role": "user",
                "content": prompts.fix_prompt(issue, chunk, excerpt, related, test_output),
            },
        ]
        messages = list(base)
        attempts: list[Attempt] = []

        for n in range(1, self.max_attempts + 1):
            # Nudge sampling on retries so a retry can actually differ.
            temperature = min(self.base_temperature + 0.2 * (n - 1), 0.7)
            try:
                reply = self.llm.complete(
                    messages, model=self.model, max_tokens=FIX_MAX_TOKENS, temperature=temperature
                )
            except LLMUnavailableError:
                raise
            except LLMError as e:
                attempts.append(Attempt(n, "llm_error", str(e)))
                continue

            parsed = parse_fix_reply(reply)
            if parsed.dismissed:
                attempts.append(Attempt(n, "dismissed", parsed.explanation))
                return FixResult(issue, FixStatus.DISMISSED, parsed.explanation, attempts=attempts)
            if parsed.error:
                attempts.append(Attempt(n, "invalid_reply", parsed.error))
                messages = _with_feedback(
                    base, reply, prompts.feedback_invalid(parsed.error, chunk)
                )
                continue

            replacement, error = fit_replacement(parsed.code or "", chunk)
            after = splice(before, chunk, replacement) if replacement is not None else before
            if error or after == before:
                error = error or "your code is identical to the original, so it changes nothing"
                attempts.append(Attempt(n, "invalid_reply", error))
                messages = _with_feedback(base, reply, prompts.feedback_invalid(error, chunk))
                continue

            check = check_edit(sf.language.name, sf.path, before, after)
            if check.ok is False:
                attempts.append(Attempt(n, "rejected_syntax", check.message))
                messages = _with_feedback(base, reply, prompts.feedback_rejected(check.message))
                continue

            self.workspace.write(issue.file, after)
            verified_by = "syntax" if check.ok else "none"
            detail = "passed syntax/static checks" if check.ok else check.message
            if self.tests is not None:
                verdict, run = self.tests.evaluate()
                if not verdict.acceptable:
                    self.workspace.write(issue.file, before)
                    attempts.append(Attempt(n, "rejected_tests", verdict.reason))
                    feedback = prompts.feedback_tests(verdict.reason, run.excerpt())
                    messages = _with_feedback(base, reply, feedback)
                    continue
                verified_by = "tests" if verdict.improved else "no-regressions"
                detail = verdict.reason

            attempts.append(Attempt(n, "applied", detail))
            self.symbols.replace_file(issue.file, chunk_source(issue.file, after, sf.language))
            return FixResult(
                issue,
                FixStatus.FIXED,
                parsed.explanation or "(no explanation given)",
                diff="".join(unified_diff(before, after, issue.file)),
                verified_by=verified_by,
                attempts=attempts,
            )

        last = attempts[-1].detail if attempts else "no attempts made"
        return FixResult(
            issue,
            FixStatus.FAILED,
            f"gave up after {len(attempts)} attempts; last: {last}",
            attempts=attempts,
        )
