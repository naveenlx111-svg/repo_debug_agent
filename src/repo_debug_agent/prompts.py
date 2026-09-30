"""Every prompt the agent sends, in one place.

Written to work with small local models (7-9B) as well as large hosted ones:
explicit output shapes, a clear list of what *not* to report (the main source
of false positives), and a way for the fixer to reject a bad report.
"""

from __future__ import annotations

from repo_debug_agent.models import Chunk, Issue

# --------------------------------------------------------------------------- review

REVIEW_SYSTEM = (
    "You are a meticulous senior software engineer hunting for bugs. You report only real "
    "defects, each located precisely, and you never report style or hypothetical concerns."
)

REVIEW_INSTRUCTIONS = """\
Review this {language} code for bugs.

Report only defects that make the code behave incorrectly at runtime:
- crashes or unhandled exceptions on reasonable inputs (empty collections, None/null, zero, missing keys, bad indices)
- wrong results: off-by-one errors, wrong operator or condition, inverted logic, wrong variable, bad arithmetic
- resource leaks (files, connections, locks never released), shared mutable default arguments, broken state handling
- security flaws: injection, path traversal, unsafe deserialization, hard-coded secrets
- misuse of library APIs or language semantics

Do NOT report: style, naming, formatting, missing docstrings/comments/type hints, logging, micro-optimizations, \
or "could validate input" suggestions. If the code already handles a case (for example by raising a clear error), \
that is not a bug.

Be skeptical of your own suspicions: most code you review is correct. For each candidate, trace what the \
code actually does on a concrete input before deciding. Only an issue you can demonstrate with a specific input is a bug.

Answer with JSON only, in exactly this shape (fields in this order):
{{"issues": [{{"line": <line number from the listing>, "symbol": "<enclosing function/method name>", \
"suspicion": "<the suspected problem, one sentence>", \
"trace": "<run the code in your head on a concrete input that should trigger it: what do the relevant lines actually do and return?>", \
"confirmed": <true if the trace shows the code misbehaves, false if it turned out fine>, \
"severity": "critical|high|medium|low", "confidence": <0.0-1.0>, "category": "crash|logic|resource|security|concurrency|api", \
"description": "<the confirmed bug in one or two sentences, with the triggering input>"}}]}}
If there are no real bugs, answer {{"issues": []}}.
"""


def review_prompt(
    rel: str,
    language: str,
    lines: list[str],
    start: int,
    end: int,
    static_hints: list[str],
    test_output: str | None,
) -> str:
    width = len(str(end))
    listing = "\n".join(f"{n:>{width}} | {lines[n - 1]}" for n in range(start, end + 1))
    parts = [REVIEW_INSTRUCTIONS.format(language=language)]
    if static_hints:
        parts.append(
            "A static analyzer (reliable) found these problems in this file; include them if real:\n"
            + "\n".join(f"- {h}" for h in static_hints)
        )
    if test_output:
        parts.append(
            "The project's test suite is failing. Output that may involve this file:\n"
            f"```\n{test_output}\n```"
        )
    span = (
        f"lines {start}-{end} of {len(lines)}" if (start, end) != (1, len(lines)) else "full file"
    )
    parts.append(f"File: {rel} ({span})\n```{language}\n{listing}\n```")
    return "\n\n".join(parts)


# --------------------------------------------------------------------------- fix

FIX_SYSTEM = (
    "You are an expert {language} engineer fixing bugs. You make the smallest change that fully "
    "fixes the bug, and you keep names, signatures and behaviour for valid inputs unchanged unless "
    "changing them is the fix."
)


def describe_target(chunk: Chunk) -> str:
    if chunk.name == "<module>":
        return f"the top-level code at lines {chunk.start_line}-{chunk.end_line}"
    return f"`{chunk.name}` (lines {chunk.start_line}-{chunk.end_line})"


def fix_prompt(
    issue: Issue,
    chunk: Chunk,
    file_excerpt: str,
    related: str,
    test_output: str | None,
    repro: tuple[str, str] | None = None,
) -> str:
    fence = chunk.language
    target = describe_target(chunk)
    whole = "the entire definition, including any decorators/annotations and its signature, "
    if chunk.name == "<module>":
        whole = ""
    line = f" (line {issue.line})" if issue.line else ""

    sections = [
        f"An automated reviewer suspects a bug in `{chunk.file}`{line}. Automated reviewers are "
        "often wrong, so verify the report yourself before changing anything; correct code "
        "must be left alone.\n"
        f"- Severity: {issue.severity}\n- Report: {issue.description}",
        f"## Code to fix: {target}\n```{fence}\n{chunk.code}\n```",
        f"## Rest of the file, for reference only\n```{fence}\n{file_excerpt}\n```",
    ]
    if related:
        sections.append(f"## Related code elsewhere in the repository\n{related}")
    if test_output:
        sections.append(
            "## Current test failures\n"
            "This output is from the project's real test suite. The expected values in its "
            f"assertions are correct by definition.\n```\n{test_output}\n```"
        )
    if repro:
        script, output = repro
        sections.append(
            "## Reproduction attempt\n"
            "The reviewer wrote this script and it fails against the current code. First check "
            "that its inputs are ones the real callers can produce and that its assertion matches "
            "how the code is meant to behave. If not, the report is still wrong: answer NOT_A_BUG. "
            "If it is a real bug, your fix must make this script pass.\n"
            f"```{fence}\n{script.rstrip()}\n```\nOutput:\n```\n{output}\n```"
        )
    import_note = (
        " If you need a new import, import inside the function." if fence == "python" else ""
    )
    sections.append(
        "## Your answer\n"
        "Use exactly this structure:\n\n"
        "ANALYSIS: <2-4 sentences. Trace what the code actually does on the input that triggers the "
        "report (and on any failing test above), step by step, then say whether the report is right.>\n"
        "VERDICT: BUG or NOT_A_BUG\n\n"
        "If the verdict is BUG, continue with:\n"
        f"```{fence}\n<the complete corrected replacement for {target}: {whole}not a diff, "
        "not only the changed lines, and nothing else>\n```\n"
        "EXPLANATION: <one sentence: what was wrong and how you fixed it>\n\n"
        "Code outside the block you return cannot be changed, so don't rely on edits elsewhere."
        + import_note
    )
    return "\n\n".join(sections)


def feedback_invalid(reason: str, chunk: Chunk) -> str:
    return (
        f"I could not use your answer: {reason}\n"
        "Reply again in the required structure: ANALYSIS, VERDICT, then (for a BUG) one "
        f"```{chunk.language} code block with the complete corrected {describe_target(chunk)}, "
        "closed with ```, then the EXPLANATION: line."
    )


def feedback_rejected(message: str) -> str:
    return (
        f"Your change was rejected by a check:\n```\n{message}\n```\n"
        "Correct it and reply again in the same format."
    )


def feedback_tests(reason: str, output: str) -> str:
    return (
        f"Your change was applied, but the test suite got worse ({reason}):\n```\n{output}\n```\n"
        "Revise the fix and reply again in the same format."
    )


def feedback_repro(output: str) -> str:
    return (
        "Your change was applied, but the reproduction script still fails:\n"
        f"```\n{output}\n```\nRevise the fix and reply again in the same format."
    )


# --------------------------------------------------------------------------- reproduction

REPRO_SYSTEM = (
    "You are a meticulous Python engineer. You write minimal scripts that demonstrate bugs by "
    "running the real code, and you never claim a bug you can't show."
)


def repro_prompt(
    issue: Issue, chunk: Chunk, file_excerpt: str, imports: str, related: str = ""
) -> str:
    line = f" (line {issue.line})" if issue.line else ""
    usage = f"## How this code is used elsewhere\n{related}\n\n" if related else ""
    return (
        f"An automated reviewer suspects a bug in `{chunk.file}`{line}. Automated reviewers are "
        "often wrong.\n"
        f"- Report: {issue.description}\n\n"
        f"## The code in question: {describe_target(chunk)}\n```python\n{chunk.code}\n```\n\n"
        f"## Rest of the file, for reference\n```python\n{file_excerpt}\n```\n\n"
        f"{usage}"
        "## Task\n"
        "Write a short standalone Python script that demonstrates the bug by running the real code.\n"
        f"- Import the code with: `{imports}` (it runs from the repository root, which is on sys.path).\n"
        "- Use only inputs the code can really receive: values its real callers could pass. "
        "Calling it with arguments that break how the rest of the code uses it (wrong types, "
        "impossible shapes, internal objects built by hand to be invalid) does not show a bug.\n"
        "- Do not use mocks, fakes or monkeypatching: they make anything fail.\n"
        "- `assert` the behaviour the code is clearly meant to have (from its name, docstring, "
        "callers and tests), not what you would prefer.\n"
        "- While the bug exists the script must fail (AssertionError or the reported exception); "
        "once the bug is fixed it must pass.\n"
        "- No test framework, no network, no files outside a temporary directory, under 30 lines.\n\n"
        "If the problem can't be shown this way (a leak, a race, or only with impossible inputs) "
        "or you conclude the code is actually correct, reply with one line: "
        "NOT_REPRODUCIBLE: <reason>\n"
        "Otherwise reply with only the script, in one ```python code block."
    )


def repro_feedback_passed() -> str:
    return (
        "Your script PASSED against the current code, so it does not demonstrate the bug. "
        "If the bug is real, revise the script so it fails because of it. If running the code "
        "shows it is actually correct, reply NOT_REPRODUCIBLE: <reason>."
    )


def repro_feedback_broken(output: str) -> str:
    return (
        "Your script failed for an unrelated reason, before reaching the code in question:\n"
        f"```\n{output}\n```\nFix the script and reply with it again, in one ```python block."
    )


def repro_feedback_mocks() -> str:
    return (
        "Your script uses mocks or monkeypatching. A mock can make any code fail, so it "
        "demonstrates the mock, not a bug. Rewrite it to run the real code with inputs its real "
        "callers could pass, or reply NOT_REPRODUCIBLE: <reason> if that can't show the bug."
    )
