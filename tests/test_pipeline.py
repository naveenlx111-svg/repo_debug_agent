"""End-to-end runs against a real temp repo and test suite, with a scripted model."""

import json

import pytest
from conftest import PYTEST_CMD, ScriptedLLM, fix_reply, review_reply, write_tree

from repo_debug_agent.llm import LLMUnavailableError
from repo_debug_agent.models import FixStatus
from repo_debug_agent.pipeline import run
from repo_debug_agent.ui import UI

CALC = """def average(xs):
    return sum(xs) / len(xs)


def double(x):
    return x * 2
"""
TESTS = """from calc import average, double


def test_average_empty():
    assert average([]) == 0


def test_average():
    assert average([2, 4]) == 3


def test_double():
    assert double(2) == 4
"""
AVERAGE_ISSUE = {
    "line": 2,
    "symbol": "average",
    "severity": "high",
    "confidence": 0.9,
    "description": "ZeroDivisionError when xs is empty",
}
GOOD_FIX = "def average(xs):\n    if not xs:\n        return 0\n    return sum(xs) / len(xs)"


@pytest.fixture
def repo(tmp_path):
    return write_tree(tmp_path / "repo", {"calc.py": CALC, "tests/test_calc.py": TESTS})


def model(*fix_replies, issues=(AVERAGE_ISSUE,)):
    """Reviews report `issues`; fix requests get `fix_replies` in order."""
    replies = iter(fix_replies)

    def respond(messages, json_mode):
        if json_mode:
            return review_reply(*issues)
        return next(replies)

    return ScriptedLLM(respond)


def fix_calls(llm):
    return [call for call in llm.calls if "Report:" in call[1]["content"]]


def test_fix_is_verified_by_tests_and_repo_is_untouched(repo, make_settings):
    llm = model(fix_reply(GOOD_FIX, "Return 0 for an empty list."))
    report = run(make_settings(repo, test_cmd=PYTEST_CMD), llm=llm, ui=UI.silent())

    [result] = report.results
    assert result.status == FixStatus.FIXED
    assert result.verified_by == "tests"
    assert report.baseline_tests.failures == 1 and report.final_tests.passed
    assert (repo / "calc.py").read_text() == CALC  # no --apply: repo unchanged

    patch = open(report.patch_file).read()
    assert "+    if not xs:" in patch and "a/calc.py" in patch
    data = json.loads((open(report.patch_file.replace("fixes.patch", "report.json"))).read())
    assert data["summary"] == {"fixed": 1}

    # the fix prompt carried the failing test output and the test as related context
    prompt = fix_calls(llm)[0][1]["content"]
    assert "ZeroDivisionError" in prompt and "test_average_empty" in prompt


def test_test_files_are_not_reviewed_by_default(repo, make_settings):
    llm = model(issues=())
    run(make_settings(repo, analyze_only=True), llm=llm, ui=UI.silent())
    reviewed = [c[1]["content"] for c in llm.calls]
    assert len(reviewed) == 1 and "File: calc.py" in reviewed[0]


def test_regression_is_rejected_and_model_gets_feedback(repo, make_settings):
    breaks_more = "def average(xs):\n    return 1 / 0"
    llm = model(fix_reply(breaks_more), fix_reply(GOOD_FIX))
    report = run(make_settings(repo, test_cmd=PYTEST_CMD), llm=llm, ui=UI.silent())

    [result] = report.results
    assert [a.outcome for a in result.attempts] == ["rejected_tests", "applied"]
    assert "from 1 to 2" in result.attempts[0].detail
    retry = fix_calls(llm)[1]
    assert retry[2]["role"] == "assistant" and "1 / 0" in retry[2]["content"]
    assert "test suite got worse" in retry[3]["content"]
    assert result.status == FixStatus.FIXED and report.final_tests.passed


def test_syntax_error_is_rejected_with_feedback(repo, make_settings):
    llm = model(fix_reply("def average(xs):\n    return (sum(xs) / len(xs)"), fix_reply(GOOD_FIX))
    report = run(make_settings(repo), llm=llm, ui=UI.silent())
    [result] = report.results
    assert [a.outcome for a in result.attempts] == ["rejected_syntax", "suggested"]
    assert "SyntaxError" in fix_calls(llm)[1][3]["content"]
    assert result.status == FixStatus.SUGGESTED and result.verified_by == "syntax"


def test_invalid_replies_count_as_attempts(repo, make_settings):
    llm = model("I would add a check.", fix_reply(CALC.split("\n\n\n")[0]), fix_reply("x = 1"))
    report = run(make_settings(repo), llm=llm, ui=UI.silent())
    [result] = report.results
    assert result.status == FixStatus.FAILED
    assert [a.outcome for a in result.attempts] == ["invalid_reply"] * 3
    assert "identical to the original" in result.attempts[1].detail
    assert report.patch_file is None


def test_dismissed_issue_changes_nothing(repo, make_settings):
    llm = model("NOT_A_BUG: callers never pass an empty list")
    report = run(make_settings(repo), llm=llm, ui=UI.silent())
    assert report.results[0].status == FixStatus.DISMISSED
    assert report.patch_file is None


def test_apply_writes_the_verified_fix(repo, make_settings):
    llm = model(fix_reply(GOOD_FIX))
    report = run(make_settings(repo, apply=True, test_cmd=PYTEST_CMD), llm=llm, ui=UI.silent())
    assert report.results[0].verified_by == "tests"
    assert report.applied_files == ["calc.py"]
    assert "if not xs:" in (repo / "calc.py").read_text()


def test_fix_without_evidence_is_a_suggestion_and_never_applied(repo, make_settings):
    llm = model(fix_reply(GOOD_FIX))
    report = run(make_settings(repo, apply=True), llm=llm, ui=UI.silent())
    [result] = report.results
    assert result.status == FixStatus.SUGGESTED and "+    if not xs:" in result.diff
    assert report.applied_files == [] and (repo / "calc.py").read_text() == CALC
    assert report.patch_file is None
    assert "+    if not xs:" in open(report.suggestions_file).read()


def test_keep_unverified_restores_applying_syntax_checked_fixes(repo, make_settings):
    llm = model(fix_reply(GOOD_FIX))
    report = run(make_settings(repo, apply=True, keep_unverified=True), llm=llm, ui=UI.silent())
    assert report.results[0].status == FixStatus.FIXED
    assert "if not xs:" in (repo / "calc.py").read_text()


def test_suggestions_do_not_become_the_base_for_later_fixes(repo, make_settings):
    # Two issues in one file, no evidence for either: each suggestion is made against the
    # original code, and nothing lands in the workspace.
    double_issue = {**AVERAGE_ISSUE, "line": 6, "symbol": "double", "description": "odd"}
    llm = model(
        fix_reply(GOOD_FIX),
        fix_reply("def double(x):\n    return x * 2  # reviewed"),
        issues=(AVERAGE_ISSUE, double_issue),
    )
    report = run(make_settings(repo), llm=llm, ui=UI.silent())
    assert [r.status for r in report.results] == [FixStatus.SUGGESTED] * 2
    assert "if not xs" not in fix_calls(llm)[1][1]["content"]


def test_static_finding_fixed_counts_as_evidence(tmp_path, make_settings):
    repo = write_tree(tmp_path / "r", {"m.py": "def area(r):\n    return pi * r * r\n"})
    llm = model(
        fix_reply("def area(r):\n    import math\n    return math.pi * r * r"),
        issues=({"line": 2, "symbol": "area", "confidence": 0.9, "description": "pi undefined"},),
    )
    report = run(make_settings(repo), llm=llm, ui=UI.silent())
    [result] = report.results
    assert "static" in result.issue.source
    assert result.status == FixStatus.FIXED and result.verified_by == "static"


def test_analyze_only_reports_without_fixing(repo, make_settings):
    llm = model()
    report = run(
        make_settings(repo, analyze_only=True, test_cmd=PYTEST_CMD), llm=llm, ui=UI.silent()
    )
    assert [r.status for r in report.results] == [FixStatus.REPORTED]
    assert fix_calls(llm) == [] and report.baseline_tests is None


def test_low_confidence_issues_are_not_fixed(repo, make_settings):
    llm = model(issues=({**AVERAGE_ISSUE, "confidence": 0.3},))
    report = run(make_settings(repo), llm=llm, ui=UI.silent())
    assert report.issues_found == 1 and report.results == []


def test_two_fixes_in_one_file_both_land(repo, make_settings):
    double_issue = {**AVERAGE_ISSUE, "line": 6, "symbol": "double", "description": "should triple"}
    llm = model(
        fix_reply(GOOD_FIX),
        fix_reply("def double(x):\n    return x * 2  # reviewed"),
        issues=(AVERAGE_ISSUE, double_issue),
    )
    report = run(make_settings(repo, apply=True, keep_unverified=True), llm=llm, ui=UI.silent())
    assert [r.status for r in report.results] == [FixStatus.FIXED, FixStatus.FIXED]
    text = (repo / "calc.py").read_text()
    assert "if not xs:" in text and "# reviewed" in text


def test_llm_outage_stops_cleanly(repo, make_settings):
    def respond(messages, json_mode):
        if json_mode:
            return review_reply(AVERAGE_ISSUE, {**AVERAGE_ISSUE, "line": 6, "symbol": "double"})
        raise LLMUnavailableError("connection refused")

    report = run(make_settings(repo), llm=ScriptedLLM(respond), ui=UI.silent())
    assert [r.status for r in report.results] == [FixStatus.SKIPPED, FixStatus.SKIPPED]
    assert any("unavailable" in w for w in report.warnings)
