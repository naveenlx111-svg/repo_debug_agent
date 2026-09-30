"""Reproduce-before-fix: unit tests plus full pipeline runs with a scripted model."""

import sys

import pytest
from conftest import ScriptedLLM, fix_reply, is_repro_request, review_reply, write_tree

from repo_debug_agent.models import Chunk, FixStatus
from repo_debug_agent.pipeline import run
from repo_debug_agent.repro import classify, import_line, module_name
from repo_debug_agent.ui import UI
from repo_debug_agent.validation import TestRun

CALC = """def average(xs):
    return sum(xs) / len(xs)


def double(x):
    return x * 2
"""
AVERAGE = {
    "line": 2,
    "symbol": "average",
    "severity": "high",
    "confidence": 0.9,
    "description": "ZeroDivisionError when xs is empty",
}
DOUBLE = {
    "line": 6,
    "symbol": "double",
    "severity": "high",
    "confidence": 0.9,
    "description": "double returns the wrong value",
}
GOOD_FIX = "def average(xs):\n    if not xs:\n        return 0\n    return sum(xs) / len(xs)"


def script(code: str) -> str:
    return f"```python\n{code}\n```"


@pytest.mark.parametrize(
    ("rel", "expected"),
    [
        ("calc.py", "calc"),
        ("src/pkg/mod.py", "pkg.mod"),
        ("pkg/__init__.py", "pkg"),
        ("scripts/my-tool.py", None),
    ],
)
def test_module_name(rel, expected):
    assert module_name(rel) == expected


def test_import_line():
    def chunk(name):
        return Chunk("src/pkg/shop.py", "python", "method", name, 1, 2, "", "k")

    assert import_line(chunk("Cart.total")) == "from pkg.shop import Cart"
    assert import_line(chunk("helper")) == "from pkg.shop import helper"
    assert import_line(chunk("<module>")) == "import pkg.shop"


def test_classify():
    def run_(code, output):
        return TestRun("t", code, None, output, 0.1)

    in_target = 'Traceback:\n  File "/tmp/x/repo/calc.py", line 2, in average\nZeroDivisionError'
    assert classify(run_(0, ""), "calc.py") == "pass"
    assert classify(run_(1, in_target), "calc.py") == "relevant"
    assert classify(run_(1, "Traceback...\nAssertionError: 3 != 4"), "calc.py") == "relevant"
    assert (
        classify(run_(1, 'File "/tmp/x/repo/mycalc.py", line 1\nValueError'), "calc.py") == "broken"
    )
    assert classify(run_(1, "ModuleNotFoundError: No module named 'calcx'"), "calc.py") == "broken"
    assert classify(run_(None, ""), "calc.py") == "broken"


@pytest.fixture
def repo(tmp_path):
    return write_tree(tmp_path / "repo", {"calc.py": CALC})


def scripted(issues, repros, fixes):
    """Review -> `issues`; repro requests -> `repros` in order; fix requests -> `fixes` in order."""
    repros, fixes = iter(repros), iter(fixes)

    def respond(messages, json_mode):
        if json_mode:
            return review_reply(*issues)
        if is_repro_request(messages):
            return next(repros)
        return next(fixes)

    return ScriptedLLM(respond)


def settings(make_settings, repo, **kw):
    return make_settings(repo, repro=True, python=sys.executable, **kw)


def test_reproduced_bug_is_fixed_and_verified_by_the_script(repo, make_settings):
    llm = scripted(
        [AVERAGE],
        [script("from calc import average\nassert average([]) == 0")],
        [fix_reply(GOOD_FIX)],
    )
    report = run(settings(make_settings, repo), llm=llm, ui=UI.silent())
    [result] = report.results
    assert result.status == FixStatus.FIXED
    assert result.verified_by == "repro" and result.repro_status == "reproduced"
    fix_prompt = next(
        c for c in llm.calls if "Report:" in c[1]["content"] and not is_repro_request(c)
    )[1]["content"]
    assert "## Reproduction (verified)" in fix_prompt and "ZeroDivisionError" in fix_prompt


def test_unreproducible_report_is_dismissed_without_a_fix_attempt(repo, make_settings):
    passes = script("from calc import double\nassert double(2) == 4")
    llm = scripted([DOUBLE], [passes, passes], [])
    report = run(settings(make_settings, repo), llm=llm, ui=UI.silent())
    [result] = report.results
    assert result.status == FixStatus.DISMISSED and result.repro_status == "not_reproduced"
    assert report.patch_file is None
    retry = llm.calls[-1]
    assert "PASSED against the current code" in retry[3]["content"]


def test_declining_after_a_passing_script_counts_as_not_reproduced(repo, make_settings):
    llm = scripted(
        [DOUBLE],
        [
            script("from calc import double\nassert double(2) == 4"),
            "NOT_REPRODUCIBLE: running it shows double is correct",
        ],
        [],
    )
    report = run(settings(make_settings, repo), llm=llm, ui=UI.silent())
    assert report.results[0].status == FixStatus.DISMISSED


def test_fix_that_leaves_the_script_failing_is_rejected(repo, make_settings):
    still_broken = "def average(xs):\n    return sum(xs) / max(len(xs), 0)"
    llm = scripted(
        [AVERAGE],
        [script("from calc import average\nassert average([]) == 0")],
        [fix_reply(still_broken), fix_reply(GOOD_FIX)],
    )
    report = run(settings(make_settings, repo), llm=llm, ui=UI.silent())
    [result] = report.results
    assert [a.outcome for a in result.attempts] == ["rejected_repro", "applied"]
    assert result.status == FixStatus.FIXED and result.verified_by == "repro"


@pytest.mark.parametrize(
    "repros",
    [
        [script("from calc import nope"), script("import calcx")],  # broken twice
        ["NOT_REPRODUCIBLE: this is a resource leak"],  # can't be scripted
    ],
)
def test_without_a_usable_script_fixing_proceeds_as_before(repo, make_settings, repros):
    llm = scripted([AVERAGE], repros, [fix_reply(GOOD_FIX)])
    report = run(settings(make_settings, repo), llm=llm, ui=UI.silent())
    [result] = report.results
    assert result.status == FixStatus.FIXED
    assert result.repro_status == "unavailable" and result.verified_by == "syntax"


def test_a_misbehaving_script_cannot_corrupt_the_repo_or_the_sandbox(repo, make_settings):
    # Overwrites the sandbox copy, then fails: "reproduced", but no fix can make it pass.
    vandal = script("open('calc.py', 'w').write('oops')\nassert False")
    llm = scripted([AVERAGE], [vandal], [fix_reply(GOOD_FIX)] * 3)
    report = run(settings(make_settings, repo), llm=llm, ui=UI.silent())
    [result] = report.results
    # Each attempt was judged against the real fix, not against the vandalized copy.
    assert [a.outcome for a in result.attempts] == ["rejected_repro"] * 3
    assert "oops" not in result.attempts[0].detail
    assert (repo / "calc.py").read_text() == CALC
