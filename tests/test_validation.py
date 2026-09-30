import shutil
from pathlib import Path

import pytest

from repo_debug_agent.validation import (
    TestRun,
    check_edit,
    count_failures,
    judge,
    python_findings,
    run_tests,
)


def test_python_findings_reports_serious_problems_only():
    src = "import os\n\ndef f():\n    return undefined_thing\n"
    findings = python_findings(src)
    assert [(f.line, f.kind) for f in findings] == [(4, "UndefinedName")]  # unused import ignored


def test_python_findings_reports_syntax_errors():
    assert python_findings("def f(:\n")[0].kind == "SyntaxError"


def test_check_edit_python_rejects_syntax_errors_and_new_undefined_names():
    before = "def f(x):\n    return x\n"
    assert check_edit("python", Path("m.py"), before, "def f(x):\n    return x +\n").ok is False
    bad = check_edit("python", Path("m.py"), before, "def f(x):\n    return math.sqrt(x)\n")
    assert bad.ok is False and "math" in bad.message
    assert check_edit("python", Path("m.py"), before, "def f(x):\n    return x or 0\n").ok is True


def test_check_edit_python_ignores_preexisting_problems():
    before = "def f():\n    return missing\n"
    after = "def f():\n    return missing or 1\n"
    assert check_edit("python", Path("m.py"), before, after).ok is True


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_check_edit_javascript_uses_node():
    before = "function f(a) { return a; }\n"
    assert check_edit("javascript", Path("a.js"), before, "function f(a) { return a + 1; }\n").ok
    result = check_edit("javascript", Path("a.js"), before, "function f(a) { return a + ; }\n")
    assert result.ok is False


def test_check_edit_unknown_language_is_unverified():
    assert check_edit("kotlin", Path("a.kt"), "fun f() {}", "fun g() {}").ok is None


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        ("=== 2 failed, 3 passed in 0.1s ===", 2),
        ("=== 1 failed, 4 passed, 2 errors in 0.1s ===", 3),
        ("FAILED (failures=2, errors=1)", 3),
        ("Tests:       4 failed, 10 passed, 14 total", 4),
        ("  3 passing\n  2 failing", 2),
        ("test result: FAILED. 3 passed; 2 failed; 0 ignored", 2),
        ("--- FAIL: TestA\n--- FAIL: TestB\nFAIL", 2),
        ("something went wrong", None),
    ],
)
def test_count_failures(output, expected):
    assert count_failures(output) == expected


def run(code, failures):
    return TestRun("t", code, failures, "", 0.1)


def test_judge():
    assert judge(run(1, 3), run(0, 0)).improved
    assert judge(run(1, 3), run(1, 2)).improved
    assert judge(run(1, 3), run(1, 3)).acceptable
    assert not judge(run(1, 3), run(1, 4)).acceptable
    assert not judge(run(0, 0), run(1, 1)).acceptable
    assert not judge(run(1, 3), run(None, None)).acceptable
    neutral = judge(run(1, None), run(1, None))
    assert neutral.acceptable and not neutral.improved


def test_run_tests_prefers_code_in_cwd(tmp_path):
    (tmp_path / "mod.py").write_text("VALUE = 42\n")
    result = run_tests('python -c "import mod; assert mod.VALUE == 42"', tmp_path, timeout=60)
    assert result.passed, result.output


def test_run_tests_timeout(tmp_path):
    result = run_tests('python -c "import time; time.sleep(30)"', tmp_path, timeout=1)
    assert result.exit_code is None and "timed out" in result.summary()


def test_check_edit_python_rejects_new_invalid_escape_sequences(capsys):
    # Seen in a real run: a "fix" turned "\\ No newline" into "\ No newline".
    before = 'MARK = "\\\\ No newline"\n'
    after = 'MARK = "\\ No newline"\n'
    result = check_edit("python", Path("m.py"), before, after)
    assert result.ok is False and "invalid escape" in result.message
    assert capsys.readouterr().err == ""  # the warning is captured, not printed
