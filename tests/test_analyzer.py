from conftest import ScriptedLLM, review_reply

from repo_debug_agent.analyzer import (
    Analyzer,
    consolidate,
    locate,
    parse_review,
    relevant_test_output,
    review_windows,
    select,
)
from repo_debug_agent.chunker import chunk_source
from repo_debug_agent.crawler import SourceFile
from repo_debug_agent.languages import BY_NAME
from repo_debug_agent.models import Chunk, Issue

SRC = """import os


def mean(values):
    return sum(values) / len(values)


def first(values):
    # comment
    return values[0]
"""


def chunks():
    return chunk_source("stats.py", SRC, BY_NAME["python"])


def test_parse_review_normalizes_model_output():
    data = {
        "issues": [
            {
                "line": "5",
                "symbol": "`mean()`",
                "severity": "HIGH",
                "confidence": 9,
                "description": "divides by zero on empty input",
            },
            {"line": 999, "severity": "weird", "confidence": "high", "issue": "out of window"},
            {"description": ""},
            "not a dict",
        ]
    }
    issues = parse_review(data, "stats.py", (1, 11))
    assert len(issues) == 2
    first, second = issues
    assert (first.line, first.symbol, first.severity, first.confidence) == (5, "mean", "high", 0.9)
    assert (second.line, second.severity, second.confidence) == (None, "medium", 0.85)


def test_parse_review_accepts_bare_lists_and_single_objects():
    assert len(parse_review([{"description": "x"}], "a.py", (1, 5))) == 1
    assert len(parse_review({"description": "x", "line": 2}, "a.py", (1, 5))) == 1
    assert parse_review({"issues": "none"}, "a.py", (1, 5)) == []


def test_locate_prefers_line_and_handles_sloppy_names():
    cs = chunks()
    assert locate(cs, 5, None).name == "mean"
    assert locate(cs, 5, "wrong_name").name == "mean"
    assert locate(cs, None, "first").name == "first"
    assert locate(cs, 9, None).name == "first"  # comment line inside the function
    assert locate(cs, 3, None).name == "mean"  # blank line just above a function
    assert locate(cs, None, None) is None


def test_consolidate_merges_issues_in_the_same_chunk():
    a = Issue("f.py", "bug A", "medium", 0.7, line=5, chunk_key="f.py::mean", symbol="mean")
    b = Issue("f.py", "bug B", "high", 0.9, line=5, chunk_key="f.py::mean", symbol="mean")
    c = Issue("f.py", "bug C", "low", 0.9, line=9, chunk_key="f.py::first", symbol="first")
    lost = Issue("f.py", "unlocated", "high", 0.9)
    merged = consolidate([a, b, c, lost])
    assert len(merged) == 2
    mean = next(i for i in merged if i.symbol == "mean")
    assert mean.severity == "high" and mean.confidence == 0.9
    assert "bug A" in mean.description and "bug B" in mean.description


def test_select_filters_ranks_and_numbers():
    issues = [
        Issue("a.py", "low sev", "low", 0.9, chunk_key="k1"),
        Issue("a.py", "unsure", "critical", 0.3, chunk_key="k2"),
        Issue("a.py", "crit", "critical", 0.8, chunk_key="k3"),
        Issue("a.py", "high", "high", 0.7, chunk_key="k4"),
    ]
    picked = select(issues, min_confidence=0.6, limit=2)
    assert [(i.id, i.description) for i in picked] == [("B1", "crit"), ("B2", "high")]


def test_review_windows_cut_at_chunk_boundaries():
    cs = [
        Chunk("f", "python", "function", f"f{i}", 1 + i * 50, 50 + i * 50, "", f"k{i}")
        for i in range(6)
    ]
    windows = review_windows(300, cs, max_lines=120)
    assert windows == [(1, 100), (101, 200), (201, 300)]
    assert review_windows(10, cs, max_lines=120) == [(1, 10)]


def test_review_windows_split_oversized_chunks():
    cs = [Chunk("f", "python", "function", "huge", 1, 500, "", "k")]
    assert review_windows(500, cs, max_lines=200) == [(1, 200), (201, 400), (401, 500)]


def test_analyzer_combines_static_and_llm_findings():
    src = "def f():\n    return undefined_name\n\n\ndef g(x):\n    return x[0]\n"
    sf = SourceFile(path=None, rel="m.py", language=BY_NAME["python"], is_test=False)
    llm = ScriptedLLM(
        lambda messages, json_mode: review_reply(
            {
                "line": 6,
                "symbol": "g",
                "severity": "medium",
                "confidence": 0.8,
                "description": "IndexError on empty list",
            }
        )
    )
    review = Analyzer(llm, "fake", 400).review(sf, src, chunk_source("m.py", src, sf.language))
    assert review.error is None
    by_source = {i.source: i for i in review.issues}
    assert by_source["static"].symbol == "f" and "undefined_name" in by_source["static"].description
    assert by_source["llm"].symbol == "g"
    # the static finding is passed to the model as a hint
    assert "undefined name 'undefined_name'" in llm.calls[0][1]["content"]


def test_analyzer_survives_garbage_replies():
    src = "def g(x):\n    return x\n"
    sf = SourceFile(path=None, rel="m.py", language=BY_NAME["python"], is_test=False)
    llm = ScriptedLLM(lambda messages, json_mode: "I think the code is fine!")
    review = Analyzer(llm, "fake", 400).review(sf, src, chunk_source("m.py", src, sf.language))
    assert review.issues == [] and "no JSON" in review.error


def test_relevant_test_output():
    output = "\n".join(
        ["noise"] * 30
        + ["tests/test_stats.py:12: in test_mean", "stats.py:5: ZeroDivisionError"]
        + ["noise"] * 30
    )
    excerpt = relevant_test_output(output, "stats.py")
    assert "ZeroDivisionError" in excerpt and excerpt.count("noise") < 30
    assert relevant_test_output(output, "other.py") is None


def test_parse_review_drops_findings_the_models_own_trace_cleared():
    data = {
        "issues": [
            {
                "line": 2,
                "suspicion": "might divide by zero",
                "trace": "guarded above",
                "confirmed": False,
                "description": "x",
            },
            {
                "line": 3,
                "suspicion": "index error",
                "trace": "[] -> IndexError",
                "confirmed": "true",
                "description": "crashes on []",
            },
        ]
    }
    [issue] = parse_review(data, "a.py", (1, 5))
    assert issue.description == "crashes on []"


def test_truncated_review_is_split_and_retried():
    from repo_debug_agent.llm import LLMTruncatedError

    src = "\n\n".join(f"def f{i}(x):\n    return x[{i}]" for i in range(30))
    sf = SourceFile(path=None, rel="m.py", language=BY_NAME["python"], is_test=False)
    windows = []

    def respond(messages, json_mode):
        listing = messages[1]["content"].split("```python\n")[-1]
        numbers = [int(line.split("|")[0]) for line in listing.splitlines() if "|" in line]
        windows.append((numbers[0], numbers[-1]))
        if numbers[-1] - numbers[0] > 60:
            raise LLMTruncatedError("reply was cut off")
        return review_reply({"line": numbers[0] + 1, "description": "bug", "confidence": 0.9})

    review = Analyzer(ScriptedLLM(respond), "fake", 400).review(
        sf, src, chunk_source("m.py", src, sf.language)
    )
    assert review.error is None
    assert windows[0] == (1, 89) and len(windows) > 2
    covered = sorted(w for w in windows[1:] if w[1] - w[0] <= 60)
    assert covered[0][0] == 1 and covered[-1][1] == 89
    assert len(review.issues) == len(covered)


def test_truncated_review_keeps_complete_findings_instead_of_losing_the_file():
    from repo_debug_agent.llm import LLMTruncatedError

    src = "def f(x):\n    return x[0]\n\n\ndef g(x):\n    return x[1]\n"
    sf = SourceFile(path=None, rel="m.py", language=BY_NAME["python"], is_test=False)
    partial = (
        '{"issues": [{"line": 2, "symbol": "f", "confirmed": true, "confidence": 0.9, '
        '"description": "IndexError on []"}, {"line": 5, "trace": "Wait... No. Wait'
    )
    calls = []

    def respond(messages, json_mode):
        calls.append(1)
        raise LLMTruncatedError("cut off", partial=partial)

    review = Analyzer(ScriptedLLM(respond), "fake", 400).review(
        sf, src, chunk_source("m.py", src, sf.language)
    )
    assert [i.symbol for i in review.issues] == ["f"]
    assert "kept its 1 complete finding" in review.error
    assert len(calls) == 1  # salvaged, so no split-and-retry


def test_finished_reply_with_broken_json_keeps_every_complete_finding():
    src = "def f(x):\n    return x[0]\n\n\ndef g(x):\n    return x[1]\n"
    sf = SourceFile(path=None, rel="m.py", language=BY_NAME["python"], is_test=False)
    broken = (
        '{"issues": [{"line": 2, "description": "IndexError in f"}, '
        '{"line": 6, "description": "IndexError in g"}, {"line": ": ", "trace": "}'
    )
    review = Analyzer(ScriptedLLM(lambda m, j: broken), "fake", 400).review(
        sf, src, chunk_source("m.py", src, sf.language)
    )
    assert sorted(i.symbol for i in review.issues) == ["f", "g"]


def test_review_token_cap_scales_with_window():
    from repo_debug_agent.analyzer import REVIEW_MAX_TOKENS, review_max_tokens

    assert review_max_tokens(33) < 1500  # a loop in a small file stops early
    assert review_max_tokens(400) == REVIEW_MAX_TOKENS
