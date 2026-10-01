from repo_debug_agent.chunker import chunk_source
from repo_debug_agent.fixer import (
    file_excerpt,
    fit_replacement,
    parse_fix_reply,
    splice,
    whole_class,
)
from repo_debug_agent.languages import BY_NAME

SRC = """class Cart:
    tax = 0.1

    def __init__(self, items=[]):
        self.items = items

    def total(self):
        return sum(i.price for i in self.items)
"""


def chunk(name, src=SRC, lang="python"):
    return next(c for c in chunk_source("m.py", src, BY_NAME[lang]) if c.name == name)


def test_parse_fix_reply_variants():
    ok = parse_fix_reply("```python\ndef f():\n    return 1\n```\nEXPLANATION: returns one")
    assert ok.code == "def f():\n    return 1\n" and ok.explanation == "returns one"
    dismissed = parse_fix_reply("NOT_A_BUG: the caller guarantees a non-empty list")
    assert dismissed.dismissed and "non-empty" in dismissed.explanation
    assert parse_fix_reply("**NOT_A_BUG**: fine").dismissed
    assert "no fenced code block" in parse_fix_reply("just change the < to <=").error


def test_parse_fix_reply_analysis_first_structure():
    reply = (
        "ANALYSIS: For [4, 1, 3, 2], mid is 2, so the code averages ordered[2] and ordered[3],\n"
        "which is 3.5 instead of 2.5. The report is right.\n"
        "VERDICT: BUG\n"
        "```python\ndef median(v):\n    return v\n```\n"
        "EXPLANATION: Use the two middle elements."
    )
    parsed = parse_fix_reply(reply)
    assert parsed.code == "def median(v):\n    return v\n"
    assert parsed.analysis.startswith("For [4, 1, 3, 2]") and parsed.analysis.endswith("right.")
    assert parsed.explanation == "Use the two middle elements."

    no = parse_fix_reply("ANALYSIS: len 5 < 5 is False, so it truncates.\n**VERDICT:** NOT_A_BUG")
    assert no.dismissed and no.explanation == "len 5 < 5 is False, so it truncates."


def test_parse_fix_reply_accepts_an_unclosed_fence_before_the_explanation():
    # Seen from Qwen3.5-9B: the closing ``` is skipped and EXPLANATION follows the code.
    reply = (
        "```python\ndef mean(values):\n"
        "    return 0.0 if not values else sum(values) / len(values)\n"
        "EXPLANATION: Return 0.0 for empty input."
    )
    parsed = parse_fix_reply(reply)
    assert parsed.code.rstrip() == (
        "def mean(values):\n    return 0.0 if not values else sum(values) / len(values)"
    )
    assert parsed.explanation == "Return 0.0 for empty input."


def test_fit_replacement_reindents_a_dedented_method():
    target = chunk("Cart.__init__")
    code = "def __init__(self, items=None):\n    self.items = list(items or [])\n"
    fitted, error = fit_replacement(code, target)
    assert error is None
    assert fitted == "    def __init__(self, items=None):\n        self.items = list(items or [])"


def test_fit_replacement_extracts_method_from_a_returned_class():
    target = chunk("Cart.total")
    code = (
        "class Cart:\n    tax = 0.1\n\n    def total(self):\n"
        "        return sum(i.price * i.quantity for i in self.items)\n"
    )
    fitted, error = fit_replacement(code, target)
    assert error is None
    assert fitted.startswith("    def total(self):") and "class Cart" not in fitted


def test_fit_replacement_rejects_the_wrong_definition():
    fitted, error = fit_replacement("def something_else():\n    pass\n", chunk("Cart.total"))
    assert fitted is None and "does not define `total`" in error


def test_splice_replaces_exactly_the_chunk_lines():
    target = chunk("Cart.total")
    new = splice(SRC, target, "    def total(self):\n        return 0")
    assert new.split("\n")[:6] == SRC.split("\n")[:6]
    assert new.endswith("    def total(self):\n        return 0\n")


def test_whole_class_expands_header_chunks():
    header = chunk("Cart")
    assert (header.start_line, header.end_line) == (1, 2)
    full = whole_class(header, SRC)
    assert (full.start_line, full.end_line) == (1, 8) and "def total" in full.code


def test_file_excerpt_marks_the_target_and_windows_large_files():
    excerpt, whole = file_excerpt(SRC, chunk("Cart.total"))
    assert whole and "# >>> `Cart.total` (lines 7-8) goes here" in excerpt
    assert "sum(i.price" not in excerpt

    big = "\n".join(f"x{i} = {i}" for i in range(1000)) + "\n\ndef f():\n    return 1\n"
    target = chunk("f", big)
    excerpt, whole = file_excerpt(big, target)
    assert not whole
    assert "x0 = 0" in excerpt and "omitted" in excerpt and "x999 = 999" in excerpt
    assert "x500 = 500" not in excerpt


def test_fix_token_budget_scales_with_the_code():
    from repo_debug_agent.fixer import FIX_MAX_TOKENS, fix_max_tokens

    small = chunk("Cart.total")
    assert fix_max_tokens(small) < 1200  # a loop on a small function stops early
    big = small.__class__(**{**small.__dict__, "code": "x = 1\n" * 3000})
    assert fix_max_tokens(big) == FIX_MAX_TOKENS
