import pytest

from repo_debug_agent.llm import parse_json, pick_code_block, strip_reasoning


def test_strip_reasoning():
    assert strip_reasoning('<think>hmm {"a": 1}</think>\nanswer') == "answer"
    assert strip_reasoning("<think>never finished") == ""


@pytest.mark.parametrize(
    "reply",
    [
        '{"issues": []}',
        '```json\n{"issues": []}\n```',
        'Sure! Here is the result:\n{"issues": []}\nHope that helps.',
        '<think>{"issues": [1]} maybe</think>{"issues": []}',
    ],
)
def test_parse_json_tolerates_wrapping(reply):
    assert parse_json(reply) == {"issues": []}


def test_parse_json_skips_invalid_candidates():
    assert parse_json('see {not json} then {"ok": true}') == {"ok": True}
    assert parse_json("[1, 2]") == [1, 2]
    with pytest.raises(ValueError):
        parse_json("no json here")


def test_pick_code_block_prefers_code_over_diffs_and_logs():
    reply = "```diff\n- a\n+ b\n```\n```python\ndef f():\n    return 1\n```\n```\nshort\n```"
    assert pick_code_block(reply) == "def f():\n    return 1\n"


def test_pick_code_block_ignores_fences_inside_code_lines():
    reply = '```python\ndef f():\n    return "```"\n```'
    assert pick_code_block(reply) == 'def f():\n    return "```"\n'


def test_pick_code_block_none_when_unterminated():
    assert pick_code_block("```python\ndef f():\n    return 1\n") is None
