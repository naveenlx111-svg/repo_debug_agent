from conftest import ScriptedLLM, review_reply, write_tree

from repo_debug_agent.cache import ReplyCache
from repo_debug_agent.pipeline import run
from repo_debug_agent.ui import UI


def test_reply_cache_roundtrip_and_counts(tmp_path):
    cache = ReplyCache(tmp_path / "c")
    msgs = [{"role": "user", "content": "hi"}]
    key = cache.key("m", msgs, max_tokens=10)
    assert cache.key("m", msgs, max_tokens=10) == key
    assert cache.key("m", msgs, max_tokens=11) != key
    assert cache.key("other", msgs, max_tokens=10) != key
    assert cache.get(key) is None
    cache.put(key, '{"issues": []}')
    assert cache.get(key) == ('{"issues": []}', False)
    cache.put(key, '{"issues": [', truncated=True)
    assert cache.get(key) == ('{"issues": [', True)
    assert (cache.hits, cache.misses) == (2, 1)


def test_corrupt_entries_are_misses(tmp_path):
    cache = ReplyCache(tmp_path / "c")
    key = cache.key("m", [])
    path = tmp_path / "c" / key[:2] / f"{key}.json"
    path.parent.mkdir(parents=True)
    path.write_text("{truncated")
    assert cache.get(key) is None


def test_second_run_reuses_reviews_of_unchanged_files(tmp_path, make_settings):
    repo = write_tree(
        tmp_path / "repo", {"a.py": "def f():\n    return 1\n", "b.py": "def g():\n    return 2\n"}
    )
    llm = ScriptedLLM(lambda messages, json_mode: review_reply())
    settings = make_settings(repo, analyze_only=True, review_cache=True)

    run(settings, llm=llm, ui=UI.silent())
    assert len(llm.calls) == 2
    run(settings, llm=llm, ui=UI.silent())
    assert len(llm.calls) == 2  # both reviews came from the cache

    (repo / "b.py").write_text("def g():\n    return 3\n")
    run(settings, llm=llm, ui=UI.silent())
    assert len(llm.calls) == 3  # only the changed file was reviewed again


def test_unparseable_replies_are_not_cached(tmp_path, make_settings):
    repo = write_tree(tmp_path / "repo", {"a.py": "def f():\n    return 1\n"})
    llm = ScriptedLLM(lambda messages, json_mode: "sorry, I can't")
    settings = make_settings(repo, analyze_only=True, review_cache=True)
    run(settings, llm=llm, ui=UI.silent())
    run(settings, llm=llm, ui=UI.silent())
    assert len(llm.calls) == 2


def test_truncated_reviews_are_cached_with_their_partial_reply(tmp_path, make_settings):
    from repo_debug_agent.llm import LLMTruncatedError

    repo = write_tree(tmp_path / "repo", {"a.py": "def f(x):\n    return x[0]\n"})
    partial = '{"issues": [{"line": 2, "confidence": 0.9, "description": "IndexError"}, {"tr'
    calls = []

    def respond(messages, json_mode):
        calls.append(1)
        raise LLMTruncatedError("cut off", partial=partial)

    settings = make_settings(repo, analyze_only=True, review_cache=True)
    first = run(settings, llm=ScriptedLLM(respond), ui=UI.silent())
    second = run(settings, llm=ScriptedLLM(respond), ui=UI.silent())
    assert len(calls) == 1  # the doomed request isn't repeated
    assert len(first.results) == len(second.results) == 1
