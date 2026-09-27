import hashlib

import pytest

from repo_debug_agent.chunker import chunk_source
from repo_debug_agent.languages import BY_NAME
from repo_debug_agent.retrieval import ContextRetriever, SymbolIndex

LIB = """def parse_price(text):
    return float(text.strip("$"))


def format_price(amount):
    return f"${amount:.2f}"
"""
APP = """from lib import parse_price


def order_total(lines):
    return sum(parse_price(line) for line in lines)
"""
TESTS = """from app import order_total


def test_order_total():
    assert order_total(["$1", "$2"]) == 3
"""


def build():
    py = BY_NAME["python"]
    chunks = (
        chunk_source("lib.py", LIB, py)
        + chunk_source("app.py", APP, py)
        + chunk_source("tests/test_app.py", TESTS, py)
    )
    return SymbolIndex(chunks)


def test_related_finds_callees_and_callers():
    index = build()
    target = next(c for c in index.chunks("app.py") if c.name == "order_total")
    related = [(c.name, why) for c, why in index.related(target)]
    names = [name for name, _ in related]
    assert names[0] == "parse_price"  # its dependency
    assert "test_order_total" in names  # the test that pins its behaviour
    assert "format_price" not in names


def test_replace_file_updates_links():
    index = build()
    index.replace_file(
        "app.py",
        chunk_source("app.py", "def order_total(lines):\n    return 0\n", BY_NAME["python"]),
    )
    target = next(c for c in index.chunks("app.py") if c.name == "order_total")
    assert "parse_price" not in [c.name for c, _ in index.related(target)]


def test_render_respects_budget_and_exclusions():
    index = build()
    target = next(c for c in index.chunks("app.py") if c.name == "order_total")
    text = ContextRetriever(index, None, budget_chars=10_000).render(target, "query")
    assert "lib.py:1-2 `parse_price`" in text and "```python" in text
    assert ContextRetriever(index, None, budget_chars=10).render(target, "query") == ""
    excluded = ContextRetriever(index, None, 10_000).render(target, "q", exclude_files={"lib.py"})
    assert "parse_price" not in excluded.split("\n")[0]


def hash_embedding():
    """A deterministic bag-of-words embedder, so the test needs no model download."""
    from chromadb.api.types import EmbeddingFunction

    class HashEmbedding(EmbeddingFunction):
        def __init__(self):
            pass

        def __call__(self, input):
            vectors = []
            for text in input:
                vec = [0.0] * 64
                for word in text.lower().split():
                    vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % 64] += 1.0
                vectors.append(vec)
            return vectors

        @staticmethod
        def name():
            return "hash-test"

        def get_config(self):
            return {}

        @staticmethod
        def build_from_config(config):
            return HashEmbedding()

    return HashEmbedding()


def test_vector_index_is_incremental(tmp_path):
    pytest.importorskip("chromadb")
    from repo_debug_agent.retrieval import VectorIndex

    index = build()
    chunks = index.chunks()
    vi = VectorIndex(tmp_path / "idx", tmp_path, "hash", embedding_function=hash_embedding())
    embedded, removed = vi.sync(chunks)
    assert (embedded, removed) == (vi.count(), 0) and vi.count() > 0

    assert vi.sync(chunks) == (0, 0)  # nothing changed: nothing re-embedded

    changed = [c for c in chunks if c.name != "format_price"]
    assert vi.sync(changed) == (0, 1)

    hits = vi.search("parse price text strip float", n=3)
    assert hits and hits[0][0] == "lib.py::parse_price"
