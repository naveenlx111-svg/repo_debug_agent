import subprocess

import pytest
from conftest import write_tree

from repo_debug_agent.crawler import discover, is_test_path


@pytest.mark.parametrize(
    ("rel", "expected"),
    [
        ("src/app.py", False),
        ("tests/test_app.py", True),
        ("pkg/app_test.go", True),
        ("web/cart.test.js", True),
        ("web/__tests__/cart.js", True),
        ("src/main/java/FooTest.java", True),
        ("conftest.py", True),
        ("src/testing_utils.py", False),
    ],
)
def test_is_test_path(rel, expected):
    assert is_test_path(rel) is expected


def test_discover_skips_junk_and_detects_languages(tmp_path):
    write_tree(
        tmp_path,
        {
            "app.py": "x = 1\n",
            "web/app.js": "let x = 1;\n",
            "web/app.min.js": "let x=1;\n",
            "node_modules/lib/index.js": "module.exports = 1;\n",
            ".venv/lib/site.py": "x = 1\n",
            "notes.txt": "hello\n",
            "tests/test_app.py": "def test(): pass\n",
        },
    )
    (tmp_path / "blob.py").write_bytes(b"\x00\x01binary")
    (tmp_path / "latin1.py").write_bytes("x = 'caf\xe9'\n".encode("latin-1"))

    found = {f.rel: f for f in discover(tmp_path)}
    assert set(found) == {"app.py", "web/app.js", "tests/test_app.py"}
    assert found["web/app.js"].language.name == "javascript"
    assert found["tests/test_app.py"].is_test


def test_discover_respects_gitignore(tmp_path):
    write_tree(tmp_path, {"keep.py": "x = 1\n", "gen/out.py": "x = 2\n", ".gitignore": "gen/\n"})
    try:
        subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("git not available")
    assert [f.rel for f in discover(tmp_path)] == ["keep.py"]


def test_discover_skips_large_files(tmp_path):
    write_tree(tmp_path, {"big.py": "x = 1\n" * 1000, "small.py": "x = 1\n"})
    assert [f.rel for f in discover(tmp_path, max_bytes=100)] == ["small.py"]
