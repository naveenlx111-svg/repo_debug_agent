import os
import subprocess

import pytest
from conftest import write_tree

from repo_debug_agent.workspace import ApplyConflict, Workspace


def test_edits_stay_in_memory_until_applied(tmp_path):
    write_tree(tmp_path, {"a.py": "x = 1\n"})
    ws = Workspace(tmp_path)
    ws.write("a.py", "x = 2\n")
    assert ws.read("a.py") == "x = 2\n"
    assert (tmp_path / "a.py").read_text() == "x = 1\n"
    assert ws.changed_files() == ["a.py"]
    assert ws.apply() == ["a.py"]
    assert (tmp_path / "a.py").read_text() == "x = 2\n"


def test_reverting_to_original_is_not_a_change(tmp_path):
    write_tree(tmp_path, {"a.py": "x = 1\n"})
    ws = Workspace(tmp_path)
    ws.write("a.py", "x = 2\n")
    ws.write("a.py", "x = 1\n")
    assert ws.changed_files() == [] and ws.diff() == ""


def test_crlf_line_endings_are_preserved(tmp_path):
    (tmp_path / "w.py").write_bytes(b"a = 1\r\nb = 2\r\n")
    ws = Workspace(tmp_path)
    assert ws.read("w.py") == "a = 1\nb = 2\n"
    ws.write("w.py", "a = 1\nb = 3\n")
    ws.apply()
    assert (tmp_path / "w.py").read_bytes() == b"a = 1\r\nb = 3\r\n"


def test_apply_refuses_files_changed_during_the_run(tmp_path):
    write_tree(tmp_path, {"a.py": "x = 1\n"})
    ws = Workspace(tmp_path)
    ws.write("a.py", "x = 2\n")
    (tmp_path / "a.py").write_text("x = 99\n")  # the user edited it meanwhile
    with pytest.raises(ApplyConflict):
        ws.apply()
    assert (tmp_path / "a.py").read_text() == "x = 99\n"


def test_sandbox_copies_code_links_dependencies_and_mirrors_edits(tmp_path):
    repo = write_tree(
        tmp_path / "repo",
        {
            "a.py": "x = 1\n",
            "node_modules/dep/index.js": "module.exports = 1;\n",
            ".git/HEAD": "ref\n",
            "__pycache__/a.pyc": "junk",
        },
    )
    ws = Workspace(repo)
    ws.write("a.py", "x = 2\n")
    sandbox = ws.materialize()
    try:
        assert (sandbox / "a.py").read_text() == "x = 2\n"
        assert (sandbox / "node_modules").is_symlink()
        assert not (sandbox / ".git").exists() and not (sandbox / "__pycache__").exists()
        ws.write("a.py", "x = 3\n")
        assert (sandbox / "a.py").read_text() == "x = 3\n"
        assert (repo / "a.py").read_text() == "x = 1\n"
    finally:
        ws.cleanup()
    assert not sandbox.exists()


def test_diff_is_git_apply_compatible(tmp_path):
    write_tree(tmp_path, {"a.py": "x = 1\ny = 2", "b.py": "z = 1\n"})
    ws = Workspace(tmp_path)
    ws.write("a.py", "x = 10\ny = 2")  # no trailing newline, on purpose
    ws.write("b.py", "z = 2\n")
    patch = ws.diff()
    assert "\\ No newline at end of file" in patch
    patch_file = tmp_path.parent / "fix.patch"
    patch_file.write_text(patch)
    try:
        result = subprocess.run(
            ["git", "apply", "--check", str(patch_file)], cwd=tmp_path, capture_output=True
        )
    except OSError:
        pytest.skip("git not available")
    assert result.returncode == 0, result.stderr
    os.remove(patch_file)
