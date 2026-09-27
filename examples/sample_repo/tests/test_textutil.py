from textutil import slugify, truncate, word_count


def test_word_count(tmp_path):
    path = tmp_path / "doc.txt"
    path.write_text("one two  three\nfour")
    assert word_count(path) == 4


def test_slugify():
    assert slugify("Hello, World!") == "hello-world"


def test_truncate_keeps_text_that_fits():
    assert truncate("hello", 5) == "hello"


def test_truncate_marks_cut():
    assert truncate("hello world", 8) == "hello..."
