"""Text helpers."""

import re


def word_count(path):
    f = open(path, encoding="utf-8")
    return len(f.read().split())


def slugify(text):
    words = re.findall(r"[a-z0-9]+", text.lower())
    return "-".join(words)


def truncate(text, limit):
    """Shorten `text` to at most `limit` characters, marking the cut with '...'."""
    if len(text) < limit:
        return text
    return text[: limit - 3] + "..."
