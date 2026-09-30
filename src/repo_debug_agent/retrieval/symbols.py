"""Structural retrieval: who defines the names a chunk uses, and who uses the chunk.

For fixing a function, its callees' definitions and its callers (especially
the tests that call it) are usually far more useful context than whatever is
semantically nearest, and this costs nothing to compute.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable

from repo_debug_agent.models import Chunk

_IDENTIFIER = re.compile(r"[A-Za-z_$][\w$]*")
# Names too generic to link on; they would connect everything to everything.
_STOPWORDS = frozenset(
    {
        "self",
        "cls",
        "this",
        "super",
        "return",
        "if",
        "else",
        "elif",
        "for",
        "while",
        "in",
        "is",
        "not",
        "and",
        "or",
        "def",
        "class",
        "function",
        "func",
        "fn",
        "const",
        "let",
        "var",
        "new",
        "None",
        "True",
        "False",
        "null",
        "nil",
        "true",
        "false",
        "undefined",
        "import",
        "from",
        "export",
        "default",
        "public",
        "private",
        "protected",
        "static",
        "void",
        "int",
        "str",
        "float",
        "bool",
        "string",
        "len",
        "print",
        "range",
        "list",
        "dict",
        "set",
        "tuple",
        "type",
        "object",
        "get",
        "run",
        "main",
        "init",
        "__init__",
        "value",
        "data",
        "result",
        "item",
        "items",
        "name",
        "key",
        "args",
        "kwargs",
        "err",
        "error",
        "e",
        "x",
        "y",
        "i",
        "j",
        "k",
        "n",
        "async",
        "await",
        "try",
        "except",
        "catch",
        "finally",
        "raise",
        "throw",
        "with",
        "as",
        "pass",
        "break",
        "continue",
        "lambda",
        "yield",
        "struct",
        "impl",
        "pub",
        "mut",
        "package",
    }
)
# A name defined in more places than this is ambiguous; linking on it is noise.
MAX_DEFINITIONS = 3


def identifiers(code: str) -> set[str]:
    return {t for t in _IDENTIFIER.findall(code) if t not in _STOPWORDS and len(t) > 1}


class SymbolIndex:
    def __init__(self, chunks: Iterable[Chunk] = ()):
        self._chunks: dict[str, Chunk] = {}
        self._defined_by: dict[str, set[str]] = defaultdict(set)  # short name -> chunk keys
        self._used_by: dict[str, set[str]] = defaultdict(set)  # identifier -> chunk keys
        self._uses: dict[str, set[str]] = {}  # chunk key -> identifiers
        for chunk in chunks:
            self.add(chunk)

    def __len__(self) -> int:
        return len(self._chunks)

    def get(self, key: str) -> Chunk | None:
        return self._chunks.get(key)

    def chunks(self, file: str | None = None) -> list[Chunk]:
        found = [c for c in self._chunks.values() if file is None or c.file == file]
        return sorted(found, key=lambda c: (c.file, c.start_line))

    def add(self, chunk: Chunk) -> None:
        self._chunks[chunk.key] = chunk
        if chunk.is_definition:
            self._defined_by[chunk.short_name].add(chunk.key)
        uses = identifiers(chunk.code)
        self._uses[chunk.key] = uses
        for ident in uses:
            self._used_by[ident].add(chunk.key)

    def remove_file(self, file: str) -> None:
        for key in [k for k, c in self._chunks.items() if c.file == file]:
            chunk = self._chunks.pop(key)
            self._defined_by[chunk.short_name].discard(key)
            for ident in self._uses.pop(key, ()):
                self._used_by[ident].discard(key)

    def replace_file(self, file: str, chunks: Iterable[Chunk]) -> None:
        self.remove_file(file)
        for chunk in chunks:
            self.add(chunk)

    def related(self, chunk: Chunk, limit: int = 8) -> list[tuple[Chunk, str]]:
        """Definitions `chunk` depends on, then chunks that use it, each with a reason."""
        results: list[tuple[Chunk, str]] = []
        seen = {chunk.key}

        # Callees, in the order they first appear in the chunk's code.
        for ident in dict.fromkeys(_IDENTIFIER.findall(chunk.code)):
            if ident == chunk.short_name or ident in _STOPWORDS:
                continue
            definers = self._defined_by.get(ident, set())
            if not 0 < len(definers) <= MAX_DEFINITIONS:
                continue
            for key in sorted(definers - seen):
                seen.add(key)
                results.append((self._chunks[key], f"defines `{ident}`, used by `{chunk.name}`"))

        # Callers: tests first, since they document the expected behaviour.
        if len(self._defined_by.get(chunk.short_name, ())) <= MAX_DEFINITIONS:
            callers = [self._chunks[k] for k in self._used_by.get(chunk.short_name, set()) - seen]
            callers.sort(key=lambda c: (not _looks_like_test(c), c.file, c.start_line))
            for caller in callers:
                seen.add(caller.key)
                results.append((caller, f"uses `{chunk.short_name}`"))

        return results[:limit]


def _looks_like_test(chunk: Chunk) -> bool:
    return "test" in chunk.file.lower() or chunk.short_name.lower().startswith("test")
