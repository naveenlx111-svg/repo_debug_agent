"""Split source files into function / method / class-header chunks with exact line ranges.

Chunks serve two purposes: they are the retrieval units for context, and they
are the edit units for fixes (the fixer rewrites one chunk's line range). So
ranges must be exact. Python is parsed with `ast`. Brace languages are parsed
structurally: strings and comments are masked out, `{...}` blocks are matched,
and each top-level block's header is classified. Code outside any definition
becomes "<module>" chunks.
"""

from __future__ import annotations

import ast
import re
from collections import Counter
from dataclasses import dataclass, field

from repo_debug_agent.languages import Language
from repo_debug_agent.models import Chunk

MAX_MODULE_CHUNK_LINES = 120


@dataclass
class _Span:
    kind: str
    name: str
    start: int  # 1-based inclusive
    end: int  # 1-based inclusive
    body_end: int = 0  # classes: end of the whole class, members included


def chunk_source(rel: str, source: str, language: Language) -> list[Chunk]:
    lines = source.split("\n")
    if language.family == "python":
        spans = _python_spans(source)
    elif language.family == "ruby":
        spans = _ruby_spans(lines)
    else:
        spans = _brace_spans(source, language)

    if spans is None:  # unparseable file: treat it as one unit
        spans = [_Span("module", "<module>", 1, len(lines))]
    spans += _module_spans(lines, spans, language)
    spans.sort(key=lambda s: (s.start, -s.end))
    return _materialize(rel, language, lines, spans)


def _materialize(rel: str, language: Language, lines: list[str], spans: list[_Span]) -> list[Chunk]:
    chunks = []
    seen: Counter[str] = Counter()
    for span in spans:
        start, end = max(1, span.start), min(span.end, len(lines))
        while end > start and not lines[end - 1].strip():
            end -= 1
        code = "\n".join(lines[start - 1 : end])
        if not code.strip():
            continue
        seen[span.name] += 1
        key = f"{rel}::{span.name}" + (f"#{seen[span.name]}" if seen[span.name] > 1 else "")
        body_end = min(span.body_end, len(lines)) if span.body_end else end
        chunks.append(
            Chunk(rel, language.name, span.kind, span.name, start, end, code, key, body_end)
        )
    return chunks


# --------------------------------------------------------------------------- python


def _python_spans(source: str) -> list[_Span] | None:
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return None

    spans: list[_Span] = []
    defs = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)

    def first_line(node: ast.stmt) -> int:
        return min([node.lineno, *(d.lineno for d in getattr(node, "decorator_list", []))])

    def visit(body: list[ast.stmt], prefix: str, in_class: bool) -> None:
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                kind = "method" if in_class else "function"
                spans.append(_Span(kind, prefix + node.name, first_line(node), node.end_lineno))
            elif isinstance(node, ast.ClassDef):
                name = prefix + node.name
                start = first_line(node)
                members = [n for n in node.body if isinstance(n, defs)]
                header_end = first_line(members[0]) - 1 if members else node.end_lineno
                spans.append(_Span("class", name, start, max(start, header_end), node.end_lineno))
                visit(node.body, name + ".", in_class=True)

    visit(tree.body, "", in_class=False)
    return spans


# --------------------------------------------------------------------------- brace languages

_TOKEN_START = re.compile(r"//|/\*|[\"'`#]")
_CHAR_LITERAL = re.compile(r"'(?:\\.[^'\n]{0,8}|[^'\\\n])'")
_NOT_NEWLINE = re.compile(r"[^\n]")


def _blank(text: str) -> str:
    return _NOT_NEWLINE.sub(" ", text)


def mask_code(source: str, language: Language) -> str:
    """Blank out comments and string/char literals, keeping every newline and column.

    Brace matching on the result is not fooled by "{" inside strings or comments.
    """
    out: list[str] = []
    i, n = 0, len(source)
    while i < n:
        m = _TOKEN_START.search(source, i)
        if m is None:
            out.append(source[i:])
            break
        j = m.start()
        out.append(source[i:j])
        tok = m.group()

        if tok == language.line_comment:
            end = source.find("\n", j)
            end = n if end == -1 else end
            out.append(_blank(source[j:end]))
            i = end
        elif tok == "/*":
            end = source.find("*/", j + 2)
            end = n if end == -1 else end + 2
            out.append(_blank(source[j:end]))
            i = end
        elif (
            tok == '"'
            or (tok == "`" and language.backtick_strings)
            or (tok == "'" and not language.single_quote_is_char)
        ):
            end = _string_end(source, j, multiline=tok == "`")
            closed = end - 1 > j and source[end - 1] == tok
            out.append(
                tok + _blank(source[j + 1 : end - 1 if closed else end]) + (tok if closed else "")
            )
            i = end
        elif tok == "'":
            lit = _CHAR_LITERAL.match(source, j)
            if lit:  # otherwise a Rust lifetime or C++ digit separator
                out.append(_blank(lit.group()))
                i = lit.end()
            else:
                out.append(tok)
                i = j + 1
        else:
            out.append(tok)
            i = j + 1
    return "".join(out)


def _string_end(source: str, start: int, multiline: bool) -> int:
    """Index just past the closing quote of the string starting at `start`."""
    quote = source[start]
    i, n = start + 1, len(source)
    while i < n:
        c = source[i]
        if c == "\\":
            i += 2
            continue
        if c == quote:
            return i + 1
        if c == "\n" and not multiline:
            return i  # unterminated on this line; stop here
        i += 1
    return n


@dataclass
class _Block:
    open_line: int  # 0-based
    open_col: int
    close_line: int = -1
    children: list[_Block] = field(default_factory=list)


def _parse_blocks(masked_lines: list[str]) -> list[_Block]:
    roots: list[_Block] = []
    stack: list[_Block] = []
    for ln, text in enumerate(masked_lines):
        for col, ch in enumerate(text):
            if ch == "{":
                block = _Block(ln, col)
                (stack[-1].children if stack else roots).append(block)
                stack.append(block)
            elif ch == "}" and stack:
                stack.pop().close_line = ln
    for block in stack:  # unbalanced: close at EOF
        block.close_line = len(masked_lines) - 1
    return roots


_CONTROL = frozenset(
    {
        "if",
        "else",
        "for",
        "while",
        "do",
        "switch",
        "try",
        "catch",
        "finally",
        "return",
        "with",
        "using",
        "lock",
        "foreach",
        "synchronized",
        "when",
        "loop",
        "match",
        "select",
        "defer",
        "go",
    }
)
_NOT_NAMES = _CONTROL | {
    "function",
    "func",
    "fn",
    "new",
    "sizeof",
    "typeof",
    "await",
    "super",
    "this",
    "async",
    "require",
    "import",
    "assert",
    "static",
    "unsafe",
}
_ANNOTATION = re.compile(r"@[\w.]+(?:\([^()]*\))?")
_GO_METHOD = re.compile(r"^func\s*\(\s*\w*\s*\*?\s*([A-Za-z_]\w*)[^)]*\)\s*([A-Za-z_]\w*)")
_GO_TYPE = re.compile(r"^type\s+([A-Za-z_]\w*)\s+(?:struct|interface)\b")
_CONTAINER_KEYWORDS = frozenset(
    {
        "class",
        "struct",
        "interface",
        "enum",
        "trait",
        "impl",
        "namespace",
        "mod",
        "module",
        "object",
        "record",
        "protocol",
        "extension",
        "union",
    }
)
# Lookahead for the name so "enum class Color" can match "class" after "enum".
_CONTAINER = re.compile(
    r"\b(" + "|".join(sorted(_CONTAINER_KEYWORDS)) + r")\b(?=\s*(?:<[^>]*>\s*)?([A-Za-z_$][\w$]*))"
)
# Unnamed wrappers whose contents are still top-level declarations.
_TRANSPARENT = re.compile(r'^(?:extern\s*(?:"\s*")?|namespace|export\s+default)\s*$')
_ASSIGNMENT = re.compile(
    r"^(?:(?:export|const|let|var|public|private|protected|static|readonly|final)\s+)*"
    r"([A-Za-z_$][\w$]*)\s*(?::[^=]*)?=(?![=>])"
)
_CALL_LIKE = re.compile(r"([A-Za-z_$][\w$]*(?:(?:\.|::)[A-Za-z_$][\w$]*)*)\s*(?:<[^()]*>)?\s*\(")


def _classify(header: str) -> tuple[str, str] | None:
    """(kind, name) for the declaration that opens a block, or None if it isn't one."""
    h = " ".join(_ANNOTATION.sub(" ", header).split())
    first = re.match(r"[A-Za-z_]\w*", h)
    if not h or (first and first.group() in _CONTROL):
        return None
    if m := _GO_METHOD.match(h):
        return "method", f"{m[1]}.{m[2]}"
    if m := _GO_TYPE.match(h):
        return "class", m[1]
    for m in _CONTAINER.finditer(h):
        if m[2] in _CONTAINER_KEYWORDS:
            continue
        if "(" in h[: m.start()]:
            break  # keyword inside a parameter list
        rest = h[m.end(2) :].lstrip()
        if m[1] in ("struct", "union", "enum") and "(" in rest and not rest.startswith("("):
            break  # C function returning a struct: `struct node *make(int v)`
        name = m[2]
        if m[1] == "impl" and (target := re.search(r"\bfor\s+([A-Za-z_]\w*)", h)):
            name = target[1]
        return "class", name
    if m := _ASSIGNMENT.match(h):
        rest = h[m.end() :]
        if "=>" in rest or re.match(r"\s*(?:async\s+)?function\b", rest):
            return "function", m[1]
        return None  # data (object literal etc.): left to <module> chunks
    for m in _CALL_LIKE.finditer(h):
        name = m[1].replace("::", ".")
        if name.rsplit(".", 1)[-1] not in _NOT_NAMES:
            return "function", name
    return None


def _brace_spans(source: str, language: Language) -> list[_Span]:
    masked = mask_code(source, language).split("\n")
    spans: list[_Span] = []

    def header_start(block: _Block, floor: int) -> int:
        """First line of the declaration whose body opens at `block`."""
        start = block.open_line
        ln = block.open_line - 1
        while ln >= floor:
            text = masked[ln].strip()
            if not text or text.endswith((";", "}", "{")):
                break
            if text.startswith("#") and not text.startswith("#["):
                break  # preprocessor line (Rust #[attributes] do belong to the item)
            start = ln
            ln -= 1
        return start

    def header_text(block: _Block, start: int) -> str:
        parts = masked[start : block.open_line] + [masked[block.open_line][: block.open_col]]
        return " ".join(parts)

    def visit(blocks: list[_Block], floor: int, prefix: str) -> int | None:
        """Emit spans for `blocks`; return the first emitted start line (0-based)."""
        first_start = None
        prev_close = floor - 1
        for block in blocks:
            start = header_start(block, max(floor, prev_close + 1))
            prev_close = block.close_line
            header = header_text(block, start)
            found = _classify(header)
            if found is None:
                if _TRANSPARENT.match(" ".join(header.split())):
                    inner = visit(block.children, block.open_line + 1, prefix)
                    first_start = inner if first_start is None else first_start
                continue
            kind, name = found
            if first_start is None:
                first_start = start
            if kind == "class":
                qualified = prefix + name
                member_start = visit(block.children, block.open_line + 1, qualified + ".")
                end = block.close_line if member_start is None else max(start, member_start - 1)
                spans.append(_Span("class", qualified, start + 1, end + 1, block.close_line + 1))
            else:
                if prefix and kind == "function":
                    kind = "method"
                spans.append(_Span(kind, prefix + name, start + 1, block.close_line + 1))
        return first_start

    visit(_parse_blocks(masked), 0, "")
    return spans


# --------------------------------------------------------------------------- ruby

_RUBY_DEF = re.compile(r"^(\s*)(def|class|module)\s+([\w:.?!=<>+\-*/\[\]]+)")
_RUBY_ENDLESS = re.compile(r"^\s*def\s+[\w.?!]+\s*(?:\([^)]*\))?\s*=\s*\S")


def _ruby_spans(lines: list[str]) -> list[_Span]:
    spans: list[_Span] = []

    def block_end(i: int, indent: str, hi: int) -> int:
        if re.search(r"\bend\s*$", lines[i]) or _RUBY_ENDLESS.match(lines[i]):
            return i  # one-liner or endless method
        for j in range(i + 1, hi):
            line = lines[j]
            if not line.strip():
                continue
            lead = len(line) - len(line.lstrip())
            if lead == len(indent) and re.match(r"\s*end\b", line):
                return j
            if lead < len(indent):
                return j - 1
        return hi - 1

    def visit(lo: int, hi: int, prefix: str) -> int | None:
        first = None
        i = lo
        while i < hi:
            m = _RUBY_DEF.match(lines[i])
            if not m:
                i += 1
                continue
            indent, keyword, name = m.groups()
            first = i if first is None else first
            end = block_end(i, indent, hi)
            if keyword == "def":
                kind = "method" if prefix else "function"
                spans.append(_Span(kind, prefix + name.removeprefix("self."), i + 1, end + 1))
            else:
                qualified = prefix + name.replace("::", ".")
                member = visit(i + 1, end, qualified + ".")
                header_end = end if member is None else max(i, member - 1)
                spans.append(_Span("class", qualified, i + 1, header_end + 1, end + 1))
            i = end + 1
        return first

    visit(0, len(lines), "")
    return spans


# --------------------------------------------------------------------------- module-level code


_CLOSERS_ONLY = re.compile(r"(?:[}\])]+[;,]?\s*)+|end")


def _is_trivial(line: str, language: Language) -> bool:
    """Blank, comment-only, or just the closing of an enclosing block."""
    s = line.strip()
    if not s or s.startswith((language.line_comment, "/*", "*", "*/")):
        return True
    return language.family != "python" and _CLOSERS_ONLY.fullmatch(s) is not None


def _module_spans(lines: list[str], spans: list[_Span], language: Language) -> list[_Span]:
    """Chunks for top-level code not covered by any definition (imports, constants, scripts)."""
    covered = bytearray(len(lines) + 2)
    for s in spans:
        covered[s.start : s.end + 1] = b"\1" * (s.end - s.start + 1)

    runs, start = [], None
    for ln in range(1, len(lines) + 1):
        if not covered[ln] and start is None:
            start = ln
        elif covered[ln] and start is not None:
            runs.append((start, ln - 1))
            start = None
    if start is not None:
        runs.append((start, len(lines)))

    result = []
    for a, b in runs:
        while a <= b and _is_trivial(lines[a - 1], language):
            a += 1
        while b >= a and _is_trivial(lines[b - 1], language):
            b -= 1
        while a <= b:  # split long runs, preferring to cut at a blank line
            cut = min(b, a + MAX_MODULE_CHUNK_LINES - 1)
            if cut < b:
                blank = next((ln for ln in range(cut, a, -1) if not lines[ln - 1].strip()), None)
                cut = blank or cut
            result.append(_Span("module", "<module>", a, cut))
            a = cut + 1
    return result
