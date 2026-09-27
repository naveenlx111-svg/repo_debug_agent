"""Registry of supported languages, keyed by file extension."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Language:
    name: str
    extensions: tuple[str, ...]
    # How the chunker finds definitions: "python" (ast), "brace" ({...}), "ruby" (def ... end).
    family: str
    line_comment: str = "//"
    # In C-like languages '...' is a char literal; in JS/PHP it is a string.
    single_quote_is_char: bool = False
    # Backticks delimit (possibly multi-line) strings in JS/TS/Go.
    backtick_strings: bool = False

    @property
    def fence(self) -> str:
        """Markdown code-fence tag."""
        return self.name


LANGUAGES: tuple[Language, ...] = (
    Language("python", (".py",), "python", line_comment="#"),
    Language("javascript", (".js", ".jsx", ".mjs", ".cjs"), "brace", backtick_strings=True),
    Language("typescript", (".ts", ".tsx", ".mts", ".cts"), "brace", backtick_strings=True),
    Language("java", (".java",), "brace", single_quote_is_char=True),
    Language("kotlin", (".kt", ".kts"), "brace", single_quote_is_char=True),
    Language("csharp", (".cs",), "brace", single_quote_is_char=True),
    Language("go", (".go",), "brace", single_quote_is_char=True, backtick_strings=True),
    Language("rust", (".rs",), "brace", single_quote_is_char=True),
    Language("c", (".c", ".h"), "brace", single_quote_is_char=True),
    Language(
        "cpp", (".cpp", ".cc", ".cxx", ".hpp", ".hh", ".hxx"), "brace", single_quote_is_char=True
    ),
    Language("php", (".php",), "brace"),
    Language("ruby", (".rb",), "ruby", line_comment="#"),
)

BY_EXTENSION: dict[str, Language] = {ext: lang for lang in LANGUAGES for ext in lang.extensions}
BY_NAME: dict[str, Language] = {lang.name: lang for lang in LANGUAGES}


def detect_language(path: str | Path) -> Language | None:
    return BY_EXTENSION.get(Path(path).suffix.lower())
