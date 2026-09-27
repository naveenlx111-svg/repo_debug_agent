"""Data types shared across pipeline stages."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum

SEVERITIES = ("critical", "high", "medium", "low")
SEVERITY_RANK = {name: rank for rank, name in enumerate(SEVERITIES)}


@dataclass(frozen=True)
class Chunk:
    """A contiguous, self-contained piece of a source file (function, method, class header, ...)."""

    file: str  # repo-relative POSIX path
    language: str
    kind: str  # "function" | "method" | "class" | "module"
    name: str  # qualified name, e.g. "Cart.total"; "<module>" for top-level code
    start_line: int  # 1-based, inclusive
    end_line: int  # 1-based, inclusive
    code: str
    # Stable identity across runs and line shifts: "<file>::<name>", plus "#n" for duplicates.
    key: str = ""
    # For class chunks (which cover only the header), the last line of the whole class body.
    body_end: int = 0

    @property
    def short_name(self) -> str:
        return self.name.rsplit(".", 1)[-1]

    @property
    def is_definition(self) -> bool:
        return self.kind in ("function", "method", "class")

    def contains(self, line: int) -> bool:
        return self.start_line <= line <= self.end_line

    @property
    def span(self) -> int:
        return self.end_line - self.start_line + 1


@dataclass
class Issue:
    """A suspected bug, localized to one chunk."""

    file: str
    description: str
    severity: str = "medium"
    confidence: float = 0.5
    line: int | None = None
    symbol: str | None = None  # chunk name the issue was mapped to
    chunk_key: str | None = None
    category: str = "bug"
    source: str = "llm"  # "llm" | "static"
    id: str = ""

    @property
    def rank(self) -> tuple[int, float]:
        return (SEVERITY_RANK.get(self.severity, 2), -self.confidence)

    @property
    def location(self) -> str:
        where = f"{self.file}:{self.line}" if self.line else self.file
        return f"{where} ({self.symbol})" if self.symbol else where


class FixStatus(str, Enum):
    FIXED = "fixed"
    DISMISSED = "dismissed"  # the fixer judged the report a false positive
    FAILED = "failed"
    SKIPPED = "skipped"
    REPORTED = "reported"  # --analyze-only: found, not attempted


@dataclass
class Attempt:
    number: int
    # "applied" | "invalid_reply" | "rejected_syntax" | "rejected_tests" | "llm_error" | "dismissed"
    outcome: str
    detail: str = ""


@dataclass
class FixResult:
    issue: Issue
    status: FixStatus
    explanation: str = ""
    diff: str = ""
    # How the accepted change was verified: "tests" | "repro" | "no-regressions" | "syntax" | "none"
    verified_by: str = "none"
    attempts: list[Attempt] = field(default_factory=list)
    # "reproduced" | "not_reproduced" | "unavailable" | "" (not attempted), and the script used
    repro_status: str = ""
    repro_script: str = ""

    def to_dict(self) -> dict:
        data = asdict(self)
        data["status"] = self.status.value
        return data
