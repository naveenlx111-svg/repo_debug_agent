"""Find suspected bugs: static checks plus an LLM review of each file.

Every issue is localized to a chunk (function / method / class / module code),
because a chunk is what the fixer rewrites. Several issues in one chunk are
merged so they are fixed together in a single edit.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from repo_debug_agent import prompts
from repo_debug_agent.cache import ReplyCache
from repo_debug_agent.crawler import SourceFile
from repo_debug_agent.llm import (
    ChatModel,
    LLMError,
    LLMTruncatedError,
    LLMUnavailableError,
    parse_json,
    salvage_array_items,
)
from repo_debug_agent.models import SEVERITIES, SEVERITY_RANK, Chunk, Issue
from repo_debug_agent.validation import python_findings

log = logging.getLogger(__name__)

REVIEW_MAX_TOKENS = 4096
STATIC_CONFIDENCE = 0.95
MIN_SPLIT_LINES = 20
# Part of the review cache key: bump when REVIEW_SCHEMA changes shape.
REVIEW_SCHEMA_ID = "review-v2"


@dataclass
class FileReview:
    file: str
    issues: list[Issue] = field(default_factory=list)
    error: str | None = None


class Analyzer:
    def __init__(
        self,
        llm: ChatModel,
        model: str,
        max_lines: int,
        cache: ReplyCache | None = None,
        passes: int = 1,
    ):
        self.llm = llm
        self.model = model
        self.max_lines = max_lines
        self.cache = cache
        self.passes = passes  # >1: follow-up passes look for what earlier passes missed

    def review(
        self, sf: SourceFile, source: str, chunks: list[Chunk], test_output: str | None = None
    ) -> FileReview:
        """Static findings + LLM findings for one file, each mapped to a chunk."""
        review = FileReview(sf.rel)
        lines = source.split("\n")

        static = python_findings(source, sf.rel) if sf.language.name == "python" else []
        for finding in static:
            review.issues.append(
                Issue(
                    file=sf.rel,
                    description=f"{finding.message} (reported by static analysis)",
                    severity={"SyntaxError": "critical", "SyntaxWarning": "medium"}.get(
                        finding.kind, "high"
                    ),
                    confidence=STATIC_CONFIDENCE,
                    line=finding.line,
                    category="crash",
                    source="static",
                )
            )
        hints = [f"line {f.line}: {f.message}" for f in static]

        for start, end in review_windows(len(lines), chunks, self.max_lines):
            first = len(review.issues)
            self._review_window(sf, lines, start, end, chunks, hints, test_output, review)
            for pass_no in range(2, self.passes + 1):
                prior = list(review.issues[first:])
                self._review_window(
                    sf, lines, start, end, chunks, hints, test_output, review, pass_no, prior
                )

        for issue in review.issues:
            chunk = locate(chunks, issue.line, issue.symbol)
            if chunk is not None:
                issue.chunk_key = chunk.key
                issue.symbol = chunk.name
        return review

    def _review_window(
        self,
        sf: SourceFile,
        lines: list[str],
        start: int,
        end: int,
        chunks: list[Chunk],
        hints: list[str],
        test_output: str | None,
        review: FileReview,
        pass_no: int = 1,
        prior: list[Issue] | None = None,
    ) -> None:
        prompt = prompts.review_prompt(
            sf.rel, sf.language.name, lines, start, end, hints, test_output
        )
        if pass_no > 1:
            prompt += "\n\n" + prompts.review_followup(
                [f"line {i.line or '?'} ({i.symbol or '?'}): {i.description}" for i in prior or []]
            )
        source = "llm" if pass_no == 1 else f"llm-pass{pass_no}"
        messages = [
            {"role": "system", "content": prompts.REVIEW_SYSTEM},
            {"role": "user", "content": prompt},
        ]
        max_tokens = review_max_tokens(end - start + 1)
        key = (
            self.cache.key(self.model, messages, max_tokens=max_tokens, schema=REVIEW_SCHEMA_ID)
            if self.cache
            else ""
        )
        cached = self.cache.get(key) if self.cache else None
        try:
            if cached is not None:
                reply, truncated = cached
            else:
                try:
                    reply = self.llm.complete(
                        messages,
                        model=self.model,
                        max_tokens=max_tokens,
                        json_mode=True,
                        json_schema=prompts.REVIEW_SCHEMA,
                    )
                    truncated = False
                except LLMTruncatedError as e:
                    reply, truncated = e.partial, True

            if truncated:
                if cached is None and self.cache:
                    self.cache.put(key, reply, truncated=True)
                self._handle_truncated(
                    sf, lines, start, end, chunks, hints, test_output, review, reply, pass_no, prior
                )
                return

            data = parse_json(reply)
            if not (isinstance(data, dict) and "issues" in data):
                # A reply that finished but whose JSON broke part-way: keep the complete findings.
                salvaged = salvage_array_items(reply, "issues")
                if salvaged:
                    data = {"issues": salvaged}
            if cached is None and self.cache:
                self.cache.put(key, reply)  # only replies that parsed are worth keeping
            review.issues.extend(parse_review(data, sf.rel, (start, end), source))
        except LLMUnavailableError:
            raise
        except (LLMError, ValueError) as e:
            review.error = f"review of lines {start}-{end} failed: {e}"
            log.warning("%s: %s", sf.rel, review.error)

    def _handle_truncated(
        self,
        sf: SourceFile,
        lines: list[str],
        start: int,
        end: int,
        chunks: list[Chunk],
        hints: list[str],
        test_output: str | None,
        review: FileReview,
        partial: str,
        pass_no: int = 1,
        prior: list[Issue] | None = None,
    ) -> None:
        """A review that hit the token limit, usually because the model started looping after
        its real findings. Keep the complete findings; with none, retry on smaller windows."""
        salvaged = salvage_array_items(partial, "issues")
        if salvaged:
            source = "llm" if pass_no == 1 else f"llm-pass{pass_no}"
            review.issues.extend(parse_review({"issues": salvaged}, sf.rel, (start, end), source))
            review.error = (
                f"review of lines {start}-{end} was cut off; kept its {len(salvaged)} "
                "complete finding(s)"
            )
            return
        if end - start + 1 < 2 * MIN_SPLIT_LINES:
            review.error = f"review of lines {start}-{end} was cut off with no complete findings"
            return
        middle = (start + end) // 2
        cuts = [c.start_line for c in chunks if start + MIN_SPLIT_LINES <= c.start_line <= end]
        mid = min(cuts, key=lambda c: abs(c - middle)) if cuts else middle
        log.info("%s: review of %d-%d truncated; splitting at %d", sf.rel, start, end, mid)
        for lo, hi in ((start, mid - 1), (mid, end)):
            self._review_window(
                sf, lines, lo, hi, chunks, hints, test_output, review, pass_no, prior
            )


def review_max_tokens(window_lines: int) -> int:
    """Room for several findings with traces. Smaller windows get a smaller cap, so a model
    stuck in a repetition loop is cut off after seconds rather than minutes."""
    return min(REVIEW_MAX_TOKENS, 1024 + 12 * window_lines)


def review_windows(total: int, chunks: list[Chunk], max_lines: int) -> list[tuple[int, int]]:
    """Split a file into line windows of at most `max_lines`, cutting at chunk boundaries."""
    if total <= max_lines:
        return [(1, max(total, 1))]
    cuts = sorted({1} | {c.start_line for c in chunks if 1 < c.start_line <= total})
    bounds = cuts + [total + 1]
    windows: list[tuple[int, int]] = []
    current: tuple[int, int] | None = None
    for a, b in ((bounds[i], bounds[i + 1] - 1) for i in range(len(cuts))):
        if b - a + 1 > max_lines:  # one oversized chunk: split it on its own
            if current:
                windows.append(current)
                current = None
            windows.extend((s, min(b, s + max_lines - 1)) for s in range(a, b + 1, max_lines))
        elif current and b - current[0] + 1 <= max_lines:
            current = (current[0], b)
        else:
            if current:
                windows.append(current)
            current = (a, b)
    if current:
        windows.append(current)
    return windows


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    match = re.search(r"\d+", str(value or ""))
    return int(match.group()) if match else None


def _as_confidence(value: Any) -> float:
    try:
        conf = float(value)
    except (TypeError, ValueError):
        conf = {"high": 0.85, "medium": 0.6, "low": 0.3}.get(str(value).strip().lower(), 0.5)
    if conf > 1:  # some models answer on a 0-10 or 0-100 scale
        conf = conf / 10 if conf <= 10 else conf / 100
    return min(max(conf, 0.0), 1.0)


def parse_review(data: Any, rel: str, window: tuple[int, int], source: str = "llm") -> list[Issue]:
    """Turn a model's JSON answer into Issues, tolerating the usual shape variations."""
    items: Any = []
    if isinstance(data, dict):
        items = data.get("issues", data.get("bugs"))
        if items is None and "description" in data:
            items = [data]
    elif isinstance(data, list):
        items = data
    if not isinstance(items, list):
        return []

    issues = []
    for raw in items:
        if not isinstance(raw, dict):
            continue
        if raw.get("confirmed") is False or str(raw.get("confirmed")).lower() == "false":
            continue  # the model's own trace cleared it
        description = str(
            raw.get("description")
            or raw.get("issue")
            or raw.get("problem")
            or raw.get("suspicion")
            or ""
        ).strip()
        if not description or description.lower() in ("none", "n/a", "no bugs"):
            continue
        line = _as_int(raw.get("line"))
        if line is not None and not window[0] <= line <= window[1]:
            line = None
        severity = str(raw.get("severity") or "medium").strip().lower()
        symbol = raw.get("symbol") or raw.get("function")
        issues.append(
            Issue(
                file=rel,
                description=description,
                severity=severity if severity in SEVERITIES else "medium",
                confidence=_as_confidence(raw.get("confidence", 0.5)),
                line=line,
                symbol=str(symbol).strip().strip("`").removesuffix("()") if symbol else None,
                category=str(raw.get("category") or "bug")[:40],
                source=source,
            )
        )
    return issues


def locate(chunks: list[Chunk], line: int | None, symbol: str | None) -> Chunk | None:
    """The chunk an issue refers to, trusting the line number over the (often sloppy) name."""
    containing = sorted(
        (c for c in chunks if line is not None and c.contains(line)),
        key=lambda c: (c.kind not in ("function", "method"), c.span),
    )
    named: list[Chunk] = []
    if symbol:
        short = symbol.rsplit(".", 1)[-1]
        named = [c for c in chunks if c.name == symbol or c.short_name == short]
        for chunk in named:
            if line is not None and chunk.contains(line):
                return chunk
    if containing and containing[0].kind in ("function", "method"):
        return containing[0]
    if named:
        if line is None:
            return named[0]
        return min(named, key=lambda c: abs(c.start_line - line))
    if containing:
        return containing[0]
    if line is not None:  # e.g. a comment line just above a function
        following = [c for c in chunks if 0 < c.start_line - line <= 3]
        if following:
            return min(following, key=lambda c: c.start_line)
    return None


def consolidate(issues: list[Issue]) -> list[Issue]:
    """Merge issues that land in the same chunk; drop ones we couldn't localize."""
    groups: dict[tuple[str, str], list[Issue]] = defaultdict(list)
    for issue in issues:
        if issue.chunk_key:
            groups[(issue.file, issue.chunk_key)].append(issue)

    merged = []
    for group in groups.values():
        group.sort(key=lambda i: i.rank)
        best = group[0]
        if len(group) == 1:
            merged.append(best)
            continue
        seen: set[str] = set()
        parts = []
        for issue in group:
            if issue.description.lower() in seen:
                continue
            seen.add(issue.description.lower())
            where = f"line {issue.line}: " if issue.line else ""
            parts.append(f"{where}{issue.description}")
        merged.append(
            Issue(
                file=best.file,
                description=parts[0]
                if len(parts) == 1
                else "\n".join(f"{n}. {p}" for n, p in enumerate(parts, 1)),
                severity=min((i.severity for i in group), key=lambda s: SEVERITY_RANK[s]),
                confidence=max(i.confidence for i in group),
                line=best.line,
                symbol=best.symbol,
                chunk_key=best.chunk_key,
                category=best.category,
                source="+".join(sorted({i.source for i in group})),
            )
        )
    return merged


def select(issues: list[Issue], min_confidence: float, limit: int) -> list[Issue]:
    kept = [i for i in issues if i.confidence >= min_confidence]
    kept.sort(key=lambda i: (i.rank, i.file, i.line or 0))
    kept = kept[:limit]
    for n, issue in enumerate(kept, 1):
        issue.id = f"B{n}"
    return kept


def relevant_test_output(output: str, rel: str, limit: int = 2500) -> str | None:
    """The parts of a failing test run's output that mention `rel`."""
    name = rel.rsplit("/", 1)[-1]
    lines = output.split("\n")
    hits = [
        i
        for i, text in enumerate(lines)
        if rel in text or re.search(rf"\b{re.escape(name)}\b", text)
    ]
    if not hits:
        return None
    keep: set[int] = set()
    for i in hits:
        keep.update(range(max(0, i - 6), min(len(lines), i + 10)))
    excerpt, last = [], -2
    for i in sorted(keep):
        if i != last + 1:
            excerpt.append("...")
        excerpt.append(lines[i])
        last = i
    text = "\n".join(excerpt)
    return text if len(text) <= limit else text[:limit] + "\n..."
