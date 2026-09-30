"""Terminal output. Kept separate so the pipeline can run silently (tests, library use)."""

from __future__ import annotations

import io
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

from rich.console import Console
from rich.markup import escape
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table

from repo_debug_agent.models import FixResult, FixStatus, Issue
from repo_debug_agent.report import STATUS_ICON, VERIFIED_LABEL, RunReport

_STATUS_STYLE = {
    FixStatus.FIXED: "green",
    FixStatus.DISMISSED: "yellow",
    FixStatus.FAILED: "red",
    FixStatus.SKIPPED: "dim",
    FixStatus.REPORTED: "cyan",
}
_SEVERITY_STYLE = {"critical": "bold red", "high": "red", "medium": "yellow", "low": "dim"}


class UI:
    def __init__(self, console: Console | None = None, verbose: bool = False):
        self.console = console or Console(highlight=False)
        self.verbose = verbose
        self._stage_started: float | None = None

    @classmethod
    def silent(cls) -> UI:
        return cls(Console(file=io.StringIO(), highlight=False))

    def header(self, repo: Path, provider: str, triage: str, fix: str) -> None:
        self.console.rule("[bold]Repo Debug Agent")
        self.console.print(f"[bold]Repo:[/]   {escape(str(repo))}")
        model = fix if triage == fix else f"triage {triage} · fix {fix}"
        self.console.print(f"[bold]Model:[/]  {escape(model)} [dim]({provider})[/]")

    def stage(self, title: str) -> None:
        self._end_stage()
        self.console.print(f"\n[bold cyan]▸ {title}[/]")
        self._stage_started = time.monotonic()

    def _end_stage(self) -> None:
        if self._stage_started is not None:
            self.detail(f"({time.monotonic() - self._stage_started:.1f}s)")
            self._stage_started = None

    # Messages are escaped: they often contain model output or paths, where "[...]" is
    # ordinary text (e.g. `ordered[mid - 1]`), not Rich markup.
    def info(self, message: str) -> None:
        self.console.print(f"  {escape(message)}")

    def detail(self, message: str) -> None:
        if self.verbose:
            self.console.print(f"  [dim]{escape(message)}[/]")

    def warn(self, message: str) -> None:
        self.console.print(f"  [yellow]⚠ {escape(message)}[/]")

    @contextmanager
    def progress(self, description: str, total: int) -> Iterator[Callable[[], None]]:
        with Progress(
            TextColumn("  {task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=self.console,
            transient=True,
        ) as progress:
            task = progress.add_task(description, total=total)
            yield lambda: progress.advance(task)

    def issues(self, issues: list[Issue], dropped: int) -> None:
        if not issues:
            self.console.print("  [green]No issues above the confidence threshold.[/]")
            return
        table = Table(show_header=True, header_style="bold", box=None, padding=(0, 1))
        table.add_column("ID")
        table.add_column("Severity")
        table.add_column("Conf", justify="right")
        table.add_column("Location")
        table.add_column("Issue", overflow="fold")
        for issue in issues:
            summary = issue.description.split("\n")[0]
            table.add_row(
                issue.id,
                f"[{_SEVERITY_STYLE.get(issue.severity, '')}]{issue.severity}[/]",
                f"{issue.confidence:.2f}",
                escape(issue.location),
                escape(summary if len(summary) < 160 else summary[:157] + "..."),
            )
        self.console.print(table)
        if dropped:
            self.detail(f"{dropped} lower-confidence or overflow issue(s) not shown")

    def fix_started(self, issue: Issue) -> None:
        self.console.print(f"  [bold]{issue.id}[/] {escape(issue.location)}", end=" ")

    def fix_finished(self, result: FixResult) -> None:
        style = _STATUS_STYLE[result.status]
        tries = len(result.attempts)
        suffix = f" after {tries} attempts" if tries > 1 else ""
        self.console.print(
            f"[{style}]{STATUS_ICON[result.status]} {result.status.value}{suffix}[/]"
        )
        if result.explanation:
            self.console.print(f"      [dim]{escape(result.explanation)}[/]")
        if result.repro_status:
            self.detail(f"reproduction: {result.repro_status.replace('_', ' ')}")
        if result.status == FixStatus.FIXED:
            self.detail(
                f"verified by: {VERIFIED_LABEL.get(result.verified_by, result.verified_by)}"
            )

    def summary(self, report: RunReport, paths: dict[str, Path], applied: bool) -> None:
        self._end_stage()
        self.console.rule("[bold]Summary")
        counts = report.counts()
        parts = [
            f"[{_STATUS_STYLE[s]}]{counts[s.value]} {s.value}[/]"
            for s in FixStatus
            if counts.get(s.value)
        ]
        self.console.print(
            f"Issues: {', '.join(parts) or 'none'}  [dim]({report.duration_s:.0f}s, "
            f"{report.usage.calls} LLM calls)[/]"
        )
        if report.baseline_tests:
            after = report.final_tests.summary() if report.final_tests else "n/a"
            self.console.print(f"Tests:  before {report.baseline_tests.summary()} → after {after}")
        for warning in report.warnings:
            self.warn(warning)
        self.console.print(f"Report: {escape(str(paths['markdown']))}")
        if "transcript" in paths:
            self.detail(f"LLM transcript: {paths['transcript']}")
        if "patch" in paths:
            self.console.print(f"Patch:  {escape(str(paths['patch']))}")
            if applied and report.applied_files:
                self.console.print(
                    f"[green]Applied to {len(report.applied_files)} file(s) in the repo.[/]"
                )
            elif not applied:
                self.console.print(
                    "[dim]The repo was not modified. Re-run with --apply, or: "
                    f"git -C {escape(report.repo)} apply {escape(str(paths['patch']))}[/]"
                )
