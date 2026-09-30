"""Run results: a JSON report for machines, Markdown for humans, and a git-apply-able patch."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

from repo_debug_agent.llm import Usage
from repo_debug_agent.models import FixResult, FixStatus
from repo_debug_agent.validation import TestRun

STATUS_ICON = {
    FixStatus.FIXED: "✅",
    FixStatus.DISMISSED: "🚫",
    FixStatus.FAILED: "❌",
    FixStatus.SKIPPED: "⏭️",
    FixStatus.REPORTED: "🔎",
}
VERIFIED_LABEL = {
    "tests": "test suite (improved)",
    "repro": "a model-written reproduction script (failed before, passes after); "
    "check that its assertion is the intended behaviour",
    "no-regressions": "test suite (no regressions; no test covered the bug)",
    "syntax": "syntax + static checks only",
    "none": "not verified (no checker for this language)",
}


@dataclass
class RunReport:
    repo: str
    started_at: str
    duration_s: float
    provider: str
    triage_model: str
    fix_model: str
    files_scanned: int = 0
    files_reviewed: int = 0
    chunks: int = 0
    semantic_index: str = "off"
    issues_found: int = 0
    results: list[FixResult] = field(default_factory=list)
    baseline_tests: TestRun | None = None
    final_tests: TestRun | None = None
    usage: Usage = field(default_factory=Usage)
    applied_files: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    patch_file: str | None = None

    def counts(self) -> Counter[str]:
        return Counter(r.status.value for r in self.results)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["results"] = [r.to_dict() for r in self.results]
        data["summary"] = dict(self.counts())
        for key in ("baseline_tests", "final_tests"):
            run = getattr(self, key)
            if run is not None:
                data[key] = {**asdict(run), "output": run.excerpt(4000), "summary": run.summary()}
        return data


def render_markdown(report: RunReport) -> str:
    counts = report.counts()
    lines = [
        "# Repo Debug Agent report",
        "",
        f"- **Repo:** `{report.repo}`",
        f"- **Started:** {report.started_at} ({report.duration_s:.0f}s)",
        f"- **Models:** triage `{report.triage_model}`, fix `{report.fix_model}` "
        f"({report.provider})",
        f"- **Scanned:** {report.files_scanned} files, {report.files_reviewed} reviewed, "
        f"{report.chunks} chunks; semantic index: {report.semantic_index}",
        f"- **LLM usage:** {report.usage.calls} calls, {report.usage.prompt_tokens:,} prompt + "
        f"{report.usage.completion_tokens:,} completion tokens",
    ]
    if report.baseline_tests:
        final = report.final_tests.summary() if report.final_tests else "n/a"
        lines.append(
            f"- **Tests** (`{report.baseline_tests.command}`): "
            f"before: {report.baseline_tests.summary()}; after: {final}"
        )
    if report.patch_file:
        lines.append(f"- **Patch:** `{report.patch_file}`")
    if report.applied_files:
        lines.append(f"- **Applied to:** {', '.join(f'`{f}`' for f in report.applied_files)}")
    lines += ["", "| Result | Count |", "| --- | ---: |"]
    lines += [
        f"| {status.value} | {counts.get(status.value, 0)} |"
        for status in FixStatus
        if counts.get(status.value)
    ]
    lines.append(
        f"| **issues selected** | **{len(report.results)}** (of {report.issues_found} found) |"
    )

    for warning in report.warnings:
        lines.append(f"\n> ⚠️ {warning}")

    for result in report.results:
        issue = result.issue
        lines += [
            "",
            f"## {STATUS_ICON[result.status]} {issue.id} · {result.status.value} · "
            f"{issue.location}",
            "",
            f"**Severity:** {issue.severity} · **confidence:** {issue.confidence:.2f} · "
            f"**source:** {issue.source} · **category:** {issue.category}",
            "",
            f"**Report:** {issue.description}",
        ]
        if result.explanation:
            label = "Fix" if result.status == FixStatus.FIXED else "Outcome"
            lines += ["", f"**{label}:** {result.explanation}"]
        if result.status == FixStatus.FIXED:
            lines += [
                "",
                f"**Verified by:** {VERIFIED_LABEL.get(result.verified_by, result.verified_by)}",
            ]
        if result.repro_status:
            lines += ["", f"**Reproduction:** {result.repro_status.replace('_', ' ')}"]
        if result.diff:
            lines += ["", "```diff", result.diff.rstrip("\n"), "```"]
        if result.repro_script:
            lines += [
                "",
                "<details><summary>Reproduction script</summary>",
                "",
                "```python",
                result.repro_script.rstrip("\n"),
                "```",
                "",
                "</details>",
            ]
        if result.attempts:
            lines += ["", "<details><summary>Attempts</summary>", ""]
            for attempt in result.attempts:
                detail = " ".join(attempt.detail.split())[:300]
                lines.append(f"{attempt.number}. `{attempt.outcome}` {detail}")
            lines += ["", "</details>"]
    return "\n".join(lines) + "\n"


def write_outputs(
    report: RunReport, patch: str, out_dir: Path, transcript: list[dict] | None = None
) -> dict[str, Path]:
    import json

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {"json": out_dir / "report.json", "markdown": out_dir / "report.md"}
    if transcript:
        paths["transcript"] = out_dir / "transcript.jsonl"
        with paths["transcript"].open("w", encoding="utf-8") as f:
            for record in transcript:
                f.write(json.dumps(record) + "\n")
    if patch:
        paths["patch"] = out_dir / "fixes.patch"
        paths["patch"].write_text(patch, encoding="utf-8")
        report.patch_file = str(paths["patch"])
    paths["json"].write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    paths["markdown"].write_text(render_markdown(report), encoding="utf-8")
    return paths
