"""Orchestration: scan -> index -> baseline tests -> review -> fix -> report."""

from __future__ import annotations

import logging
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from repo_debug_agent.analyzer import Analyzer, FileReview, consolidate, select
from repo_debug_agent.cache import ReplyCache
from repo_debug_agent.chunker import chunk_source
from repo_debug_agent.config import AgentSettings, ConfigError, LLMSettings
from repo_debug_agent.crawler import SourceFile, discover
from repo_debug_agent.fixer import Fixer, TestHarness
from repo_debug_agent.llm import (
    ChatModel,
    LLMError,
    LLMUnavailableError,
    OpenAICompatibleLLM,
    RecordingLLM,
)
from repo_debug_agent.models import Chunk, FixResult, FixStatus
from repo_debug_agent.report import RunReport, write_outputs
from repo_debug_agent.repro import Reproducer
from repo_debug_agent.retrieval import (
    ContextRetriever,
    SymbolIndex,
    VectorIndex,
    VectorIndexUnavailable,
)
from repo_debug_agent.ui import UI
from repo_debug_agent.workspace import ApplyConflict, Workspace

log = logging.getLogger(__name__)


def connect(settings: LLMSettings) -> OpenAICompatibleLLM:
    """Create the client, check the endpoint is reachable, and fill in auto-detected models."""
    llm = OpenAICompatibleLLM(settings)
    try:
        available = llm.list_models()
    except LLMUnavailableError as e:
        hint = ""
        if settings.provider == "local":
            hint = (
                "\nIs your local server running? e.g. "
                "llama-server -m model.gguf --port 8080 (or pass --base-url)"
            )
        raise LLMUnavailableError(f"{e}{hint}") from e
    except LLMError:
        available = []  # some servers don't implement /models; that's fine if models are given

    if not settings.triage_model or not settings.fix_model:
        if not available:
            raise ConfigError("no model configured and the server lists none; pass --model")
        settings.triage_model = settings.triage_model or available[0]
        settings.fix_model = settings.fix_model or available[0]
    return llm


def run(settings: AgentSettings, llm: ChatModel | None = None, ui: UI | None = None) -> RunReport:
    ui = ui or UI()
    started = time.monotonic()
    if llm is None:
        llm = connect(settings.llm)
    llm = RecordingLLM(llm)
    triage_model = settings.llm.triage_model or ""
    fix_model = settings.llm.fix_model or ""

    report = RunReport(
        repo=str(settings.repo),
        started_at=datetime.now().isoformat(timespec="seconds"),
        duration_s=0.0,
        provider=settings.llm.provider,
        triage_model=triage_model,
        fix_model=fix_model,
    )
    ui.header(settings.repo, settings.llm.provider, triage_model, fix_model)
    workspace = Workspace(settings.repo)

    try:
        # ------------------------------------------------------------------ scan
        ui.stage("Scanning")
        files = discover(settings.repo, settings.max_file_bytes)
        by_rel: dict[str, SourceFile] = {f.rel: f for f in files}
        sources = {f.rel: workspace.read(f.rel) for f in files}
        chunks: dict[str, list[Chunk]] = {
            f.rel: chunk_source(f.rel, sources[f.rel], f.language) for f in files
        }
        all_chunks = [c for cs in chunks.values() for c in cs]
        report.files_scanned, report.chunks = len(files), len(all_chunks)
        languages = Counter(f.language.name for f in files)
        breakdown = ", ".join(f"{n} {lang}" for lang, n in languages.most_common())
        ui.info(f"{len(files)} source files ({breakdown}), {len(all_chunks)} chunks")
        if not files:
            ui.warn("no supported source files found")
            return _finish(report, workspace, settings, ui, started)

        # ------------------------------------------------------------------ index
        symbols = SymbolIndex(all_chunks)
        vectors = None
        if settings.use_embeddings:
            ui.stage("Indexing")
            try:
                vectors = VectorIndex(settings.index_dir, settings.repo, settings.embed_model)
                embedded, removed = vectors.sync(all_chunks, reset=settings.reindex)
                report.semantic_index = f"on ({vectors.count()} chunks)"
                ui.info(
                    f"semantic index: {vectors.count()} chunks "
                    f"({embedded} embedded, {removed} removed, rest cached)"
                )
            except VectorIndexUnavailable as e:
                vectors = None
                report.semantic_index = "off (unavailable)"
                report.warnings.append(f"semantic retrieval disabled: {e}")
                ui.warn(f"semantic retrieval disabled: {e}")
            except Exception as e:  # corrupt index, model download failure, ...
                vectors = None
                report.semantic_index = "off (error)"
                report.warnings.append(f"semantic retrieval disabled: {e}")
                ui.warn(f"semantic retrieval disabled ({e}); try --reindex")
        retriever = ContextRetriever(symbols, vectors, settings.context_chars)

        # ------------------------------------------------------------------ baseline tests
        tests = None
        if settings.test_cmd and not settings.analyze_only:
            ui.stage("Baseline tests")
            tests = TestHarness(settings.test_cmd, workspace, settings.test_timeout)
            baseline = tests.start()
            report.baseline_tests = baseline
            ui.info(f"`{settings.test_cmd}`: {baseline.summary()}")
            if baseline.exit_code is None or baseline.no_tests:
                why = "timed out" if baseline.exit_code is None else "collected no tests"
                report.warnings.append(f"test command {why}; fixes verified by syntax checks only")
                ui.warn(f"test command {why}; fixes will be verified by syntax checks only")
                tests = None
            elif baseline.failures is None and not baseline.passed:
                ui.detail(
                    "couldn't parse a failure count; any fix that still fails counts as neutral"
                )

        if settings.apply and tests is None and not settings.analyze_only:
            ui.warn("--apply without a working --test-cmd: fixes are only syntax-checked")

        # ------------------------------------------------------------------ review
        targets = [f for f in files if settings.include_tests or not f.is_test]
        report.files_reviewed = len(targets)
        ui.stage(f"Reviewing {len(targets)} files")
        cache = ReplyCache(settings.index_dir.parent / "reviews") if settings.review_cache else None
        analyzer = Analyzer(llm, triage_model, settings.review_max_lines, cache)
        reviews = _review_all(analyzer, targets, sources, chunks, tests, settings.workers, ui)
        if cache and cache.hits:
            ui.info(f"{cache.hits} review(s) reused from cache (--no-cache to redo them)")
        for review in reviews:
            if review.error:
                report.warnings.append(f"{review.file}: {review.error}")

        found = [i for r in reviews for i in r.issues]
        unlocated = sum(1 for i in found if not i.chunk_key)
        issues = consolidate(found)
        report.issues_found = len(issues)
        selected = select(issues, settings.min_confidence, settings.max_issues)
        ui.info(
            f"{len(issues)} suspected issue(s), {len(selected)} selected "
            f"(confidence ≥ {settings.min_confidence}, max {settings.max_issues})"
        )
        if unlocated:
            ui.detail(f"{unlocated} finding(s) couldn't be mapped to code and were dropped")
        ui.issues(selected, len(issues) - len(selected))

        # ------------------------------------------------------------------ fix
        if settings.analyze_only:
            report.results = [FixResult(i, FixStatus.REPORTED) for i in selected]
        elif selected:
            ui.stage(f"Fixing {len(selected)} issue(s)")
            reproducer = None
            if settings.repro and any(by_rel[i.file].language.name == "python" for i in selected):
                reproducer = Reproducer(llm, fix_model, workspace, settings.python)
                ui.detail(f"reproducing Python bugs with {reproducer.python}")
            fixer = Fixer(
                llm,
                fix_model,
                workspace,
                symbols,
                retriever,
                tests,
                settings.max_attempts,
                settings.llm.temperature,
                reproducer,
                settings.keep_unverified,
            )
            for n, issue in enumerate(selected):
                ui.fix_started(issue)
                try:
                    result = fixer.fix(issue, by_rel[issue.file])
                except LLMUnavailableError as e:
                    ui.console.print("[red]LLM unavailable[/]")
                    report.warnings.append(f"stopped early, LLM endpoint unavailable: {e}")
                    report.results += [
                        FixResult(i, FixStatus.SKIPPED, "LLM endpoint became unavailable")
                        for i in selected[n:]
                    ]
                    break
                ui.fix_finished(result)
                report.results.append(result)
        report.final_tests = tests.baseline if tests else None  # ratcheted: the last accepted state

        return _finish(report, workspace, settings, ui, started, usage_from=llm)
    finally:
        if settings.keep_sandbox and workspace.sandbox:
            ui.info(f"sandbox kept at {workspace.sandbox}")
        else:
            workspace.cleanup()


def _review_all(
    analyzer: Analyzer,
    targets: list[SourceFile],
    sources: dict[str, str],
    chunks: dict[str, list[Chunk]],
    tests: TestHarness | None,
    workers: int,
    ui: UI,
) -> list[FileReview]:
    reviews: list[FileReview] = []
    with ui.progress("reviewing", len(targets)) as advance, ThreadPoolExecutor(workers) as pool:
        futures = {
            pool.submit(
                analyzer.review,
                f,
                sources[f.rel],
                chunks[f.rel],
                tests.output_for(f.rel) if tests else None,
            ): f
            for f in targets
        }
        try:
            for future in as_completed(futures):
                reviews.append(future.result())
                advance()
        except BaseException:
            pool.shutdown(wait=False, cancel_futures=True)
            raise
    reviews.sort(key=lambda r: r.file)
    return reviews


def _finish(
    report: RunReport,
    workspace: Workspace,
    settings: AgentSettings,
    ui: UI,
    started: float,
    usage_from: ChatModel | None = None,
) -> RunReport:
    if usage_from is not None:
        report.usage = usage_from.usage
    patch = workspace.diff()
    if settings.apply and patch:
        try:
            report.applied_files = workspace.apply()
        except ApplyConflict as e:
            report.warnings.append(str(e))
    report.duration_s = time.monotonic() - started
    out_dir = settings.out_dir / datetime.now().strftime("%Y%m%d-%H%M%S")
    transcript = usage_from.records if isinstance(usage_from, RecordingLLM) else None
    suggestions = "".join(r.diff for r in report.results if r.status == FixStatus.SUGGESTED)
    paths = write_outputs(report, patch, out_dir, transcript, suggestions)
    ui.summary(report, paths, applied=settings.apply)
    return report
