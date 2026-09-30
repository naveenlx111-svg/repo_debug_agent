"""Command-line entry point: `repo-debug-agent PATH [options]`."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from repo_debug_agent import __version__
from repo_debug_agent.config import PROVIDERS, AgentSettings, ConfigError, LLMSettings

EXIT_OK, EXIT_ERROR, EXIT_USAGE, EXIT_INTERRUPTED = 0, 1, 2, 130


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="repo-debug-agent",
        description="Find bugs in a repository with an LLM, fix them, and verify the fixes "
        "against syntax checks and your test suite. The repo is never modified unless you "
        "pass --apply; fixes are written to a patch file.",
    )
    p.add_argument("repo", nargs="?", help="path to the repository to debug")
    p.add_argument("--repo", dest="repo_flag", metavar="PATH", help=argparse.SUPPRESS)
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    p.add_argument("-v", "--verbose", action="store_true", help="show more detail and debug logs")

    g = p.add_argument_group("model")
    g.add_argument(
        "--provider",
        choices=sorted(PROVIDERS),
        help="LLM provider (default: groq if GROQ_API_KEY is set, else local)",
    )
    g.add_argument("--base-url", help="OpenAI-compatible endpoint URL (overrides the provider's)")
    g.add_argument("--model", help="model for both review and fixing (local: auto-detected)")
    g.add_argument("--triage-model", help="model for reviewing files")
    g.add_argument("--fix-model", help="model for writing fixes")
    g.add_argument("--timeout", type=float, help="seconds per LLM request (default 300)")

    g = p.add_argument_group("verification")
    g.add_argument(
        "--test-cmd",
        metavar="CMD",
        help='test command run in a sandbox copy to verify each fix, e.g. "pytest -q"',
    )
    g.add_argument("--test-timeout", type=float, default=600.0, help="seconds (default 600)")
    g.add_argument(
        "--no-repro",
        action="store_true",
        help="don't reproduce Python bugs with a model-written script before fixing "
        "(the script runs in the sandbox copy)",
    )
    g.add_argument(
        "--python",
        metavar="PATH",
        help="interpreter for reproduction scripts (default: the repo's .venv, else python3)",
    )

    g = p.add_argument_group("output")
    g.add_argument("--apply", action="store_true", help="write accepted fixes into the repo")
    g.add_argument("--analyze-only", action="store_true", help="report issues without fixing")
    g.add_argument(
        "--out",
        type=Path,
        default=Path("debug_reports"),
        help="directory for reports and patches (default ./debug_reports)",
    )
    g.add_argument("--keep-sandbox", action="store_true", help="don't delete the sandbox copy")

    g = p.add_argument_group("analysis")
    g.add_argument(
        "--include-tests",
        action="store_true",
        help="also review and fix test files (default: tests are treated as the spec)",
    )
    g.add_argument("--max-issues", type=int, default=20, help="max issues to fix (default 20)")
    g.add_argument(
        "--min-confidence",
        type=float,
        default=0.6,
        help="skip issues the reviewer is less sure about (0-1, default 0.6)",
    )
    g.add_argument("--max-attempts", type=int, default=3, help="fix attempts per issue (default 3)")
    g.add_argument(
        "--workers",
        type=int,
        help="parallel file reviews (default: 4 for groq, 1 for local servers)",
    )

    g = p.add_argument_group("retrieval")
    g.add_argument(
        "--no-embeddings",
        action="store_true",
        help="structural retrieval only (skip the ChromaDB semantic index)",
    )
    g.add_argument(
        "--embed-model",
        default="default",
        help='"default" (bundled ONNX MiniLM) or a sentence-transformers model name',
    )
    g.add_argument("--index-dir", type=Path, help="where the vector index lives")
    g.add_argument("--reindex", action="store_true", help="rebuild the vector index from scratch")
    g.add_argument(
        "--no-cache",
        action="store_true",
        help="re-review every file instead of reusing cached reviews of unchanged code",
    )
    return p


def settings_from_args(args: argparse.Namespace) -> AgentSettings:
    repo = args.repo or args.repo_flag
    if not repo:
        raise ConfigError("no repository given (usage: repo-debug-agent PATH)")
    llm = LLMSettings.resolve(
        provider=args.provider,
        base_url=args.base_url,
        model=args.model,
        triage_model=args.triage_model,
        fix_model=args.fix_model,
        timeout=args.timeout,
    )
    extra = {"index_dir": args.index_dir} if args.index_dir else {}
    return AgentSettings(
        repo=Path(repo),
        llm=llm,
        out_dir=args.out.resolve(),
        test_cmd=args.test_cmd,
        test_timeout=args.test_timeout,
        apply=args.apply,
        analyze_only=args.analyze_only,
        keep_sandbox=args.keep_sandbox,
        include_tests=args.include_tests,
        max_issues=args.max_issues,
        min_confidence=args.min_confidence,
        max_attempts=args.max_attempts,
        repro=not args.no_repro,
        python=args.python,
        workers=args.workers,
        use_embeddings=not args.no_embeddings,
        embed_model=args.embed_model,
        reindex=args.reindex,
        review_cache=not args.no_cache,
        **extra,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    from rich.logging import RichHandler

    logging.basicConfig(
        level=logging.WARNING,
        format="%(name)s: %(message)s",
        handlers=[RichHandler(show_time=False, show_path=False)],
    )
    if args.verbose:  # our own debug output only; HTTP client internals stay quiet
        logging.getLogger("repo_debug_agent").setLevel(logging.DEBUG)

    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv(usecwd=True))

    # Imported here so `--help` and argument errors stay instant.
    from rich.markup import escape

    from repo_debug_agent.llm import LLMError
    from repo_debug_agent.pipeline import run
    from repo_debug_agent.ui import UI

    ui = UI(verbose=args.verbose)
    try:
        settings = settings_from_args(args)
        run(settings, ui=ui)
    except ConfigError as e:
        ui.console.print(f"[red]error:[/] {escape(str(e))}")
        return EXIT_USAGE
    except LLMError as e:
        ui.console.print(f"[red]LLM error:[/] {escape(str(e))}")
        return EXIT_ERROR
    except KeyboardInterrupt:
        ui.console.print("\n[yellow]interrupted[/]")
        return EXIT_INTERRUPTED
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
