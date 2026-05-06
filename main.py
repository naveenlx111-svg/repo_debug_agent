import argparse
from src.crawler import crawl_repo, summarize_crawl
from src.chunker import chunk_files
from src.indexer import index_chunks
from src.triage import triage_repo
from src.fixer import fix_bug
from src.reporter import generate_report,save_report,print_report

def main(repo_path: str, reset_index: bool = True):
    print(f"\n{'='*50}")
    print(f"REPO DEBUG AGENT")
    print(f"Target: {repo_path}")
    print(f"{'='*50}\n")

    # ── Step 1: Crawl ─────────────────────────────────────────────
    print("Step 1: Crawling repo...")
    files = crawl_repo(repo_path)
    summarize_crawl(files)

    if not files:
        print("No code files found. Exiting.")
        return

    # ── Step 2: Chunk ─────────────────────────────────────────────
    print("\nStep 2: Chunking files...")
    chunks = chunk_files(files)
    print(f"Total chunks: {len(chunks)}")

    # ── Step 3: Index ─────────────────────────────────────────────
    print("\nStep 3: Building vector index...")
    collection = index_chunks(chunks, reset=reset_index)

    # ── Step 4: Triage ────────────────────────────────────────────
    print("\nStep 4: Triaging files for bugs...")
    targets = triage_repo(files)

    if not targets:
        print("\nNo bugs found — repo looks clean!")
        return

    # ── Step 5: Fix ───────────────────────────────────────────────
    print(f"\nStep 5: Fixing {len(targets)} bug(s)...")
    for target in targets:
        fix_bug(target, collection)

    # ── Step 6: Report ────────────────────────────────────────────
    print("\nStep 6: Generating report...")
    report      = generate_report(targets, repo_path)
    output_path = save_report(report)
    print_report(report)
    print(f"\nFull report saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Autonomous codebase debugging agent using RAG and LLMs"
    )
    parser.add_argument(
        "--repo",
        required=True,
        help="Path to the repo folder to debug"
    )
    parser.add_argument(
        "--no-reset",
        action="store_true",
        help="Don't reset the vector index (use existing one)"
    )
    args = parser.parse_args()
    main(args.repo, reset_index=not args.no_reset)
