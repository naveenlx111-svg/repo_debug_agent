"""Cross-file context for fix prompts: structural links first, then semantic neighbours."""

from __future__ import annotations

import logging

from repo_debug_agent.models import Chunk
from repo_debug_agent.retrieval.symbols import SymbolIndex
from repo_debug_agent.retrieval.vector import VectorIndex, VectorIndexUnavailable

__all__ = ["ContextRetriever", "SymbolIndex", "VectorIndex", "VectorIndexUnavailable"]

log = logging.getLogger(__name__)

MAX_LINES_PER_SNIPPET = 60
MIN_SIMILARITY = 0.25


class ContextRetriever:
    def __init__(self, symbols: SymbolIndex, vectors: VectorIndex | None, budget_chars: int):
        self.symbols = symbols
        self.vectors = vectors
        self.budget_chars = budget_chars

    def related(
        self, chunk: Chunk, query: str, exclude_files: set[str] = frozenset()
    ) -> list[tuple[Chunk, str]]:
        picked = [(c, why) for c, why in self.symbols.related(chunk) if c.file not in exclude_files]
        if self.vectors is None:
            return picked

        seen = {chunk.key} | {c.key for c, _ in picked}
        try:
            hits = self.vectors.search(query, n=8)
        except Exception as e:  # a broken index must not break fixing
            log.warning("semantic search failed: %s", e)
            return picked
        for key, score in hits:
            found = self.symbols.get(key)  # current version of the chunk, not the indexed text
            if (
                found is None
                or key in seen
                or found.file in exclude_files
                or score < MIN_SIMILARITY
            ):
                continue
            seen.add(key)
            picked.append((found, f"semantically similar ({score:.2f})"))
        return picked

    def render(self, chunk: Chunk, query: str, exclude_files: set[str] = frozenset()) -> str:
        """Related code as prompt-ready text, within the character budget."""
        blocks: list[str] = []
        used = 0
        for other, why in self.related(chunk, query, exclude_files):
            lines = other.code.split("\n")
            code = "\n".join(lines[:MAX_LINES_PER_SNIPPET])
            if len(lines) > MAX_LINES_PER_SNIPPET:
                code += f"\n... ({len(lines) - MAX_LINES_PER_SNIPPET} more lines)"
            block = (
                f"# {other.file}:{other.start_line}-{other.end_line} `{other.name}` ({why})\n"
                f"```{other.language}\n{code}\n```"
            )
            if used + len(block) > self.budget_chars:
                break
            blocks.append(block)
            used += len(block)
        return "\n\n".join(blocks)
