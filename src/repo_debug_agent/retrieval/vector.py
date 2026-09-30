"""Semantic retrieval over chunks with a persistent, incremental ChromaDB index.

Each (repo, embedding model) pair gets its own collection. Chunks are keyed by
`Chunk.key` and carry a content hash, so re-runs only embed what changed.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

from repo_debug_agent.models import Chunk

log = logging.getLogger(__name__)

MIN_CHUNK_CHARS = 40  # tiny chunks (one import, a closing brace) are retrieval noise
EMBED_CHARS = 2000  # embedders truncate anyway (MiniLM: 256 tokens); the header comes first
BATCH = 64


class VectorIndexUnavailable(RuntimeError):
    pass


def content_hash(chunk: Chunk) -> str:
    return hashlib.sha1(f"{chunk.file}\n{chunk.name}\n{chunk.code}".encode()).hexdigest()


def embed_text(chunk: Chunk) -> str:
    return f"{chunk.language} {chunk.kind} {chunk.name} in {chunk.file}\n{chunk.code}"[:EMBED_CHARS]


def _embedding_function(model: str) -> Any:
    from chromadb.utils import embedding_functions

    if model == "default":
        # all-MiniLM-L6-v2 via onnxruntime; downloaded once (~80 MB), no torch needed.
        return embedding_functions.DefaultEmbeddingFunction()
    try:
        return embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model)
    except Exception as e:  # missing package or unknown model
        raise VectorIndexUnavailable(
            f"cannot load embedding model {model!r} ({e}); "
            "install sentence-transformers or use --embed-model default"
        ) from e


class VectorIndex:
    def __init__(
        self,
        index_dir: Path,
        repo_root: Path,
        embed_model: str = "default",
        embedding_function: Any = None,
    ):
        try:
            import chromadb
        except ImportError as e:
            raise VectorIndexUnavailable(
                "chromadb is not installed (pip install 'repo-debug-agent[rag]')"
            ) from e

        index_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(index_dir), settings=chromadb.Settings(anonymized_telemetry=False)
        )
        self._ef = embedding_function or _embedding_function(embed_model)
        digest = hashlib.sha1(f"{repo_root.resolve()}|{embed_model}".encode()).hexdigest()
        self.name = f"repo-{digest[:24]}"
        self._collection = self._open()

    def _open(self) -> Any:
        return self._client.get_or_create_collection(
            name=self.name, embedding_function=self._ef, metadata={"hnsw:space": "cosine"}
        )

    def count(self) -> int:
        return self._collection.count()

    def sync(self, chunks: list[Chunk], reset: bool = False) -> tuple[int, int]:
        """Make the index match `chunks`. Returns (embedded, removed)."""
        if reset:
            self._client.delete_collection(self.name)
            self._collection = self._open()

        wanted = {c.key: c for c in chunks if len(c.code.strip()) >= MIN_CHUNK_CHARS}
        existing = self._collection.get(include=["metadatas"])
        stored = {
            id_: (meta or {}).get("hash")
            for id_, meta in zip(existing["ids"], existing["metadatas"] or [], strict=False)
        }

        stale = [id_ for id_ in stored if id_ not in wanted]
        changed = [c for key, c in wanted.items() if stored.get(key) != content_hash(c)]

        for i in range(0, len(stale), BATCH * 16):
            self._collection.delete(ids=stale[i : i + BATCH * 16])
        for i in range(0, len(changed), BATCH):
            batch = changed[i : i + BATCH]
            self._collection.upsert(
                ids=[c.key for c in batch],
                documents=[embed_text(c) for c in batch],
                metadatas=[
                    {
                        "file": c.file,
                        "name": c.name,
                        "kind": c.kind,
                        "language": c.language,
                        "start_line": c.start_line,
                        "hash": content_hash(c),
                    }
                    for c in batch
                ],
            )
        return len(changed), len(stale)

    def search(self, query: str, n: int = 8) -> list[tuple[str, float]]:
        """(chunk key, similarity in [0, 1]) for the `n` nearest chunks."""
        total = self.count()
        if total == 0:
            return []
        result = self._collection.query(
            query_texts=[query], n_results=min(n, total), include=["distances"]
        )
        return [
            (id_, max(0.0, 1.0 - dist))
            for id_, dist in zip(result["ids"][0], result["distances"][0], strict=True)
        ]
