# src/retriever.py
import chromadb
from src.indexer import embedder


def retrieve(
    query: str,
    collection: chromadb.Collection,
    n: int = 6,
    language: str = None
) -> list[dict]:
    """
    Search the vector store for code chunks relevant to a query.

    Args:
        query:      natural language or code search query
        collection: the ChromaDB collection to search
        n:          number of results to return
        language:   optional filter e.g. 'py', 'js'

    Returns:
        list of dicts with keys: code, file, name, type, start_line, score
    """
    # embed the query using the same model used for indexing
    query_vector = embedder.encode(query).tolist()

    # optional metadata filter
    where = {"language": language} if language else None

    results = collection.query(
        query_embeddings = [query_vector],
        n_results        = n,
        where            = where,
        include          = ["documents", "metadatas", "distances"]
    )

    # reformat into clean list of dicts
    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        chunks.append({
            "code":       doc,
            "file":       meta["file"],
            "name":       meta["name"],
            "type":       meta["type"],
            "start_line": meta["start_line"],
            "score":      round(max(0.0, 1 - dist), 3)
        })

    # most relevant first
    chunks.sort(key=lambda x: x["score"], reverse=True)
    return chunks


def format_context(chunks: list[dict]) -> str:
    """
    Format retrieved chunks into a string to inject into LLM prompt.
    """
    blocks = []
    for c in chunks:
        blocks.append(
            f"# File: {c['file']} | Function: {c['name']} | Line: {c['start_line']}\n"
            f"{c['code']}"
        )
    return "\n\n---\n\n".join(blocks)