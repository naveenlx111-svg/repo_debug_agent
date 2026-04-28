from sentence_transformers import SentenceTransformer
import chromadb
from tqdm import tqdm

from src.chunker import Chunk
from src.config import INDEX_DIR,EMBED_MODEL,CHUNK_BATCH_SIZE


print(f"Loading embedding model: {EMBED_MODEL}")
embedder = SentenceTransformer(EMBED_MODEL)
print("Embedding model loaded.")


def get_collection(reset: bool=False) -> chromadb.Collection:
    """
    Get or create the chromadb collection.
    If reset=True,wipe the existing index and start fresh.
    """
    client = chromadb.PersistentClient(path=str(INDEX_DIR))

    if reset:
        try:
            client.delete_collection("codebase")
            print(" Wiped existing index.")
        except Exception:
            pass #collection might not exist yet
    
    collection = client.get_or_create_collection(
        name="codebase",
        metadata={"hnsw:space":"cosine"}
    )
    return collection

#Embed+store chunks
def index_chunks(chunks: list[Chunk],reset: bool=True)->chromadb.Collection:
    """
    Embed all chunks and store them in ChromaDB.
    Returns the collection so the caller can query it.
    """
    collection = get_collection(reset)

    print(f"\nIndexing {len(chunks)} chunks in batches of {CHUNK_BATCH_SIZE}...")

    for batch_start in tqdm(range(0,len(chunks),CHUNK_BATCH_SIZE)):
        batch = chunks[batch_start:batch_start+CHUNK_BATCH_SIZE]


        #cap at 512 chars because MiniLM has 256 token limit
        texts = [c.code[:512] for c in batch]
        #embed the whole batch at once - much faster than one by one
        vectors = embedder.encode(texts,show_progress_bar=False).tolist()

        #metadata stored alongside for each vector - for filtering + display
        metadatas = [
            {
                "file":       c.file_path,
                "name":       c.name,
                "type":       c.chunk_type,
                "language":   c.language,
                "start_line": c.start_line,
            }
            for c in batch
        ]

        collection.add(
            ids=[c.chunk_id for c in batch],
            embeddings=vectors,
            metadatas=metadatas,
            documents=texts
        )

        print(f"Done! {collection.count()} chunks stored in index.")
        return collection

def index_stats(collection: chromadb.Collection) -> None:
    """
    Print a summary about whats in the indexed collection.
    """
    print(f"\nCollection '{collection.name}' stats:")
    print(f"Total chunks: {collection.count()}")

    #Count by language
    from collections import Counter
    languages = [m["language"] for m in collection.get(include=["metadatas"])["metadatas"]]
    lang_counts = Counter(languages)
    print("Chunks by language:")
    for lang,count in lang_counts.most_common():
        print(f"  {lang}: {count}")

