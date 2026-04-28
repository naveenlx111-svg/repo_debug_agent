from src.crawler import crawl_repo
from src.chunker import chunk_files
from src.indexer import index_chunks, index_stats

# crawl + chunk
files  = crawl_repo('.')
chunks = chunk_files(files)

# index
collection = index_chunks(chunks, reset=True)
index_stats(collection)