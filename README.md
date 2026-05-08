content = """# Repo Debug Agent 🤖

An autonomous codebase debugging agent that uses RAG and LLMs to automatically find and fix bugs across an entire repository.

## How it works

1. **Crawl** — walks the repo and finds all code files
2. **Chunk** — splits files into functions/classes
3. **Index** — embeds chunks into a ChromaDB vector store
4. **Triage** — fast LLM scores every file for bug risk
5. **Fix** — smart LLM retrieves context via RAG and rewrites buggy functions
6. **Report** — saves a full JSON report of all fixes

## Setup

```bash
git clone https://github.com/naveenlx111-svg/repo_debug_agent
cd repo_debug_agent
python -m venv venv
venv\\Scripts\\activate.bat
pip install -r requirements.txt
echo GROQ_API_KEY=your_key_here > .env
```

## Usage

```bash
python main.py --repo path/to/your/repo
```

## Tech stack

- LLM: Groq Llama 3.3 70B (free)
- Embeddings: sentence-transformers (local, free)
- Vector store: ChromaDB (local, free)