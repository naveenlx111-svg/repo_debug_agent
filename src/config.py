from pathlib import Path
from dotenv import load_dotenv
import os
load_dotenv()

BASE_DIR = Path(__file__).parent.parent
INDEX_DIR = BASE_DIR/".agent_index"
GROP_API_KEY = os.getenv("GROP_API_KEY")
TRIAGE_MODEL = "llama-3.1-8b-instant"
FIX_MODEL = "llama-3.3-70b-versatile"

EMBED_MODEL = "all-MiniLM-L6-v2"
CODE_EXTENSIONS = {".py",".js",".ts",".java",".cpp",".c",".go",".rb"}

SKIP_DIRS = {
    ".git","__pycache__","venv",".venv","env","node_modules","mypy_cache","dist","build",".agent_index"
}

#Agent Settings

MAX_RETRIES = 3
MAX_BUGS = 20
MAX_FILE_PREVIEW = 3000
MIN_RISK_SCORE = 4.0
CHUNK_BATCH_SIZE = 50

