"""Local document search settings."""

import os
from pathlib import Path

from config.settings import PROJECT_ROOT

DATA_DIR = Path(os.getenv("RETRIEVAL_DATA_DIR", PROJECT_ROOT / "data"))
DATABASE = DATA_DIR / "index.sqlite3"
DOCUMENT_SESSION = os.getenv("DOCUMENT_SESSION", "demo")
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBEDDING_MODEL = "qwen3-embedding:0.6b"
CHUNK_WORDS, OVERLAP_WORDS = 220, 35
BATCH_SIZE, CANDIDATES, RRF_K = 8, 30, 60
PIPELINE_ID = f"simple-v1:{CHUNK_WORDS}:{OVERLAP_WORDS}"
