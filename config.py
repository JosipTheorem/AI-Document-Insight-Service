"""Application settings, model factory and published example files."""

import json
import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

# Models: the main agent and security guard use the same fast 2B model.
# model_name = "qwen3.5:9b-q4_K_M"
model_name = "qwen3.5:2b-q4_K_M"
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
API_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
EMBEDDING_MODEL = "qwen3-embedding:0.6b"

# SQLite is persistent; Redis is an optional cache (empty URL disables it).
FILES_DIR = PROJECT_ROOT / "data" / "files"
DATA_DIR = Path(os.getenv("RETRIEVAL_DATA_DIR", PROJECT_ROOT / "data" / "runtime"))
DATABASE = DATA_DIR / "index.sqlite3"
DOCUMENT_SESSION = os.getenv("DOCUMENT_SESSION", "demo")
REDIS_URL = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")
EMBEDDING_CACHE_TTL = int(os.getenv("EMBEDDING_CACHE_TTL", str(7 * 24 * 60 * 60)))
CHUNK_WORDS, OVERLAP_WORDS = 220, 35
BATCH_SIZE, CANDIDATES, RRF_K = 8, 30, 60
RRF_WEIGHTS = (1, 2)  # BM25, vector: semantic matching handles Croatian inflections.
PIPELINE_ID = f"simple-v2-articles:{CHUNK_WORDS}:{OVERLAP_WORDS}"
MAX_SESSION_MESSAGES = 10
SESSION_LIMIT_MESSAGE = "Iskoristio si svih 10 poruka u ovoj sesiji. Pokreni novi razgovor za nastavak."


def get_model(purpose="agent"):
    from langchain_ollama import ChatOllama
    from app_logging.callbacks import UsageLogger

    return ChatOllama(model=model_name, base_url=OLLAMA_URL, temperature=0, reasoning=True,
                      num_ctx=8192, num_predict=2048, callbacks=[UsageLogger(purpose, model_name)])  #logging


def example_files():
    """Only these published hashes may be copied into a new conversation."""
    sources = json.loads((FILES_DIR / "sources.json").read_text(encoding="utf-8"))
    return sorted((item for item in sources if item["file"].startswith(("pdf/", "png/"))),
                  key=lambda item: item["file"])
