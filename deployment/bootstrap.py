"""Download missing runtime models and index the published examples once."""

from ollama import Client

from app_logging import log_event
from config import EMBEDDING_MODEL, FILES_DIR, OLLAMA_URL, example_files, model_name
from retrieval import Index


def main():
    client = Client(host=OLLAMA_URL, timeout=600)
    installed = {model.model for model in client.list().models}
    for name in (model_name, EMBEDDING_MODEL):
        if name not in installed:
            print(f"Downloading {name}...", flush=True)  #logging
            for _ in client.pull(name, stream=True):
                pass
            log_event("model_download_completed", model=name)  #logging
    index = Index()
    for source in example_files():
        result = index.ingest(FILES_DIR / source["file"])
        print(f"{result['filename']}: {result['status']}", flush=True)  #logging
    log_event("demo_ready", document_count=len(index.documents()))  #logging


if __name__ == "__main__":
    main()
