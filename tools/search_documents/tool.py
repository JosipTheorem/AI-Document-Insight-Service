from functools import lru_cache

from config.retrieval import DOCUMENT_SESSION
from retrieval import Index


@lru_cache(maxsize=1)
def _index():
    return Index()


def search_documents(query: str) -> str:
    """Search indexed PDF/image documents for passages relevant to a question.

    Pass the user's full question, not a keyword list. Preserve its language,
    names and identifiers. Returns source text, filename and page for citations.
    """
    try:
        hits = _index().search(query, session_id=DOCUMENT_SESSION, top_k=3)
        return "\n\n".join(
            f"[{hit['filename']}, str. {hit['page_number']}]\n{hit['text']}"
            for hit in hits
        ) or "No passages found in this document session."
    except Exception as error:
        return f"Document search failed: {error}"
