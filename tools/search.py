import re

from config import DOCUMENT_SESSION
from retrieval import Index

CITATION = re.compile(r"\[([^\]\n]+), str\. ([1-9]\d*)\]")


def make_search_documents(session_id=DOCUMENT_SESSION, index=None):
    """Bind the document scope on the server, outside the model's tool arguments."""
    index = index if index is not None else Index()

    def search_documents(query: str) -> str:
        """Search indexed PDF/image documents for passages relevant to a question.

        Pass the user's full question, not a keyword list. Preserve its language,
        names and identifiers. Returns source text, filename and page for citations.
        """
        try:
            hits = index.search(query, session_id=session_id, top_k=3)
            return "\n\n".join(
                f"[{hit['filename']}, str. {hit['page_number']}]\n{hit['text']}"
                for hit in hits
            ) or "No passages found in this document session."
        except Exception as error:
            return f"Document search failed: {error}"

    return search_documents
