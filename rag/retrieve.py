"""
FPL Intel — RAG retriever

Wraps the ChromaDB vector store with:
  * Configurable k (default 6)
  * Optional player-name metadata pre-filter
  * MMR (Maximal Marginal Relevance) retrieval to reduce redundant chunks
"""

import os
import re
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

VECTOR_DIR = os.path.join(os.path.dirname(__file__), "..", "fpl_vector_db")
COLLECTION = "fpl_rag"
EMBED_MODEL = "all-MiniLM-L6-v2"

_embeddings: HuggingFaceEmbeddings | None = None
_vector_db: Chroma | None = None


def _get_db() -> Chroma:
    global _embeddings, _vector_db
    if _vector_db is None:
        _embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
        _vector_db = Chroma(
            persist_directory=VECTOR_DIR,
            embedding_function=_embeddings,
            collection_name=COLLECTION,
        )
    return _vector_db


def _detect_player_name(query: str) -> str | None:
    """
    Heuristic: look for a capitalised word or short phrase that could be a
    player name.  Returns the first candidate or None.

    This is intentionally simple — a false negative just falls back to
    unfiltered search, which is still correct.
    """
    # Strip common FPL question words
    stopwords = {
        "should", "i", "is", "are", "what", "how", "when", "will", "can",
        "the", "a", "an", "about", "for", "to", "his", "her", "their",
        "injury", "news", "update", "points", "score", "form", "fit",
        "play", "playing", "bench", "start", "captain", "available",
        "predict", "worth", "price", "value", "pick", "buy", "sell",
    }
    words = re.findall(r"[A-Z][a-z]+", query)
    candidates = [w for w in words if w.lower() not in stopwords]
    return candidates[0] if candidates else None


def retrieve(query: str, k: int = 6) -> list[Document]:
    """
    Retrieve the top-k relevant documents for *query*.

    Strategy:
      1. If a player name is detected, fetch that player's document directly
         (metadata filter) plus MMR results for broader context.
      2. Otherwise use MMR over the full collection.
    """
    db = _get_db()
    player_name = _detect_player_name(query)
    results: list[Document] = []

    if player_name:
        # Guaranteed player-specific doc
        try:
            player_docs = db.similarity_search(
                query,
                k=1,
                filter={"name": player_name},
            )
            results.extend(player_docs)
        except Exception:
            pass

    # MMR for diversity (fetch up to k, then subtract already-fetched)
    remaining_k = max(1, k - len(results))
    fetch_k = max(remaining_k * 3, 10)  # ensure enough candidates for effective MMR diversity
    try:
        mmr_docs = db.max_marginal_relevance_search(query, k=remaining_k, fetch_k=fetch_k)
        # Avoid duplicates
        seen = {d.page_content for d in results}
        for doc in mmr_docs:
            if doc.page_content not in seen:
                results.append(doc)
                seen.add(doc.page_content)
    except Exception:
        # Fallback to plain similarity search
        fallback = db.similarity_search(query, k=remaining_k)
        seen = {d.page_content for d in results}
        for doc in fallback:
            if doc.page_content not in seen:
                results.append(doc)

    return results[:k]


if __name__ == "__main__":
    query = input("Enter your question: ")
    docs = retrieve(query)
    print(f"\nTop {len(docs)} results:\n")
    for i, doc in enumerate(docs, 1):
        print(f"Result {i} [{doc.metadata.get('type', '?')}]:")
        print(doc.page_content)
        print("-" * 50)
