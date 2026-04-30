"""
FPL Intel — RAG answer generator

Calls the full RAG pipeline (retrieve + Gemini) and returns the result in the
standard response dict format expected by the orchestrator and Streamlit UI.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag_system import generate_answer


def get_news_answer(query: str, chat_history: list | None = None) -> dict:
    """
    Run the RAG pipeline for *query* and return a response dict.

    Parameters
    ----------
    query : str
        The user's question.
    chat_history : list[dict] | None
        Optional list of previous {"role": ..., "content": ...} messages for
        multi-turn context.

    Returns
    -------
    dict with keys: player, predicted_points, news, advice
    """
    try:
        answer = generate_answer(query, chat_history=chat_history)
        return {
            "player": query,
            "predicted_points": None,   # RAG path — no ML prediction
            "news": answer,
            "advice": None,
            "_type": "rag",
        }
    except Exception as exc:
        return {
            "player": query,
            "predicted_points": None,
            "news": f"RAG pipeline error: {exc}",
            "advice": "Please check your GOOGLE_API_KEY environment variable.",
            "_type": "rag_error",
        }
