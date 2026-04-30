"""
FPL Intel — NLU intent classifier

Routes user queries to the appropriate pipeline:
  * "rag"  — questions about news, injuries, availability, player updates
  * "ml"   — requests for point predictions, price/value analysis
  * "both" — general FPL questions (captaincy, fixtures, form, differentials)

Primary strategy: keyword matching (fast, no model loading required).
"""

import re

# ---------------------------------------------------------------------------
# Keyword sets
# ---------------------------------------------------------------------------

_RAG_KEYWORDS = {
    "injury", "injured", "injur", "fit", "fitness", "doubt", "doubtful",
    "ill", "illness", "operation", "surgery", "knock", "strain", "hamstring",
    "suspended", "suspension", "ban", "banned", "available", "unavailable",
    "return", "returning", "out", "ruled", "news", "update", "latest",
    "status", "absence", "absent", "recover", "recovery", "muscle",
    "red card", "yellow card",
}

_ML_KEYWORDS = {
    "predict", "prediction", "points", "score", "worth", "price", "value",
    "cost", "pick", "buy", "sell", "transfer", "invest", "ownership",
    "selected", "xg", "expected goals", "expected assists", "xa",
    "bps", "bonus", "ict",
}

_BOTH_KEYWORDS = {
    "captain", "captaincy", "vice", "start", "bench", "differential",
    "form", "fixture", "fdr", "difficulty", "gameweek", "gw",
    "should i", "recommend", "advice", "who", "best",
}

# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------


def classify_intent(query: str) -> str:
    """
    Classify a user query into one of: 'rag', 'ml', or 'both'.

    Returns
    -------
    str — 'rag', 'ml', or 'both'
    """
    q_lower = query.lower()

    rag_hit = any(kw in q_lower for kw in _RAG_KEYWORDS)
    ml_hit = any(kw in q_lower for kw in _ML_KEYWORDS)

    if rag_hit and not ml_hit:
        return "rag"
    if ml_hit and not rag_hit:
        return "ml"
    # Explicit "both" keywords, or ambiguous (both/neither matched) → both
    return "both"
