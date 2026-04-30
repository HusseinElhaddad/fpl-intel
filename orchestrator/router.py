from ml.predict import predict_points
from rag.generate import get_news_answer
from nlu.predict import classify_intent


def route(query: str, chat_history: list | None = None) -> dict:
    """
    Route *query* to the appropriate pipeline based on NLU intent.

    Parameters
    ----------
    query : str
        The user's question.
    chat_history : list[dict] | None
        Previous conversation messages for multi-turn RAG context.

    Returns
    -------
    dict — one of:
      * ML result dict (keys: player, predicted_points, news, advice, _type="ml")
      * RAG result dict (keys: player, news, _type="rag")
      * Combined dict (keys: "ml", "rag") when intent is "both"
    """
    intent = classify_intent(query)

    if intent == "ml":
        return predict_points(query)
    elif intent == "rag":
        return get_news_answer(query, chat_history=chat_history)
    else:
        return {
            "ml": predict_points(query),
            "rag": get_news_answer(query, chat_history=chat_history),
        }
