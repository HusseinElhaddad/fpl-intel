from ml.predict import predict_points
from rag.generate import get_news_answer
from nlu.predict import classify_intent

def route(query):
    intent = classify_intent(query)

    if intent == "ml":
        return predict_points(query)
    elif intent == "rag":
        return get_news_answer(query)
    else:
        return {
            "ml": predict_points(query),
            "rag": get_news_answer(query)
        }
