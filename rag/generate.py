def get_news_answer(query):
    return {
        "player": query,
        "predicted_points": 0.0,
        "news": "News retrieved from RAG: Player is fit.",
        "advice": "Consider based on upcoming fixtures"
    }
