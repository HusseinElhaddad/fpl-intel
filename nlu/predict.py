def classify_intent(query):
    if "injury" in query:
        return "rag"
    return "ml"
