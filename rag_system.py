"""
FPL Intel — RAG system (retrieve + Gemini generate)

Provides ``generate_answer(question, chat_history)`` which:
  1. Retrieves the top-6 most relevant chunks from ChromaDB using MMR.
  2. Builds a concise, FPL-focused prompt.
  3. Calls Gemini 2.0 Flash with the last 4 conversation turns for context.
  4. Returns a plain-text answer in the Verdict / Reason / Risk format.
"""

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from rag.retrieve import retrieve

load_dotenv()

# Groq — Ultra-fast inference with a great free tier
# Model: Llama 3.3 70B Versatile (Supported)
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.2,
    max_tokens=300,
    groq_api_key=os.getenv("GROQ_API_KEY")
)

# System instructions for formatting
_SYSTEM_INSTRUCTION = (
    "You are a concise, expert Fantasy Premier League (FPL) assistant. "
    "Give sharp, actionable advice based ONLY on the provided context. "
    "Format every answer with exactly three labelled lines:\n"
    "**Verdict:** one-line action (Start / Bench / Avoid / Buy / Sell / Hold)\n"
    "**Reason:** 1-2 sentences covering key stats or news\n"
    "**Risk:** 1 sentence on injury, fixture difficulty, or form concern\n"
    "If the context is insufficient, say: 'I don't have enough data to advise on this.'"
)

def generate_answer(question: str, chat_history: list | None = None) -> str:
    """
    Retrieve context and generate an answer using Groq (Llama 3.3).
    """
    # 1. Retrieve context
    docs = retrieve(question, k=5)
    context = "\n".join([f"- {d.page_content}" for d in docs])

    # 2. Build messages for ChatGroq
    messages = [("system", _SYSTEM_INSTRUCTION)]
    
    # Add history
    if chat_history:
        for m in chat_history[-4:]:
            role = "human" if m["role"] == "user" else "assistant"
            messages.append((role, m["content"]))

    # Add current question with context
    messages.append(("human", f"Context (FPL data and news):\n{context}\n\nQuestion: {question}"))

    # 3. Call Groq
    try:
        response = llm.invoke(messages)
        return response.content.strip()
    except Exception as e:
        return f"Error connecting to Groq: {str(e)}"


if __name__ == "__main__":
    history = []
    while True:
        q = input("Ask: ").strip()
        if q.lower() in ("exit", "quit"):
            break
        answer = generate_answer(q, chat_history=history)
        print(f"\n{answer}\n{'-' * 60}")
        history.append({"role": "user", "content": q})
        history.append({"role": "assistant", "content": answer})
