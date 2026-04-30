"""
FPL Intel — RAG system (retrieve + Gemini generate)

Provides ``generate_answer(question, chat_history)`` which:
  1. Retrieves the top-6 most relevant chunks from ChromaDB using MMR.
  2. Builds a concise, FPL-focused prompt.
  3. Calls Gemini 2.0 Flash with the last 4 conversation turns for context.
  4. Returns a plain-text answer in the Verdict / Reason / Risk format.
"""

import os

from google import genai
from google.genai import types

from rag.retrieve import retrieve

# Gemini client (API key from environment)
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# Upgraded model
GEMINI_MODEL = "gemini-2.0-flash"

# System instructions for concise FPL answers
_SYSTEM_INSTRUCTION = (
    "You are a concise, expert Fantasy Premier League (FPL) assistant. "
    "You give sharp, actionable advice to FPL managers based only on the "
    "provided context. Never start your answer with phrases like "
    "'Based on the context provided' or 'According to the information'. "
    "Answer directly and confidently. "
    "Format every answer with exactly three labelled lines:\n"
    "**Verdict:** one-line action (Start / Bench / Avoid / Buy / Sell / Hold)\n"
    "**Reason:** 1-2 sentences covering key stats or news\n"
    "**Risk:** 1 sentence on injury, fixture difficulty, or form concern\n"
    "If the context does not contain enough information to answer, say: "
    "'I don't have enough current data to advise on this — check the latest FPL news.'"
)


def generate_answer(question: str, chat_history: list | None = None) -> str:
    """
    Retrieve relevant FPL context and generate a Gemini answer.

    Parameters
    ----------
    question : str
        The user's question.
    chat_history : list[dict] | None
        Previous conversation turns as [{"role": "user"|"assistant", "content": str}, ...].
        The last 4 pairs (8 messages) are used for context.

    Returns
    -------
    str  — The model's plain-text answer.
    """
    # 1. Retrieve context
    docs = retrieve(question, k=6)
    context = "\n\n---\n\n".join(doc.page_content for doc in docs)

    # 2. Build the prompt
    prompt = (
        f"Context (FPL data and news):\n{context}\n\n"
        f"Question: {question}"
    )

    # 3. Build conversation turns for multi-turn context
    contents = []

    # Include up to the last 4 conversation pairs for context
    if chat_history:
        recent = chat_history[-8:]  # 4 user + 4 assistant = 8 messages
        for msg in recent:
            role = "user" if msg.get("role") == "user" else "model"
            contents.append(types.Content(role=role, parts=[types.Part(text=msg["content"])]))

    # Add the current question with context
    contents.append(types.Content(role="user", parts=[types.Part(text=prompt)]))

    # 4. Call Gemini
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=_SYSTEM_INSTRUCTION,
            temperature=0.2,
            max_output_tokens=300,
        ),
    )

    return response.text


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
