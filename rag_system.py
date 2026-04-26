import os
from google import genai
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


# ✅ Load Gemini client
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# ✅ Load embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# ✅ Load vector DB
vector_db = Chroma(
    persist_directory="./fpl_vector_db",
    embedding_function=embeddings,
    collection_name="fpl_rag"
)


def retrieve(query: str, k: int = 3):
    return vector_db.similarity_search(query, k=k)


def generate_answer(question: str) -> str:
    docs = retrieve(question)

    context = "\n\n".join(doc.page_content for doc in docs)

    prompt = f"""
You are a Fantasy Premier League assistant.

Use ONLY the context below to answer the question.
If the answer is not in the context, say you don't know.

Context:
{context}

Question:
{question}
"""

    response = client.models.generate_content(
   model="gemini-pro",
    contents=prompt,
)

    return response.text


if __name__ == "__main__":
    while True:
        q = input("Ask: ")
        if q.lower() == "exit":
            break

        answer = generate_answer(q)
        print("\nAnswer:\n")
        print(answer)
        print("-" * 60)