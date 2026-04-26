from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


# Load embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# Load existing vector DB
vector_db = Chroma(
    persist_directory="./fpl_vector_db",
    embedding_function=embeddings,
    collection_name="fpl_rag"
)


def retrieve(query: str, k: int = 3):
    results = vector_db.similarity_search(query, k=k)
    return results


if __name__ == "__main__":
    query = input("Enter your question: ")
    docs = retrieve(query)

    print("\nTop Results:\n")
    for i, doc in enumerate(docs, 1):
        print(f"Result {i}:")
        print(doc.page_content)
        print("-" * 50)