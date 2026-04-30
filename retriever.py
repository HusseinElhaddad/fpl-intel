"""
FPL Intel — top-level retriever (thin wrapper around rag.retrieve)

Kept for backwards compatibility. New code should import from ``rag.retrieve``.
"""

from rag.retrieve import retrieve


if __name__ == "__main__":
    query = input("Enter your question: ")
    docs = retrieve(query)
    print(f"\nTop {len(docs)} Results:\n")
    for i, doc in enumerate(docs, 1):
        print(f"Result {i} [{doc.metadata.get('type', '?')}]:")
        print(doc.page_content)
        print("-" * 50)
