# FPL-Intel — Architecture Documentation

This folder contains the architecture documentation for **FPL-Intel**, a hybrid RAG + ML system that answers Fantasy Premier League questions and predicts player points.

All diagrams are written in **Mermaid** so GitHub renders them inline — open any `.md` file on github.com (or in a Mermaid-aware preview) and the diagrams appear automatically.

## Reading order

Read these top-down — each layer adds detail to the previous one.

| # | Layer | File | What you learn |
|---|-------|------|----------------|
| 1 | C4 — Context | [`architecture/c4-context.md`](architecture/c4-context.md) | Who uses the system and which external services it depends on |
| 2 | C4 — Containers | [`architecture/c4-container.md`](architecture/c4-container.md) | How the system is split into runnable parts (UI, orchestrator, RAG service, ML service, stores) |
| 3 | C4 — Components | [`architecture/c4-component.md`](architecture/c4-component.md) | What lives inside the RAG and Orchestrator containers |
| 4 | Data flow — Ingestion / ML | [`dataflow/ingestion-pipeline.md`](dataflow/ingestion-pipeline.md) | How FPL data becomes trained models |
| 5 | Data flow — RAG indexing | [`dataflow/rag-indexing-pipeline.md`](dataflow/rag-indexing-pipeline.md) | How SQLite rows become a Chroma vector store |
| 6 | Request lifecycle | [`sequence/request-lifecycle.md`](sequence/request-lifecycle.md) | What happens for any user query, end-to-end |
| 7 | Sequence — RAG chat | [`sequence/chat-rag-sequence.md`](sequence/chat-rag-sequence.md) | Detailed flow when intent is "rag" |
| 8 | Sequence — ML prediction | [`sequence/ml-prediction-sequence.md`](sequence/ml-prediction-sequence.md) | Detailed flow when intent is "ml" |
| 9 | Sequence — Offline training | [`sequence/offline-training-sequence.md`](sequence/offline-training-sequence.md) | How a maintainer rebuilds models and the vector store |

## Conventions

- **Solid arrows** → synchronous calls.
- **Dashed arrows** → returned values / responses.
- **Cylinders** → persistent stores (SQLite, Chroma, pickle files).
- **Hexagons / actors** → external services (FPL API, Google Gemini, HuggingFace Hub).

## A note on "designed vs. wired"

Some modules in the repo are scaffolded but not fully wired up yet (for example, `rag/generate.py` returns a stub string, and `app.py` POSTs to a `localhost:8000/chat` endpoint that isn't currently served). The diagrams describe the **intended architecture** as designed, and call out gaps in the per-diagram notes where relevant. This makes the docs useful both as an onboarding map and as a punch list of what to finish wiring.
