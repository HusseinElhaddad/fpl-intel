# C4 — Level 3: Components

This level zooms inside the two most interesting containers — **Orchestrator** and **RAG Service** — and names the components that do the work. The ML, indexing, and training containers are intentionally not expanded here: each is a single short script and the data-flow diagrams already show their internals.

## Components inside Orchestrator + RAG

```mermaid
flowchart TB
    ui["<b>Streamlit UI</b><br/><span style='font-size:11px'>[Container]</span>"]:::container

    subgraph orch ["<b>Orchestrator</b> — Container"]
        direction TB
        router["<b>Router</b><br/><span style='font-size:11px'>[function route(query)]</span><br/>Top-level dispatcher.<br/>Calls intent classifier,<br/>forwards to handler."]:::component
        intent["<b>Intent Classifier</b><br/><span style='font-size:11px'>[classify_intent(query)]</span><br/>Keyword-based:<br/>'injury' → rag,<br/>else → ml."]:::component
        mlAdapter["<b>ML Adapter</b><br/><span style='font-size:11px'>[predict_points · planned]</span><br/>Wraps the ML service.<br/>Import target is a stub today."]:::component
        ragAdapter["<b>RAG Adapter</b><br/><span style='font-size:11px'>[get_news_answer · stub]</span><br/>1-line stub today.<br/>Real entry: generate_answer."]:::component
    end

    subgraph rag ["<b>RAG Service</b> — Container"]
        direction TB
        retriever["<b>Retriever</b><br/><span style='font-size:11px'>[retrieve(query, k)]</span><br/>Calls Chroma similarity_search,<br/>returns top-k Documents."]:::component
        promptBuilder["<b>Prompt Builder</b><br/><span style='font-size:11px'>[inline template]</span><br/>Stitches retrieved chunks into<br/>system + context + question."]:::component
        geminiClient["<b>Gemini Client</b><br/><span style='font-size:11px'>[google.genai.Client]</span><br/>HTTPS client for gemini-pro.<br/>Reads GOOGLE_API_KEY from env."]:::component
        embedder["<b>Embedder</b><br/><span style='font-size:11px'>[HuggingFaceEmbeddings]</span><br/>all-MiniLM-L6-v2.<br/>Used for indexing + queries."]:::component
        chromaHandle["<b>Chroma Handle</b><br/><span style='font-size:11px'>[langchain_chroma.Chroma]</span><br/>Persistent client bound to<br/>fpl_vector_db/ + 'fpl_rag'."]:::component
    end

    chromaStore[("<b>Chroma</b><br/>fpl_vector_db/")]:::store
    gemini["<b>Google Gemini Pro</b><br/><span style='font-size:11px'>[External]</span>"]:::external
    hf["<b>HuggingFace Hub</b><br/><span style='font-size:11px'>[External]</span>"]:::external

    ui -->|"route(query)"| router
    router -->|"classify_intent(query)"| intent
    router -->|"intent == 'ml'"| mlAdapter
    router -->|"intent == 'rag'"| ragAdapter

    ragAdapter -.->|"delegate<br/>(intended)"| retriever
    retriever -->|"similarity_search<br/>(query, k=3)"| chromaHandle
    chromaHandle -->|"HNSW search"| chromaStore
    retriever -->|"embeds query"| embedder
    embedder -->|"load model<br/>(first run)"| hf
    retriever -->|"top-k Documents"| promptBuilder
    promptBuilder -->|"prompt string"| geminiClient
    geminiClient -->|"generate_content(prompt)"| gemini

    classDef container fill:#438dd5,stroke:#2e69a8,stroke-width:1.5px,color:#ffffff
    classDef component fill:#85bbf0,stroke:#5d82a8,stroke-width:1px,color:#000000
    classDef store fill:#1168bd,stroke:#0b4884,stroke-width:2px,color:#ffffff
    classDef external fill:#999999,stroke:#6b6b6b,stroke-width:2px,color:#ffffff
    style orch fill:#eaf3fb,stroke:#a9c6e0,stroke-width:1.5px
    style rag fill:#fbf3ea,stroke:#e0c6a9,stroke-width:1.5px
```

## Component-by-component

### Orchestrator

- **Router** — `orchestrator/router.py:5`. Single function `route(query)`. Returns either an ML payload, a RAG string, or both (when intent is "unknown").
- **Intent Classifier** — `nlu/predict.py:1`. Today a 4-line keyword check (`if "injury" in query: return "rag"`). Future work: replace with a real classifier; the contract is just `query -> {"ml", "rag"}`.
- **ML Adapter** — imported as `from ml.predict import predict_points`. The current `ml/predict.py` is a CLI script with no `predict_points` function exported, so this import path is a known wiring gap. The ML pipeline itself works fine when invoked as a script.
- **RAG Adapter** — `rag/generate.py:1`. Returns the literal string `"Sample news response"`. The real RAG entry point is `rag_system.generate_answer` and should be wired here.

### RAG Service

- **Retriever** — `rag_system.py:23` (and an identical standalone copy at `retriever.py:18`). Thin wrapper around `vector_db.similarity_search(query, k=3)`.
- **Prompt Builder** — inline template at `rag_system.py:32-43`. Hard-codes the system instruction "You are a Fantasy Premier League assistant" and a "Use ONLY the context below" guardrail.
- **Gemini Client** — `rag_system.py:8`. Constructed once at import time using `GOOGLE_API_KEY`. Calls `client.models.generate_content(model="gemini-pro", contents=prompt)` per request.
- **Embedder** — `rag_system.py:11`, `build_vector_store.py:69`. The same `all-MiniLM-L6-v2` model is used for both indexing and query-time embeddings (this is critical — different models would produce incompatible vector spaces).
- **Chroma Handle** — `rag_system.py:16-20`. Persistent client bound to `./fpl_vector_db` and collection `fpl_rag`. Uses HNSW under the hood.

## What's deliberately not modelled here

- The **Streamlit tabs** (Search, Compare, Dashboard) are mostly self-contained UI logic that calls the FPL API directly — they're shown as part of the UI container at L2 and don't need a component diagram of their own.
- The **ML training internals** (StandardScaler, SimpleImputer, XGBRegressor, RandomForestRegressor, train/test split) are documented in `dataflow/ingestion-pipeline.md` and `sequence/offline-training-sequence.md` instead — those views answer "what happens during training" better than a static component diagram would.
