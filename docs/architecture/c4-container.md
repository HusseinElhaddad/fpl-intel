# C4 — Level 2: Containers

Inside the FPL-Intel system there are two long-running processes (the Streamlit UI and the planned chat backend), four logical services that run as Python modules (Orchestrator, RAG, ML, NLU), and three persistent stores. The diagram below shows them and the calls between them.

```mermaid
flowchart TB
    user["<b>FPL Manager</b><br/><span style='font-size:11px'>[Person]</span>"]:::person

    subgraph fpl ["<b>FPL-Intel</b>"]
        direction TB

        subgraph runtime ["Runtime services (request path)"]
            direction LR
            ui["<b>Streamlit UI</b><br/><span style='font-size:11px'>[Streamlit]</span><br/>Chat · Search · Compare · Dashboard tabs.<br/>Posts queries to the orchestrator."]:::container
            orchestrator["<b>Orchestrator</b><br/><span style='font-size:11px'>[Python module]</span><br/>classify_intent → route to ML or RAG.<br/>Single entry point for any query."]:::container
            rag["<b>RAG Service</b><br/><span style='font-size:11px'>[Python / LangChain]</span><br/>Retrieve, build prompt, call Gemini,<br/>return answer."]:::container
            ml["<b>ML Service</b><br/><span style='font-size:11px'>[Python / XGBoost]</span><br/>Loads pickle bundle, runs predictions,<br/>returns predicted points."]:::container
        end

        subgraph offline ["Offline pipelines (build-time)"]
            direction LR
            features["<b>Feature Pipeline</b><br/><span style='font-size:11px'>[Python / pandas]</span><br/>Reads SQLite, engineers 28 features,<br/>writes CSV."]:::container
            trainer["<b>Model Trainer</b><br/><span style='font-size:11px'>[Python / sklearn]</span><br/>Reads CSV, trains XGBoost + RF,<br/>writes pickle + reports."]:::container
            indexer["<b>Vector Store Builder</b><br/><span style='font-size:11px'>[Python / LangChain]</span><br/>Reads SQLite, chunks news, embeds,<br/>writes Chroma."]:::container
        end

        subgraph stores ["Persistent stores"]
            direction LR
            sqlite[("<b>SQLite</b><br/><span style='font-size:11px'>[fpl_intel.db]</span><br/>players · teams ·<br/>fixtures · news_articles")]:::store
            chroma[("<b>Chroma Vector DB</b><br/><span style='font-size:11px'>[fpl_vector_db/]</span><br/>HNSW index of news<br/>chunks + player cards")]:::store
            models[("<b>Model Artifacts</b><br/><span style='font-size:11px'>[models/*.pkl]</span><br/>Pickled XGBoost +<br/>RandomForest bundles")]:::store
        end
    end

    fplApi["<b>FPL API</b><br/><span style='font-size:11px'>[External]</span>"]:::external
    gemini["<b>Google Gemini Pro</b><br/><span style='font-size:11px'>[External]</span>"]:::external
    hf["<b>HuggingFace Hub</b><br/><span style='font-size:11px'>[External]</span>"]:::external

    user -->|"Uses<br/><i>[HTTPS]</i>"| ui
    ui -->|"Live player/team data<br/><i>[HTTPS / JSON]</i>"| fplApi
    ui -->|"Routes user query<br/><i>[in-process]</i>"| orchestrator
    orchestrator -->|"intent == 'rag'"| rag
    orchestrator -->|"intent == 'ml'"| ml

    rag -->|"similarity_search(k=3)"| chroma
    rag -->|"generate_content(prompt)<br/><i>[HTTPS]</i>"| gemini
    rag -->|"loads embedding<br/>model (first run)"| hf

    ml -->|"loads pickled bundle"| models

    features -->|"SELECT players,<br/>fixtures, teams"| sqlite
    features -->|"fpl_features.csv"| trainer
    trainer -->|"writes pickle bundle<br/>+ reports"| models
    indexer -->|"SELECT news + players"| sqlite
    indexer -->|"writes embeddings"| chroma
    indexer -->|"embeds chunks"| hf

    classDef person fill:#08427b,stroke:#052e56,stroke-width:2px,color:#ffffff
    classDef container fill:#438dd5,stroke:#2e69a8,stroke-width:1.5px,color:#ffffff
    classDef store fill:#1168bd,stroke:#0b4884,stroke-width:2px,color:#ffffff
    classDef external fill:#999999,stroke:#6b6b6b,stroke-width:2px,color:#ffffff
    style fpl fill:#f5f5f5,stroke:#888,stroke-width:2px,stroke-dasharray:4 4
    style runtime fill:#eaf3fb,stroke:#a9c6e0,stroke-width:1px
    style offline fill:#fbf3ea,stroke:#e0c6a9,stroke-width:1px
    style stores fill:#eaf5ea,stroke:#a9d6a9,stroke-width:1px
```

## Containers, in plain English

| Container | What it is | When it runs |
|-----------|-----------|--------------|
| **Streamlit UI** | Single-process Streamlit app; renders four tabs and holds chat session state. | Long-running while a user is browsing. |
| **Orchestrator** | A small `route(query)` function. Calls the NLU intent classifier, then dispatches. | Per request. |
| **RAG Service** | The retrieve-then-generate pipeline. Owns the Chroma handle, embeddings, and Gemini client. | Per RAG request. |
| **ML Service** | The points-prediction inference path. Loads the pickled bundle and runs `predict`. | Per ML request. |
| **Vector Store Builder** | One-shot script that builds `fpl_vector_db/` from SQLite. | Manually, when news/players change. |
| **Feature Pipeline** | One-shot script that builds `data/fpl_features.csv` from SQLite. | Manually, before retraining. |
| **Model Trainer** | One-shot script that consumes the CSV and produces `.pkl` files. | Manually, when the feature set changes. |
| **SQLite** | The single source of truth. All other stores derive from it. | Always present on disk. |
| **Chroma** | Vector store consulted during RAG. | Always present on disk. |
| **Model Artifacts** | Pickled scikit-learn + XGBoost pipeline. | Always present on disk after first training. |

## Wiring gaps (designed vs. implemented)

Two dotted lines in the diagram are aspirational rather than wired up:

- `app.py:9` defines `BACKEND_URL = "http://localhost:8000/chat"` and `app.py:68` POSTs to it, but **no HTTP server is started** by the project today. When the call fails, the chat tab falls back to a hardcoded mock response (`app.py:71-76`). The smaller `app/main.py` *does* call `orchestrator.router.route` directly in-process — that is the path that currently works end-to-end.
- `rag/generate.py:1` is a 1-line stub returning `"Sample news response"`. The real RAG implementation lives at module-level in `rag_system.py`. The orchestrator imports the stub via `from rag.generate import get_news_answer`, so RAG answers from the orchestrator path are currently placeholders.

Both are reflected in the per-sequence diagrams as well.

## Source pointers

| Container | Code |
|-----------|------|
| Streamlit UI | `app.py`, `app/main.py` |
| Orchestrator | `orchestrator/router.py:5` |
| NLU (used by orchestrator) | `nlu/predict.py:1` |
| RAG Service | `rag_system.py:23` (`retrieve`), `rag_system.py:27` (`generate_answer`) |
| ML Service | `ml/predict.py` |
| Vector Store Builder | `build_vector_store.py` |
| Feature Pipeline | `ml/features.py` |
| Model Trainer | `ml/train.py` |
| SQLite | `fpl_intel.db` |
| Chroma | `fpl_vector_db/` |
| Model Artifacts | `models/fpl_xgboost.pkl`, `models/fpl_randomforest.pkl` |
