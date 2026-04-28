# C4 — Level 1: System Context

**FPL-Intel** sits between a single human user (an FPL manager) and three external services. The user asks football questions in natural language; the system answers with player predictions, news summaries, and recommendations by combining its own data with calls to the upstream services below.

```mermaid
flowchart LR
    user["<b>FPL Manager</b><br/><span style='font-size:11px'>[Person]</span><br/><br/>Asks questions about players,<br/>fixtures, transfers, and injuries."]:::person

    fplIntel["<b>FPL-Intel</b><br/><span style='font-size:11px'>[Software System]</span><br/><br/>Hybrid RAG + ML assistant.<br/>Answers FPL questions using a vector<br/>store of news/players and an XGBoost<br/>points-prediction model."]:::system

    fplApi["<b>Fantasy Premier League API</b><br/><span style='font-size:11px'>[External System]</span><br/><br/>bootstrap-static<br/>+ element-summary endpoints"]:::external
    gemini["<b>Google Gemini Pro</b><br/><span style='font-size:11px'>[External System]</span><br/><br/>LLM used by the RAG pipeline<br/>to generate answers from<br/>retrieved context"]:::external
    hf["<b>HuggingFace Hub</b><br/><span style='font-size:11px'>[External System]</span><br/><br/>Hosts the all-MiniLM-L6-v2<br/>sentence-transformer used<br/>for embeddings"]:::external

    user -->|"Asks questions, views<br/>predictions and dashboards<br/><i>[Streamlit UI]</i>"| fplIntel
    fplIntel -->|"Fetches live player,<br/>team, fixture data<br/><i>[HTTPS / JSON]</i>"| fplApi
    fplIntel -->|"Sends prompt + retrieved<br/>context, receives answer<br/><i>[HTTPS / Gemini SDK]</i>"| gemini
    fplIntel -->|"Downloads embedding<br/>model on first run<br/><i>[HTTPS]</i>"| hf

    classDef person fill:#08427b,stroke:#052e56,stroke-width:2px,color:#ffffff
    classDef system fill:#1168bd,stroke:#0b4884,stroke-width:2px,color:#ffffff
    classDef external fill:#999999,stroke:#6b6b6b,stroke-width:2px,color:#ffffff
```

## Notes

- The user only ever interacts with FPL-Intel through the Streamlit UI — the external services are never called directly from the browser.
- The FPL API is **read-only** (no auth, no writes). The system caches its responses with `@st.cache_data` (see `app.py:17`).
- Google Gemini is only contacted on the RAG path; the ML path is fully local once models are trained.
- HuggingFace Hub is contacted once per environment to download the embedding weights — afterwards the model lives in the local HuggingFace cache.

## Source pointers

| Boundary | Code |
|----------|------|
| Streamlit UI entry | `app.py`, `app/main.py` |
| FPL API client | `app.py:17-26` (`load_data`, `get_player_history`) |
| Gemini client | `rag_system.py:8` |
| HuggingFace embeddings | `rag_system.py:11`, `build_vector_store.py:69` |
