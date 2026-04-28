# C4 — Level 1: System Context

**FPL-Intel** sits between a single human user (an FPL manager) and three external services. The user asks football questions in natural language; the system answers with player predictions, news summaries, and recommendations by combining its own data with calls to the upstream services below.

```mermaid
C4Context
    title System Context — FPL-Intel

    Person(user, "FPL Manager", "Asks questions about players, fixtures, transfers, and injuries.")

    System(fplIntel, "FPL-Intel", "Hybrid RAG + ML assistant. Answers FPL questions using a vector store of news/players and an XGBoost points-prediction model.")

    System_Ext(fplApi, "Fantasy Premier League API", "Official FPL endpoints — bootstrap-static and element-summary. Source of player, team, and fixture data.")
    System_Ext(gemini, "Google Gemini Pro", "LLM used by the RAG pipeline to generate natural-language answers from retrieved context.")
    System_Ext(hf, "HuggingFace Hub", "Hosts the all-MiniLM-L6-v2 sentence-transformer used for embeddings.")

    Rel(user, fplIntel, "Asks questions, views predictions and dashboards", "Streamlit UI")
    Rel(fplIntel, fplApi, "Fetches live player, team, fixture data", "HTTPS / JSON")
    Rel(fplIntel, gemini, "Sends prompt + retrieved context, receives answer", "HTTPS / Gemini SDK")
    Rel(fplIntel, hf, "Downloads embedding model on first run", "HTTPS")

    UpdateLayoutConfig($c4ShapeInRow="3", $c4BoundaryInRow="1")
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
