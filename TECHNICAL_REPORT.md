# FPL-Intel PRO Technical Report

Generated from repository inspection on 2026-05-06.

## 1. Executive Summary

FPL-Intel PRO is a Python/Streamlit application for Fantasy Premier League decision support. It combines:

- A Streamlit web UI with chat, player search, comparison, and analytics tabs.
- A machine learning pipeline that predicts next-gameweek expected points (`xP`) for FPL players.
- A retrieval-augmented generation pipeline that retrieves player/news context from ChromaDB and asks Groq Llama 3.3 for concise FPL advice.
- A keyword-based NLU router that dispatches user queries to ML, RAG, or both.
- Offline ingestion scripts for official FPL API data, public RSS/news feeds, feature generation, model training, and vector-store maintenance.

The repository is compact and mostly script-oriented. It contains runnable artifacts (`fpl_intel.db`, `data/*.csv`, `models/*.pkl`, `reports/*.csv/json`, `imgs/*.png`, `fpl_vector_db/`) alongside source code. Several documentation files under `docs/` describe an older architecture in places, especially references to Gemini and earlier stub modules; the current implementation uses Groq via `langchain_groq` and the orchestrator imports working ML/RAG adapters.

## 2. Repository Layout

| Path | Purpose |
|---|---|
| `app/main.py` | Streamlit UI entry point and deployment bootstrap logic. |
| `app/utils.py` | Empty placeholder. |
| `orchestrator/router.py` | Query dispatcher from intent to ML/RAG handlers. |
| `nlu/predict.py` | Keyword-based intent classifier. |
| `nlu/train.py`, `nlu/intents.json` | Empty placeholders for a future trained NLU model. |
| `ml/features.py` | Feature engineering from SQLite FPL data. |
| `ml/train.py` | Model training, evaluation, visualization, and artifact export. |
| `ml/predict.py` | Runtime and CLI prediction logic, including advice classification. |
| `pipeline/fpl_api.py` | Official FPL API ingestion into SQLite. |
| `pipeline/news_scraper.py` | RSS and official player-news ingestion into SQLite. |
| `pipeline/scheduler.py` | APScheduler-based recurring ingestion and vector update runner. |
| `build_vector_store.py` | Full/incremental Chroma vector store builder. |
| `rag/retrieve.py` | Chroma retriever with MMR and optional player metadata lookup. |
| `rag/generate.py` | Adapter returning orchestrator-compatible RAG response dictionaries. |
| `rag/ingest.py` | Thin wrapper around vector store rebuild/update functions. |
| `rag_system.py` | Main RAG retrieve-and-generate implementation using Groq. |
| `retriever.py` | Backward-compatible wrapper around `rag.retrieve.retrieve`. |
| `evaluation/*.py` | Empty placeholders. |
| `docs/` | Mermaid architecture/dataflow/sequence documentation and generated PNG diagrams. |
| `data/` | Generated feature CSVs and an empty `data/fpl_intel.db` placeholder. |
| `models/` | Pickled model bundles. |
| `reports/` | Generated ML reports and enriched player recommendations. |
| `imgs/` | Generated training/evaluation visualizations. |
| `fpl_vector_db/` | Persistent Chroma vector database. |
| `.devcontainer/devcontainer.json` | Python 3.11 devcontainer that auto-runs Streamlit. |
| `.gitignore` | Ignores `.env`, `data/`, `*.pkl`, `venv/`, `__pycache__/`, and `rag/vectordb/`. |

## 3. Runtime Architecture

### 3.1 Main User Flow

The implemented runtime path is in-process:

1. The user opens the Streamlit app from `app/main.py`.
2. Streamlit loads live FPL bootstrap data directly from `https://fantasy.premierleague.com/api/bootstrap-static/`.
3. The chat tab sends the user message to `orchestrator.router.route`.
4. `route` calls `nlu.predict.classify_intent`.
5. Depending on intent:
   - `ml` calls `ml.predict.predict_points`.
   - `rag` calls `rag.generate.get_news_answer`.
   - any other/both case calls both and returns a combined payload.
6. Streamlit renders either a structured ML prediction, a Markdown RAG answer, or side-by-side combined results.

There is no active HTTP backend in the current code. The docs mention an older/planned `localhost:8000/chat` path, but `app/main.py` uses direct imports.

### 3.2 Streamlit UI

`app/main.py` configures a wide-layout Streamlit app named `FPL-Intel PRO` and defines four tabs:

- Chat: conversational assistant with `st.chat_input`, session-state history, and robust fallback error rendering.
- Search: substring search across player web names and team names, displaying goals, assists, total points, and form progress.
- Compare: side-by-side comparison of two players across goals, assists, points, and form, plus a grouped Plotly bar chart.
- Dashboard: per-player historical charts from the FPL `element-summary/{id}` endpoint, including points, goals/assists, minutes, rolling three-gameweek form, best gameweek, and aggregate summary metrics.

The app also includes `_bootstrap_ml_pipeline`, cached with `st.cache_resource`, which attempts to create missing deployment artifacts:

- If root `fpl_intel.db` is absent or very small, it runs `pipeline.fpl_api.run`.
- If `data/fpl_features.csv` is absent, it runs `ml.features.feature_engineering`.
- If `models/fpl_xgboost.pkl` is absent, it runs `python ml/train.py`.

This makes Streamlit Cloud first-run deployment possible, but it also means first launch can be slow and network-dependent.

## 4. Data Stores and Current Artifacts

### 4.1 SQLite Database

The root `fpl_intel.db` currently contains:

| Table | Current rows |
|---|---:|
| `players` | 826 |
| `teams` | 20 |
| `fixtures` | 380 |
| `news_articles` | 20 |

The current checked-in database does not contain `player_history`, even though `pipeline/fpl_api.py` can create and populate it when its migration runs. Its existing schema is older than the latest ingestion script: for example, current `fixtures` lacks difficulty columns and current `players` lacks newer columns such as expected goals/assists and card/transfer fields. `pipeline.fpl_api._migrate` is designed to add missing columns idempotently, but the checked-in DB reflects a prior state.

### 4.2 Chroma Vector Store

`fpl_vector_db/chroma.sqlite3` contains one collection named `fpl_rag` with 887 embeddings. The vector store uses:

- Collection: `fpl_rag`
- Directory: `./fpl_vector_db`
- Embedding model: `all-MiniLM-L6-v2`
- Document types: `news` chunks and `player` cards

### 4.3 ML Artifacts

The repository includes:

- `models/fpl_xgboost.pkl`
- `models/fpl_randomForest.pkl`
- `reports/fpl_model_report.json`
- `reports/fpl_feature_importance.csv`
- `reports/fpl_players_enriched.csv`
- `imgs/viz1_points_distribution.png` through `imgs/viz9_top_players_rag.png`

There are duplicated report/image artifacts under `ml/reports/` and `ml/imgs/`, but the active training script writes to root `reports/` and `imgs/`.

## 5. Data Ingestion

### 5.1 Official FPL API Pipeline

`pipeline/fpl_api.py` fetches and persists official FPL data:

- `bootstrap-static/` for teams and season-aggregate player data.
- `fixtures/` for schedule, scores, completion state, and fixture difficulty fields.
- `element-summary/{player_id}/` for each player's per-gameweek history.

It uses a shared `requests.Session` with a custom user-agent, three-attempt retry logic, typed coercion helpers, and polite `0.05s` sleeps between player-history calls. Writes use `INSERT OR REPLACE`, so reruns update rows idempotently.

The latest script can maintain these tables:

- `players`
- `teams`
- `fixtures`
- `player_history`

The checked-in database has only the first three plus `news_articles`, so a full ingestion run is needed to align the database with the current DDL.

### 5.2 News Scraper

`pipeline/news_scraper.py` populates `news_articles` from:

- Official FPL player `news` fields.
- BBC Sport Football RSS.
- Sky Sports Football RSS.
- FPL Fixtures / Injuries RSS.
- Planet FPL RSS.
- FPL Review RSS.
- Fantasy Football Scout RSS.

It strips basic HTML, parses RSS/Atom XML, assigns stable synthetic URLs for official FPL player news, and deduplicates via the `url UNIQUE` constraint.

### 5.3 Scheduler

`pipeline/scheduler.py` is designed for long-running refresh:

- FPL API refresh every 6 hours.
- News scrape every 3 hours.
- Vector-store update after FPL refresh and after news scrape when new articles are inserted.

It uses APScheduler, but `apscheduler` is not listed in `requirements.txt`, so the scheduler will fail in a clean environment unless that dependency is installed separately.

## 6. Machine Learning Pipeline

### 6.1 Feature Engineering

`ml/features.py` reads `players`, `fixtures`, and `teams` from root `fpl_intel.db`. It can run globally or for a single `web_name` during inference.

Core transformations:

- Maps FPL position integers to `GKP`, `DEF`, `MID`, `FWD`.
- Converts `now_cost` from tenths to millions.
- Computes availability from `chance_of_playing_next_round >= 75`, defaulting missing values to 100.
- Detects injury/status risk from keywords in `news`.
- Marks loan status and active status.
- Filters to active players with minutes or total points.
- Merges team name, short name, and strength.
- Computes finished gameweek count from completed fixtures.
- Engineers minutes, per-90, per-gameweek, contribution, bonus, clean-sheet, ICT, form, ownership, team, fixture, injury, and one-hot position features.

The target is rolling expected points (`xP`):

```text
xP = pts_per_90 * (avg_minutes_per_gw / 90) * fixture_ease_factor
```

The target is then penalized by injury and availability:

- Injury news reduces `xP` to 30%.
- Unavailable players reduce `xP` to 10%.
- Final `xP` is clamped to 0-15.

The active model feature list has 22 columns:

- `avg_minutes_per_gw`
- `appearance_rate`
- `goals_per_90`
- `assists_per_90`
- `cs_per_90`
- `bonus_per_90`
- `gc_per_90`
- `ict_per_90`
- `form`
- `ownership`
- `price`
- `strength`
- `team_avg_goals_scored`
- `team_avg_goals_conceded`
- `avg_opp_strength`
- `fixture_count`
- `fixture_ease`
- `has_injury_news`
- `pos_DEF`
- `pos_FWD`
- `pos_GKP`
- `pos_MID`

Global runs write:

- `data/fpl_features.csv`
- `data/model_df.csv`

Single-player runs return a DataFrame for runtime prediction.

### 6.2 Training

`ml/train.py` reads `data/fpl_features.csv` and `data/model_df.csv`, then:

1. Selects the 22 feature columns and `xP` target.
2. Fits `SimpleImputer(strategy="median")`.
3. Splits data into 80% train and 20% test with `random_state=42`.
4. Trains:
   - `RandomForestRegressor` in a `Pipeline(StandardScaler, RandomForestRegressor)`.
   - `XGBRegressor` in a `Pipeline(StandardScaler, XGBRegressor)`.
5. Evaluates MAE, RMSE, R2, and 5-fold CV MAE.
6. Selects the lowest-MAE model as `best_name`.
7. Computes feature importance from the best model.
8. Generates nine visualizations.
9. Exports pickle bundles, reports, feature importance, and enriched player recommendation CSV.

Current report metrics:

| Model | MAE | RMSE | R2 | CV MAE |
|---|---:|---:|---:|---:|
| RandomForest | 0.3440 | 0.5108 | 0.8759 | 0.3418 |
| XGBoost | 0.2373 | 0.3623 | 0.9376 | 0.2472 |

Current dataset summary:

- Active players: 524
- Features: 22
- Train size: 419
- Test size: 105
- Best model: XGBoost

Top feature importances:

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | `appearance_rate` | 0.2680 |
| 2 | `has_injury_news` | 0.2580 |
| 3 | `avg_minutes_per_gw` | 0.1713 |
| 4 | `form` | 0.1391 |
| 5 | `bonus_per_90` | 0.0496 |

### 6.3 Enriched Recommendation Output

`reports/fpl_players_enriched.csv` adds:

- `predicted_xP`
- `value_score = predicted_xP / price`
- `rag_score`, a 0-100 weighted composite:
  - predicted points: 40%
  - value: 20%
  - form: 15%
  - fixture ease: 15%
  - availability: 10%

The current top rows include high-ranked players such as Gabriel, N.Williams, Rice, and B.Fernandes, sorted by `rag_score`.

### 6.4 Runtime Prediction

`ml/predict.py` exposes `predict_points(query)`:

1. Loads `models/fpl_xgboost.pkl`.
2. Reads active players from SQLite.
3. Finds the longest `web_name` contained in the user query.
4. Runs `feature_engineering(found)` for that player.
5. Ensures all expected feature columns exist.
6. Applies the stored imputer and model.
7. Clamps negative predictions to zero.
8. Builds advice from predicted xP, position, injury flag, and availability.

Advice thresholds are position-aware:

| Position | Strong Start | Good Start | Bench floor |
|---|---:|---:|---:|
| GKP | 5.0 | 3.5 | 2.0 |
| DEF | 5.5 | 3.5 | 2.0 |
| MID | 6.0 | 4.0 | 2.5 |
| FWD | 6.5 | 4.5 | 2.5 |

If a player cannot be matched, the response asks the user to search by FPL web name.

## 7. RAG Pipeline

### 7.1 Indexing

`build_vector_store.py` builds or updates Chroma from SQLite.

News document construction:

- Reads `id`, `title`, `body`, and `source` from `news_articles`.
- Formats source, title, and body into text.
- Splits text with `RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)`.
- Stores metadata: `type=news`, `article_id`, `source`.

Player document construction:

- Reads player stats from `players`.
- Builds a compact player card with points, cost, goals, assists, clean sheets, xG, xA, cards, saves, BPS, transfers, form, and news.
- Stores metadata: `type=player`, `name=web_name`.

Full rebuild deletes the existing Chroma collection and recreates it. Incremental update:

- Tracks `last_news_id` in SQLite table `_vector_store_state`.
- Adds only new news chunks.
- Refreshes all player documents by deleting old docs by metadata name and reinserting current stats.

### 7.2 Retrieval

`rag/retrieve.py` lazily initializes Chroma and HuggingFace embeddings. Retrieval strategy:

1. Detects a simple capitalized player-name candidate from the query.
2. If found, attempts one metadata-filtered similarity search for `{"name": player_name}`.
3. Runs MMR (`max_marginal_relevance_search`) for the remaining slots.
4. Deduplicates documents by page content.
5. Falls back to plain similarity search if MMR fails.

The player-name detector is intentionally simple and can miss FPL `web_name` formats like `B.Fernandes` or all-lowercase user input.

### 7.3 Generation

`rag_system.py` is the main RAG generator. It:

- Loads `.env`.
- Creates a `ChatGroq` client with:
  - model: `llama-3.3-70b-versatile`
  - temperature: `0.2`
  - max tokens: `300`
  - key: `GROQ_API_KEY`
- Retrieves top context with `retrieve(question, k=5)`.
- Adds the last four chat-history messages when present.
- Sends system instructions plus retrieved context and user question to Groq.

The system instruction forces a concise answer with exactly:

- `Verdict`
- `Reason`
- `Risk`

If Groq fails, the returned answer text starts with `Error connecting to Groq:`.

`rag/generate.py` wraps this into a dictionary expected by the orchestrator/UI.

## 8. Intent Routing

`nlu/predict.py` is deterministic keyword routing:

- RAG keywords include injury, fitness, availability, suspension, latest news, status, recovery, and card-related words.
- ML keywords include prediction, points, score, price/value, transfer, ownership, xG, xA, BPS, bonus, and ICT.
- Both/general keywords include captaincy, start/bench, differential, form, fixture, gameweek, recommendation, advice, and best.

Decision logic:

- RAG hit only: `rag`
- ML hit only: `ml`
- both hit, neither hit, or ambiguous: `both`

This gives low latency and no model-loading overhead, but it is brittle for wording variations and entity extraction.

## 9. Dependencies and Environment

`runtime.txt` pins Python 3.11.

`requirements.txt` includes major dependency groups:

- UI/data: `streamlit`, `pandas`, `numpy>=2.0.0`, `scipy>=1.13.0`, `plotly`, `matplotlib`, `python-dotenv`, `watchdog`
- ML: `xgboost`, `scikit-learn`, `joblib`
- RAG/LLM: `langchain`, `langchain-community`, `langchain-core`, `langchain-groq`, `langchain-huggingface`, `langchain-chroma`, `chromadb`, `sentence-transformers`, `pydantic`, `pydantic-settings`, `google-genai`
- Utilities: `requests`, `beautifulsoup4`, `tqdm`, `pyyaml`, `torchvision`, `protobuf~=3.20.0`

Important dependency issues:

- `apscheduler` is used but missing from `requirements.txt`.
- `langchain_text_splitters` is imported directly. It may arrive transitively in some LangChain installs, but explicit inclusion would be safer.
- `requirements.txt` has no trailing newline after `protobuf~=3.20.0`. This is harmless for pip, but adding the newline would avoid confusing concatenated terminal output with `runtime.txt`.
- `google-genai` remains in requirements and docs, but current RAG generation uses Groq.
- `.env` exists locally and is ignored by git. It should contain `GROQ_API_KEY`; its contents were not included in this report.

The devcontainer uses a Python 3.11 Bookworm image, installs `requirements.txt`, installs Streamlit explicitly, opens `README.md` and `app/main.py`, forwards port `8501`, and starts `streamlit run app/main.py`.

## 10. Documentation State

The repository has substantial Mermaid docs under `docs/`:

- C4 context, container, and component views.
- ML ingestion dataflow.
- RAG indexing dataflow.
- Request lifecycle sequence.
- RAG chat sequence.
- ML prediction sequence.
- Offline training sequence.
- Rendered PNG diagrams in `docs/images/`.

However, several docs are stale relative to source code:

- Some docs still say RAG uses Google Gemini; current `rag_system.py` uses Groq Llama 3.3 through `langchain_groq`.
- Some docs describe `rag/generate.py` as a stub; it is now a working adapter around `rag_system.generate_answer`.
- Some docs describe `ml/predict.py` as CLI-only with no `predict_points`; it now exposes `predict_points`.
- Some docs say `pipeline/fpl_api.py`, `pipeline/news_scraper.py`, and `pipeline/scheduler.py` are empty or manual placeholders; they now contain real implementations.
- Some docs mention 28 features and older metrics; current code/report uses 22 features and much lower xP-scale errors.

The docs are still useful as architecture scaffolding, but they should be regenerated or edited to avoid misleading maintainers.

## 11. Quality, Reliability, and Security Observations

### Strengths

- Clear module separation between UI, routing, ML, RAG, ingestion, and indexing.
- Runtime prediction uses the same feature engineering path as training for single-player inference.
- Pickle bundles include model, imputer, feature column order, model name, and metrics.
- The Streamlit app can bootstrap missing ML artifacts on first deployment.
- RAG indexing has both full rebuild and incremental update paths.
- News ingestion deduplicates by URL.
- FPL API calls include retry logic and modest rate limiting.
- RAG answers are constrained to a concise action-oriented format.

### Main Risks

- The model target is engineered from aggregate season/current fixture data, not actual future gameweek labels. Reported performance measures how well models reproduce the engineered `xP` formula, not necessarily real future predictive accuracy.
- The root SQLite database is not fully migrated to the latest DDL, so some pipeline queries may fail or produce less rich player documents until ingestion is rerun.
- `build_vector_store.py` expects newer player columns (`xg_scored`, `xa`, cards, saves, BPS, transfers). The checked-in `players` table currently lacks those columns, so vector rebuild can fail unless migration runs first.
- `pipeline.scheduler` imports missing dependency `apscheduler`.
- Streamlit first-run bootstrap can be expensive on hosted deployments because it may fetch all player histories, train models, and download embedding/model dependencies.
- Intent classification is keyword-only and can misroute natural phrasing.
- Player matching in `predict_points` relies on exact `web_name` substring matching.
- RAG player detection only looks for capitalized words and is not aligned with FPL `web_name` matching.
- The RAG prompt says to use only context, but returned answers do not cite retrieved documents or sources.
- Import-time creation of the Groq client means missing or invalid `GROQ_API_KEY` surfaces at runtime answer generation rather than app startup.
- Empty placeholder files (`evaluation/*.py`, `nlu/train.py`, `nlu/intents.json`, `app/utils.py`) make repo completeness look better than it is.

### Security and Operations

- `.env` is ignored by git, which is correct for secrets.
- Pickle model files are loaded directly. This is acceptable for trusted local artifacts, but unsafe for untrusted model files.
- The app calls public APIs and RSS feeds at runtime/bootstrap; production deployments need network availability and should handle upstream outages.
- No authentication or authorization layer exists. This is fine for a public Streamlit demo, but not for a private/team decision-support service with user data.

## 12. Testing and Evaluation

There are no implemented test files in the repository. `evaluation/ml_metrics.py` and `evaluation/rag_eval.py` are empty.

Current evaluation is generated by `ml/train.py`:

- Train/test split metrics.
- 5-fold CV MAE.
- Feature importance.
- Static visualizations.
- High/low xP confusion matrix derived from regression output.

Recommended test coverage:

- Unit tests for `classify_intent`.
- Unit tests for `classify_advice`.
- Unit tests for player matching and not-found behavior in `predict_points`.
- Feature-engineering tests using a small synthetic SQLite database.
- Migration/schema tests for `pipeline.fpl_api`.
- Vector-store document builder tests for old/new database schemas.
- RAG retriever tests with a tiny temporary Chroma collection.
- Smoke test for `route` covering `ml`, `rag`, and `both`.
- Streamlit-independent tests for data-loading and response-shaping logic.

## 13. Recommended Next Steps

1. Update `docs/` to match the current Groq-based implementation, working adapters, scheduler, and current 22-feature ML pipeline.
2. Add missing dependencies, especially `apscheduler`, and explicitly include `langchain-text-splitters` if needed.
3. Verify and fix the `requirements.txt` / `runtime.txt` newline issue.
4. Run `pipeline.fpl_api.run` against the root database to migrate it and add `player_history`.
5. Make `build_vector_store.py` tolerant of older schemas or require a migration precheck with a clear error message.
6. Add a small automated test suite around routing, prediction, feature engineering, ingestion schema, and document building.
7. Improve player/entity matching by using normalized aliases from the `players` table rather than substring/capitalization heuristics.
8. Add RAG source attribution in responses, at least source/title metadata for retrieved news.
9. Separate deployment bootstrap from user-facing app startup where possible, or cache prebuilt artifacts in the deployment environment.
10. Decide whether generated artifacts should be versioned. Current `.gitignore` ignores `data/` and `*.pkl`, but this working tree contains those artifacts.

## 14. How to Run

Install dependencies:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Set secrets:

```bash
GROQ_API_KEY=your_groq_api_key_here
```

Run the app:

```bash
streamlit run app/main.py
```

Refresh source data:

```bash
python -m pipeline.fpl_api
python -m pipeline.news_scraper
```

Regenerate ML artifacts:

```bash
python ml/features.py
python ml/train.py
```

Update vector store:

```bash
python build_vector_store.py --full
python build_vector_store.py
```

Run scheduler, after adding/installing APScheduler:

```bash
python -m pipeline.scheduler
```

## 15. Final Assessment

FPL-Intel PRO is a practical educational ML/RAG application with enough implementation to run as an end-to-end Streamlit demo. Its strongest parts are the artifact-producing ML workflow, the simple in-process orchestration, and the Chroma/Groq RAG path. The main engineering work left is not feature breadth; it is hardening: schema consistency, dependency correctness, tests, updated documentation, and more robust entity/intent handling.
