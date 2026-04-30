# FPL-Intel PRO ⚽

## Project Overview
FPL-Intel PRO is a Premier League AI Platform designed to help Fantasy Premier League (FPL) managers make data-driven decisions. It combines machine learning predictions with real-time news retrieval to provide accurate point predictions and actionable advice.

## Architecture
- **ML Pipeline**: Uses Random Forest/XGBoost to predict player points based on historical FPL data.
- **RAG Pipeline**: Leverages LangChain and a vector database (ChromaDB) to retrieve the latest player news and injury updates.
- **NLU Router**: Intelligently classifies user queries to route them to the ML pipeline, the RAG pipeline, or both for a comprehensive response.
- **Orchestrator**: Acts as the central hub, tying NLU, ML, and RAG together.
- **UI**: A Streamlit dashboard for interactive chatting, searching, comparing players, and viewing analytics.

## Setup Steps
1. Clone the repository.
2. Create and activate a virtual environment (optional but recommended).
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Run Instructions
Launch the Streamlit application:
```bash
streamlit run app/main.py
```
Open your browser at `http://localhost:8501`.

## Team Responsibilities
- **ML**: Feature engineering, model training, and point predictions (`ml/`).
- **RAG**: Document ingestion, vector database management, and news generation (`rag/`).
- **NLU**: Intent classification to route user queries correctly (`nlu/`).
- **Data**: Fetching data from the FPL API and scraping news (`data/`, `pipeline/`).
- **UI**: Building the Streamlit dashboard and visual analytics (`app/`).
- **Orchestrator**: Routing logic and integrating all modules (`orchestrator/`).
