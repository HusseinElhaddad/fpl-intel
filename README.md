# FPL-Intel PRO ⚽

**Live App:** [fpl-intel.streamlit.app](https://fpl-intel.streamlit.app/)

FPL-Intel PRO is a cutting-edge AI platform for Fantasy Premier League (FPL) managers. It combines advanced Machine Learning predictions with a Retrieval-Augmented Generation (RAG) system to provide data-driven insights and real-time news analysis.

## 🚀 Key Features
- **Rolling Expected Points (xP)**: XGBoost model predicts **next-gameweek expected points** per player, not season totals — the metric FPL managers actually need.
- **Smart Advice Engine**: Position-aware advice classification (🟢 Strong Start / ✅ Good Start / 🟡 Bench / 🔴 Avoid) based on xP, injury status, and fixture difficulty.
- **Real-Time News (RAG)**: Instant answers about injuries, transfer news, and press conferences via ChromaDB + Groq/Llama 3.3.
- **Intelligent Routing**: NLU-based intent classification that automatically routes queries to the ML pipeline, RAG pipeline, or both.
- **Interactive Dashboard**: A premium Streamlit UI with player comparisons, prediction charts, and a smart chat assistant.

---

## 🛠 Architecture
The system is built on a modular internal architecture for high performance and low latency:
- **ML Engine** (`ml/`):
  - `features.py` — Feature engineering from FPL API data. Computes per-90 rates, fixture difficulty, and the **xP target** (rolling expected points adjusted by fixture ease and injury status).
  - `train.py` — Trains XGBoost + RandomForest, selects best model, generates 9 visualizations and enriched player reports.
  - `predict.py` — Real-time inference with position-aware advice classification.
- **RAG Engine** (`rag/` + `rag_system.py`): ChromaDB vector store + Groq/Llama 3.3 70B for lightning-fast news retrieval and generation.
- **NLU Router** (`nlu/`): Classifies user queries into `ml`, `news`, or `general` intents.
- **Orchestrator** (`orchestrator/router.py`): The unified interface connecting all specialized modules.
- **UI Layer** (`app/main.py`): The user-facing Streamlit application.

---

## 📊 ML Model Details

### Target Variable: Rolling Expected Points (xP)
Unlike season-total predictions, **xP estimates how many points a player will score in the next gameweek**:

```
xP = pts_per_90 × (avg_minutes_per_gw / 90) × fixture_ease_factor
```

- **Fixture ease factor**: Normalized from upcoming opponent strength (easier fixtures → higher xP).
- **Injury penalty**: Players with injury news get xP reduced by 70%; unavailable players by 90%.
- **Range**: 0–15 points per GW (clamped).

### Advice Classification
Advice is position-specific — a 4.0 xP is great for a GKP but mediocre for a FWD:

| Advice | GKP | DEF | MID | FWD |
| :--- | :---: | :---: | :---: | :---: |
| 🟢 Strong Start | ≥5.0 | ≥5.5 | ≥6.0 | ≥6.5 |
| ✅ Good Start | ≥3.5 | ≥3.5 | ≥4.0 | ≥4.5 |
| 🟡 Bench | ≥2.0 | ≥2.0 | ≥2.5 | ≥2.5 |
| 🔴 Avoid | <2.0 | <2.0 | <2.5 | <2.5 |

### Current Performance
- **Best Model**: XGBoost
- **MAE**: 0.237 pts
- **R²**: 0.938

---

## ⚙️ Setup & Installation

### 1. Prerequisites
- Python 3.10+ (Recommended 3.11)
- A **Groq API Key** (Free tier available at [console.groq.com](https://console.groq.com))

### 2. Installation
```bash
# Clone the repository
git clone https://github.com/HusseinElhaddad/fpl-intel.git
cd fpl-intel

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install dependencies (Optimized for NumPy 2.x)
pip install -r requirements.txt
```

### 3. Environment Configuration
Create a `.env` file in the root directory:
```text
GROQ_API_KEY=your_groq_api_key_here
```

---

## 🏃 Running the System

### Launching the Dashboard
```bash
streamlit run app/main.py
```

### Initializing Data (Optional)
If you need to rebuild the databases or train the models from scratch:
```bash
# 1. Generate features and xP target from SQLite DB
python ml/features.py

# 2. Train XGBoost + RandomForest, generate visualizations
python ml/train.py

# 3. Build/Update the News Vector Store
python build_vector_store.py
```

---

## 📄 Deployment
This project is configured for **Streamlit Community Cloud**. 
- **Branch**: `dev`
- **Main file**: `app/main.py`
- **Secrets**: Ensure `GROQ_API_KEY` is added to your Streamlit Cloud secrets (TOML format).

---

## ⚠️ Troubleshooting
- **NumPy Errors**: If you see `No module named 'numpy._core'`, ensure you are using NumPy 2.0+ and SciPy 1.13+. Run `pip install --upgrade numpy scipy` to fix.
- **Groq Errors**: If a model is decommissioned, update `rag_system.py` to use the latest supported model (e.g., `llama-3.3-70b-versatile`).
- **Player Not Found**: The ML model uses FPL `web_name` identifiers (e.g., "M.Salah" not "Salah", "B.Fernandes" not "Fernandes").

---

## 👥 Contributors

This project was developed as part of a **Level 3 Machine Learning Project** by:

| Name | Role |
| :--- | :--- |
| **Hussein Elhaddad** | Project Lead & System Integration |
| **Ahmed Elsadek** | NLU Optimization & Intent Routing |
| **Hossam Salah** | RAG Development & Data Pipeline |
| **Andrew George** | ML Research & Feature Engineering |
| **Elfarouk Omar** | UI Design & Analytics Dashboard |
| **Ahmed Ehab** | Data Scraping & Model Validation |
