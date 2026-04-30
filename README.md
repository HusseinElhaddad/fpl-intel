# FPL-Intel PRO ⚽

**Live App:** [fpl-intel.streamlit.app](https://fpl-intel.streamlit.app/)

FPL-Intel PRO is a cutting-edge AI platform for Fantasy Premier League (FPL) managers. It combines advanced Machine Learning predictions with a Retrieval-Augmented Generation (RAG) system to provide data-driven insights and real-time news analysis.

## 🚀 Key Features
- **Predictive Analytics**: XGBoost-powered point predictions based on historical performance and fixture difficulty.
- **Real-Time News (RAG)**: Instant answers about injuries, transfer news, and press conferences using a local vector database.
- **Intelligent Routing**: NLU-based intent classification that automatically chooses between ML stats or RAG news.
- **Interactive Dashboard**: A premium Streamlit UI with player comparisons, prediction charts, and a smart chat assistant.

---

## 🛠 Architecture
The system is built on a modular internal architecture for high performance and low latency:
- **ML Engine**: `ml/` - Feature engineering, XGBoost training, and inference.
- **RAG Engine**: `rag/` + `rag_system.py` - ChromaDB vector store + Groq/Llama 3.3 for lightning-fast news retrieval.
- **NLU Router**: `nlu/` - Classifies user queries into `ml`, `news`, or `general` intents.
- **Orchestrator**: `orchestrator/router.py` - The unified interface connecting all specialized modules.
- **UI Layer**: `app/main.py` - The user-facing Streamlit application.

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
# 1. Generate features from SQLite DB
python ml/features.py

# 2. Train the XGBoost model
python ml/train.py

# 3. Build/Update the News Vector Store
python build_vector_store.py
```

---

## 📄 Deployment
This project is configured for **Streamlit Community Cloud**. 
- **Branch**: `dev`
- **Main file**: `app/main.py`
- **Secrets**: Ensure `GROQ_API_KEY` is added to your Streamlit Cloud secrets.

---

## ⚠️ Troubleshooting
- **NumPy Errors**: If you see `No module named 'numpy._core'`, ensure you are using NumPy 2.0+ and SciPy 1.13+. Run `pip install --upgrade numpy scipy` to fix.
- **Groq Errors**: If a model is decommissioned, update `rag_system.py` to use the latest supported model (e.g., `llama-3.3-70b-versatile`).

---

## 👥 Contributors
- **Hussein Elhaddad** (Project Lead & Integration)
- *Developed for Level 3 ML Project*
