# FPL-Intel


---

## Andrew George

My contribution to this project is the **data pipeline** that feeds the RAG recommendation system with structured, model-ready player data.

  

| File | Role |
|---|---|
| `features.py` | Loads the FPL database, engineers features, exports `fpl_features.csv` |
| `train.py` | Reads the CSV, trains ML models, exports `.pkl` files to `models/` |
| `predict.py` | Loads a model and a CSV, runs predictions, prints results |

---

  

## How to Run

  

**Step 1 — Extract features from the database:**

  

```bash

python features.py

```

  

Outputs: `data/fpl_features.csv`
must be `data/fpl_intel.dp` in the same directory other wise spcify the path after file_name
  

**Step 2 — Train the models:**

  

```bash

python train.py

```

  

Outputs: `models/fpl_xgboost.pkl`, `models/fpl_randomforest.pkl`, etc.

  

**Step 3 — Run predictions:**

  

```bash

python predict.py "data/test_group.csv"

```

  

---

  

## Features Engineered (28 total)

  

From raw FPL stats, the pipeline computes:

  

-  **Per-90 rates** — goals, assists, clean sheets, bonus, ICT per 90 mins

-  **Consistency metrics** — appearance rate, average minutes per gameweek

-  **Fixture difficulty** — upcoming opponent strength, fixture ease score

-  **Team context** — team average goals scored/conceded

-  **Flags** — injury news detected from player news text, availability

  

---

  

## Models & Results

  

Three models are trained and compared. Best performer:

  

|Model          |MAE          |RMSE |R²         |
| ------------- | ----------- | ------------ | ------------ |
| XGBoost ⭐ | ~4.9 pts | ~7.1 pts | 0.974 |
| Random Forest | 6.10 pts | 8.95 pts | 0.959 |

  

---
## Output for the RAG System

 
`fpl_features.csv` — one row per active player, containing all 28 engineered features plus identity columns (`web_name`, `position_name`, `price`, `total_points`). This is the file the RAG retriever indexes to answer questions like _"who is the best value defender with easy fixtures?"_