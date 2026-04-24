"""
FPL Intel — Load Model & Predict from CSV

Run features.py first to generate the CSV, then run this.
"""

import argparse
import pickle
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

MODEL_PATH = "models/fpl_xgboost.pkl"


parser = argparse.ArgumentParser(description="FPL Intel — predict player points")
parser.add_argument("csv", nargs="?", default="abort!!!", help="Path to .csv")
args = parser.parse_args()

CSV_PATH = args.csv

if CSV_PATH == "abort!!!":
    raise ValueError("You must specify a CSV path")
# ─────────────────────────────────────────────
# 1. LOAD MODEL
# ─────────────────────────────────────────────
print("=" * 62)
print(" Loading model...")
print("=" * 62)

with open(MODEL_PATH, "rb") as f:
    bundle = pickle.load(f)

model        = bundle["model"]
imputer      = bundle["imputer"]
FEATURE_COLS = bundle["feature_cols"]
model_name   = bundle["model_name"]
metrics      = bundle["metrics"]

print(f"  Model      : {model_name}")
print(f"  MAE        : {metrics['MAE']:.3f} pts")
print(f"  RMSE       : {metrics['RMSE']:.3f} pts")
print(f"  R2         : {metrics['R2']:.3f}")
print(f"  Features   : {len(FEATURE_COLS)}")


# ─────────────────────────────────────────────
# 2. LOAD CSV
# ─────────────────────────────────────────────
print("\n" + "=" * 62)
print(f" loading CSV")
print("=" * 62)

df = pd.read_csv(CSV_PATH)
print(f"  Total players in CSV : {len(df)}")

# Check all required feature columns are present
missing = [c for c in FEATURE_COLS if c not in df.columns]
if missing:
    raise ValueError(
        f"CSV is missing {len(missing)} feature column(s): {missing}\n"
        f"Make sure you generated this CSV using features.py from the same pipeline."
    )


# ─────────────────────────────────────────────
# 3. PREDICT
# ─────────────────────────────────────────────
print("\n" + "=" * 62)
print(" Running predictions...")
print("=" * 62)

X = df[FEATURE_COLS].copy()
for col in FEATURE_COLS:
    X[col] = pd.to_numeric(X[col], errors="coerce")

X_imputed   = imputer.transform(X.values)
X_df        = pd.DataFrame(X_imputed, columns=FEATURE_COLS)
predictions = model.predict(X_df)
print(f"  Done — {len(predictions)} predictions made")


# ─────────────────────────────────────────────
# 4. PRINT RESULTS
# ─────────────────────────────────────────────
print("\n" + "=" * 62)
print(f" PREDICTIONS  ({model_name})")
print("=" * 62)
print(f"{'#':<3} {'Player':<18} {'Team':<6} {'Pos':<5} {'Price':>6} {'Actual':>8} {'Predicted':>10} {'Diff':>7}")
print("-" * 62)

for i, (_, row) in enumerate(df.iterrows()):
    pred   = predictions[i]
    actual = row.get("total_points", float("nan"))
    diff   = pred - actual if not np.isnan(actual) else float("nan")
    sign   = "+" if (not np.isnan(diff) and diff >= 0) else ""
    diff_str   = f"{sign}{diff:>6.1f}" if not np.isnan(diff) else "   N/A"
    actual_str = f"{actual:>8.0f}"     if not np.isnan(actual) else "     N/A"

    print(
        f"{i+1:<3} "
        f"{row['web_name']:<18} "
        f"{row['short_name']:<6} "
        f"{row['position_name']:<5} "
        f"£{row['price']:>5.1f} "
        f"{actual_str} "
        f"{pred:>10.1f} "
        f"{diff_str}"
    )

print("-" * 62)
print("\n  Predicted = what the model thinks the player will score")
print("  Actual    = what they actually scored this season")
print("  Diff      = how far off the model was (+ over, - under)")
