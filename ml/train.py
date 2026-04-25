#inputs : fpl_features.csv(data)
"""
  outputs:
    models:
      1. XGBoost.pkl
      2. RandomForest.pkl
    images:
      1. viz1_points_distribution.png
      2. viz2_feature_importance.png
      3. viz3_actual_vs_predicted.png
      4. viz4_residuals.png
      5. viz5_model_comparison.png    
      6. viz6_correlation_heatmap.png
      7. viz7_price_vs_predicted.csv
      8. viz8_confusion_matrix.png
      9. viz9_top_players.png
    report:
      fpl_model_report.json
      fpl_feature_importance.csv
      fpl_players_enriched.csv
"""
import pickle
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer

from xgboost import XGBRegressor
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")          # non-interactive backend — saves to file
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

DB_PATH   = "./fpl_intel.db"
Model_DIR   = Path("./models")
IMGS_DIR   = Path("./imgs")
REPORT_DIR   = Path("./reports")
Model_DIR.mkdir(parents=True, exist_ok=True)
IMGS_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# 1. BUILD MODEL FEATURES & TARGET
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("1. loading data")
print("=" * 60)

active = pd.read_csv("data/fpl_features.csv")
print("Data is loaded")

print("\n" + "=" * 60)
print("2. SELECTING FEATURES & TARGET")
print("=" * 60)

FEATURE_COLS = [
    # Core stats
    "minutes", "avg_minutes_per_gw", "appearance_rate",
        "goals_scored", "assists", "clean_sheets", "bonus",
    "goals_per_90", "assists_per_90", "cs_per_90", "bonus_per_90",
    "gc_per_90", "ict_index", "ict_per_90",
    # Form & ownership
    "form", "ownership",
    # Price
    "price",
    # Team context
    "strength",
    "team_avg_goals_scored", "team_avg_goals_conceded",
    # Fixture
    "avg_opp_strength", "fixture_count", "fixture_ease",
    # Injury
    "has_injury_news",
    # Position (one-hot)
    "pos_DEF", "pos_FWD", "pos_GKP", "pos_MID",
]

# Ensure all pos_ cols exist
for col in ["pos_DEF", "pos_FWD", "pos_GKP", "pos_MID"]:
    if col not in active.columns:
        active[col] = 0

TARGET = "total_points"

_extra = [c for c in [TARGET, "web_name", "position_name", "short_name", "price"] if c not in FEATURE_COLS]
model_df = active[FEATURE_COLS + _extra].copy().reset_index(drop=True)
for _col in FEATURE_COLS:
    model_df[_col] = pd.to_numeric(model_df[_col], errors="coerce")

# Impute remaining NaNs
imp = SimpleImputer(strategy="median")
X_raw = model_df[FEATURE_COLS].values
y     = model_df[TARGET].values.astype(float)
 
X_imp = imp.fit_transform(X_raw)
X = pd.DataFrame(X_imp, columns=FEATURE_COLS)
 
print(f"Features : {X.shape[1]}")
print(f"Samples  : {X.shape[0]}")
print(f"Target   : {TARGET}  |  mean={y.mean():.1f}  std={y.std():.1f}  max={y.max():.0f}")
 
# ─────────────────────────────────────────────
# 5. TRAIN / TEST SPLIT
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")
 
# ─────────────────────────────────────────────
# 6. TRAIN MODELS
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("5. TRAINING MODELS")
print("=" * 60)
 
models = {
    "RandomForest": Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestRegressor(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=3,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1
        ))
    ]),
    "XGBoost": Pipeline([
        ("scaler", StandardScaler()),
        ("xgb", XGBRegressor(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        ))
    ]),
}

results = {}
trained_models = {}

for name, model in models.items():
    print(f"\n  Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)

    # Cross-validation on full dataset
    cv_mae = -cross_val_score(model, X, y, cv=5, scoring="neg_mean_absolute_error", n_jobs=-1).mean()

    results[name]      = {"MAE": mae, "RMSE": rmse, "R2": r2, "CV_MAE": cv_mae}
    trained_models[name] = model

    print(f"    MAE  : {mae:.3f}")
    print(f"    RMSE : {rmse:.3f}")
    print(f"    R²   : {r2:.3f}")
    print(f"    CV MAE (5-fold): {cv_mae:.3f}")

# ─────────────────────────────────────────────
# 7. PICK BEST MODEL
# ─────────────────────────────────────────────
best_name  = min(results, key=lambda k: results[k]["MAE"])
best_model = trained_models[best_name]
print(f"\n  ★ Best model: {best_name} (MAE={results[best_name]['MAE']:.3f})")
 
# ─────────────────────────────────────────────
# 8. FEATURE IMPORTANCE
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("6. FEATURE IMPORTANCE")
print("=" * 60)
 
# Extract from the inner estimator — works for rf, gb, xgb, or any named step
inner = next(
    (est for est in best_model.named_steps.values() if hasattr(est, "feature_importances_")),
    None
)
if inner is None:
    raise ValueError(f"No estimator with feature_importances_ found in: {list(best_model.named_steps.keys())}")
importances = pd.DataFrame({
    "feature":    FEATURE_COLS,
    "importance": inner.feature_importances_
}).sort_values("importance", ascending=False)

print(importances.to_string(index=False))

print("\n" + "=" * 60)
print("7. GENERATING VISUALIZATIONS")
print("=" * 60)

FPL_GREEN  = "#00ff87"
FPL_PURPLE = "#37003c"
FPL_CYAN   = "#04f5ff"
FPL_PINK   = "#e90052"
ACCENT     = "#f8f8f8"

def fpl_style(fig, axes_list=None):
    fig.patch.set_facecolor(FPL_PURPLE)
    if axes_list is not None:
        import numpy as np
        axes_flat = np.array(axes_list).flatten()
        for ax in axes_flat:
            ax.set_facecolor("#1a0025")
            ax.tick_params(colors=ACCENT, labelsize=9)
            ax.xaxis.label.set_color(ACCENT)
            ax.yaxis.label.set_color(ACCENT)
            ax.title.set_color(FPL_GREEN)
            for spine in ax.spines.values():
                spine.set_edgecolor("#5a0080")
# ── VIZ 1: Points distribution by position ──────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(16, 5))
fig.suptitle("Total Points Distribution by Position", color=FPL_GREEN, fontsize=14, fontweight="bold")
fpl_style(fig, axes)

pos_colors = {"GKP": FPL_CYAN, "DEF": FPL_GREEN, "MID": "#ffcc00", "FWD": FPL_PINK}
for ax, pos in zip(axes, ["GKP", "DEF", "MID", "FWD"]):
    data = active[active["position_name"] == pos]["total_points"]
    ax.hist(data, bins=20, color=pos_colors[pos], edgecolor=FPL_PURPLE, alpha=0.9)
    ax.set_title(pos)
    ax.set_xlabel("Total Points")
    ax.set_ylabel("Players")
    ax.axvline(data.mean(), color="white", linestyle="--", linewidth=1.5, label=f"Mean: {data.mean():.0f}")
    ax.legend(fontsize=8, labelcolor=ACCENT, facecolor="#2a0040")

plt.tight_layout()
plt.savefig(IMGS_DIR / "viz1_points_distribution.png", dpi=150, bbox_inches="tight", facecolor=FPL_PURPLE)
plt.close()
print("  Saved: viz1_points_distribution.png")

# ── VIZ 2: Feature importance bar chart ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 8))
fpl_style(fig, [ax])
fig.suptitle(f"Feature Importance — {best_name}", color=FPL_GREEN, fontsize=13, fontweight="bold")

top_n = importances.head(15)
colors_imp = [FPL_GREEN if i < 5 else FPL_CYAN if i < 10 else "#aaaaaa" for i in range(len(top_n))]
bars = ax.barh(top_n["feature"][::-1], top_n["importance"][::-1], color=colors_imp[::-1], edgecolor=FPL_PURPLE)
for bar, val in zip(bars, top_n["importance"][::-1]):
    ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
            f"{val:.3f}", va="center", color=ACCENT, fontsize=8)
ax.set_xlabel("Importance Score")
ax.set_title("Top 15 Features", color=FPL_GREEN)

plt.tight_layout()
plt.savefig(IMGS_DIR / "viz2_feature_importance.png", dpi=150, bbox_inches="tight", facecolor=FPL_PURPLE)
plt.close()
print("  Saved: viz2_feature_importance.png")

# ── VIZ 3: Actual vs Predicted scatter (all models) ─────────────────────────
fig, axes = plt.subplots(1, len(trained_models), figsize=(6 * len(trained_models), 5))
if len(trained_models) == 1:
    axes = [axes]
fig.suptitle("Actual vs Predicted Points", color=FPL_GREEN, fontsize=13, fontweight="bold")
fpl_style(fig, axes)

for ax, (name, model) in zip(axes, trained_models.items()):
    y_pred_scatter = model.predict(X_test)
    ax.scatter(y_test, y_pred_scatter, alpha=0.5, s=20, color=FPL_CYAN, edgecolors="none")
    mn, mx = min(y_test.min(), y_pred_scatter.min()), max(y_test.max(), y_pred_scatter.max())
    ax.plot([mn, mx], [mn, mx], color=FPL_PINK, linewidth=2, linestyle="--", label="Perfect fit")
    ax.set_xlabel("Actual Points")
    ax.set_ylabel("Predicted Points")
    ax.set_title(f"{name}\nMAE={results[name]['MAE']:.2f}  R²={results[name]['R2']:.3f}")
    ax.legend(fontsize=8, labelcolor=ACCENT, facecolor="#2a0040")

plt.tight_layout()
plt.savefig(IMGS_DIR / "viz3_actual_vs_predicted.png", dpi=150, bbox_inches="tight", facecolor=FPL_PURPLE)
plt.close()
print("  Saved: viz3_actual_vs_predicted.png")

# ── VIZ 4: Residuals plot ────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(f"Residual Analysis — {best_name}", color=FPL_GREEN, fontsize=13, fontweight="bold")
fpl_style(fig, axes)

y_pred_best = best_model.predict(X_test)
residuals   = y_test - y_pred_best

# Residuals vs predicted
axes[0].scatter(y_pred_best, residuals, alpha=0.5, s=20, color=FPL_GREEN, edgecolors="none")
axes[0].axhline(0, color=FPL_PINK, linewidth=2, linestyle="--")
axes[0].set_xlabel("Predicted Points")
axes[0].set_ylabel("Residual (Actual - Predicted)")
axes[0].set_title("Residuals vs Predicted")

# Residuals histogram
axes[1].hist(residuals, bins=30, color=FPL_CYAN, edgecolor=FPL_PURPLE, alpha=0.9)
axes[1].axvline(0, color=FPL_PINK, linewidth=2, linestyle="--")
axes[1].axvline(residuals.mean(), color="#ffcc00", linewidth=1.5, linestyle="-", label=f"Mean: {residuals.mean():.2f}")
axes[1].set_xlabel("Residual Value")
axes[1].set_ylabel("Count")
axes[1].set_title("Residual Distribution")
axes[1].legend(fontsize=8, labelcolor=ACCENT, facecolor="#2a0040")

plt.tight_layout()
plt.savefig(IMGS_DIR / "viz4_residuals.png", dpi=150, bbox_inches="tight", facecolor=FPL_PURPLE)
plt.close()
print("  Saved: viz4_residuals.png")

# ── VIZ 5: Model comparison bar chart ───────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle("Model Comparison", color=FPL_GREEN, fontsize=13, fontweight="bold")
fpl_style(fig, axes)

model_names = list(results.keys())
bar_colors  = [FPL_GREEN if n == best_name else FPL_CYAN for n in model_names]

for ax, metric in zip(axes, ["MAE", "RMSE", "R2"]):
    vals = [results[n][metric] for n in model_names]
    bars = ax.bar(model_names, vals, color=bar_colors, edgecolor=FPL_PURPLE, width=0.5)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vals)*0.01,
                f"{val:.3f}", ha="center", color=ACCENT, fontsize=9, fontweight="bold")
    ax.set_title(metric, color=FPL_GREEN)
    ax.set_ylabel(metric)
    ax.tick_params(axis="x", rotation=15)

plt.tight_layout()
plt.savefig(IMGS_DIR / "viz5_model_comparison.png", dpi=150, bbox_inches="tight", facecolor=FPL_PURPLE)
plt.close()
print("  Saved: viz5_model_comparison.png")

# ── VIZ 6: Correlation heatmap of top features ──────────────────────────────
fig, ax = plt.subplots(figsize=(12, 10))
fpl_style(fig, [ax])
fig.suptitle("Feature Correlation Heatmap", color=FPL_GREEN, fontsize=13, fontweight="bold")

top_feats  = importances.head(12)["feature"].tolist()
corr_df    = pd.DataFrame(X_imp, columns=FEATURE_COLS)[top_feats].corr()
fpl_cmap   = LinearSegmentedColormap.from_list("fpl", [FPL_PINK, FPL_PURPLE, FPL_GREEN])
im         = ax.imshow(corr_df.values, cmap=fpl_cmap, vmin=-1, vmax=1, aspect="auto")
cbar       = plt.colorbar(im, ax=ax)
cbar.ax.tick_params(colors=ACCENT)
cbar.set_label("Correlation", color=ACCENT)

ax.set_xticks(range(len(top_feats)))
ax.set_yticks(range(len(top_feats)))
ax.set_xticklabels(top_feats, rotation=45, ha="right", fontsize=8)
ax.set_yticklabels(top_feats, fontsize=8)
ax.set_title("Top 12 Features", color=FPL_GREEN)

for i in range(len(top_feats)):
    for j in range(len(top_feats)):
        val = corr_df.values[i, j]
        ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                color="white" if abs(val) > 0.4 else "#aaaaaa", fontsize=7)

plt.tight_layout()
plt.savefig(IMGS_DIR / "viz6_correlation_heatmap.png", dpi=150, bbox_inches="tight", facecolor=FPL_PURPLE)
plt.close()
print("  Saved: viz6_correlation_heatmap.png")

# ─────────────────────────────────────────────
# 9. GENERATE PLAYER PREDICTIONS + RAG CSV
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("8. GENERATING PLAYER PREDICTIONS")
print("=" * 60)

X_all      = X_imp  # already imputed full dataset
y_pred_all = best_model.predict(pd.DataFrame(X_all, columns=FEATURE_COLS))

enriched = model_df[["web_name", "position_name", "short_name", "price"]].copy().reset_index(drop=True)
enriched = enriched.join(active[["team_id","ownership","form","minutes",
                                  "goals_scored","assists","clean_sheets",
                                  "bonus","ict_index","has_injury_news",
                                  "is_available","pts_per_90","pts_per_million",
                                  "gc_per_90","appearance_rate",
                                  "fixture_ease","avg_opp_strength","fixture_count",
                                  "team_avg_goals_scored","team_avg_goals_conceded",
                                  "form_tier","ownership_tier","total_points"]].reset_index(drop=True))

enriched["predicted_points"] = np.round(y_pred_all, 2)
enriched["value_score"]      = np.round(enriched["predicted_points"] / enriched["price"], 3)

# Compute a composite RAG recommendation score (0-100)
# Weights: predicted pts 40%, value 20%, form 15%, fixture ease 15%, availability 10%
def normalize(s):
    mn, mx = s.min(), s.max()
    return (s - mn) / (mx - mn + 1e-9)

enriched["rag_score"] = (
    normalize(enriched["predicted_points"]) * 40 +
    normalize(enriched["value_score"])      * 20 +
    normalize(enriched["form"])             * 15 +
    normalize(enriched["fixture_ease"])     * 15 +
    enriched["is_available"].astype(float)  * 10
).round(2)

enriched = enriched.sort_values("rag_score", ascending=False)


# ── VIZ 7: Price vs Predicted Points (value map) ────────────────────────────
fig, ax = plt.subplots(figsize=(12, 7))
fpl_style(fig, [ax])
fig.suptitle("Price vs Predicted Points — Value Map", color=FPL_GREEN, fontsize=13, fontweight="bold")

pos_colors_map = {"GKP": FPL_CYAN, "DEF": FPL_GREEN, "MID": "#ffcc00", "FWD": FPL_PINK}
for pos in ["GKP", "DEF", "MID", "FWD"]:
    mask = enriched["position_name"] == pos
    ax.scatter(
        enriched.loc[mask, "price"],
        enriched.loc[mask, "predicted_points"],
        label=pos, color=pos_colors_map[pos],
        alpha=0.7, s=40, edgecolors="none"
    )

# Label top 5 overall
for _, row in enriched.head(5).iterrows():
    ax.annotate(row["web_name"],
                (row["price"], row["predicted_points"]),
                textcoords="offset points", xytext=(5, 5),
                color=ACCENT, fontsize=7,
                bbox=dict(boxstyle="round,pad=0.2", facecolor=FPL_PURPLE, edgecolor="#5a0080"))

ax.set_xlabel("Price (£m)")
ax.set_ylabel("Predicted Points")
ax.set_title("Spot undervalued picks below the trend line", color=FPL_GREEN)
ax.legend(facecolor="#2a0040", labelcolor=ACCENT, fontsize=9)

plt.tight_layout()
plt.savefig(IMGS_DIR / "viz7_price_vs_predicted.png", dpi=150, bbox_inches="tight", facecolor=FPL_PURPLE)
plt.close()
print("  Saved: viz7_price_vs_predicted.png")

# ── VIZ 8: Classification confusion matrix — High / Low scorer ──────────────
# We treat this as a classification problem: predict if a player is a
# high scorer (above median points) to get a confusion-matrix style view
fig, axes = plt.subplots(1, len(trained_models), figsize=(6 * len(trained_models), 5))
if len(trained_models) == 1:
    axes = [axes]
fig.suptitle("Confusion Matrix — High Scorer Classification (above median points)",
             color=FPL_GREEN, fontsize=12, fontweight="bold")
fpl_style(fig, axes)

median_pts  = np.median(y)
y_class     = (y_test > median_pts).astype(int)

for ax, (name, model) in zip(axes, trained_models.items()):
    y_pred_c = (model.predict(X_test) > median_pts).astype(int)
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_class, y_pred_c)
    labels = [["TN", "FP"], ["FN", "TP"]]
    fpl_cm_cmap = LinearSegmentedColormap.from_list("fpl_cm", [FPL_PURPLE, FPL_GREEN])
    im = ax.imshow(cm, cmap=fpl_cm_cmap, aspect="auto")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{labels[i][j]}\n{cm[i,j]}",
                    ha="center", va="center", color=ACCENT, fontsize=14, fontweight="bold")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred: Low", "Pred: High"])
    ax.set_yticklabels(["Act: Low", "Act: High"])
    accuracy = (cm[0,0] + cm[1,1]) / cm.sum()
    ax.set_title(f"{name}\nAccuracy: {accuracy:.1%}", color=FPL_GREEN)

plt.tight_layout()
plt.savefig(IMGS_DIR / "viz8_confusion_matrix.png", dpi=150, bbox_inches="tight", facecolor=FPL_PURPLE)
plt.close()
print("  Saved: viz8_confusion_matrix.png")

# ── VIZ 9: Top 10 players per position by RAG score ─────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()
fig.suptitle("Top 10 Players by RAG Score — Per Position", color=FPL_GREEN, fontsize=14, fontweight="bold")
fpl_style(fig, axes)

for ax, pos in zip(axes, ["GKP", "DEF", "MID", "FWD"]):
    top10 = enriched[enriched["position_name"] == pos].head(10).copy()
    top10 = top10.sort_values("rag_score")
    bar_c = [FPL_GREEN if s >= top10["rag_score"].quantile(0.7) else FPL_CYAN for s in top10["rag_score"]]
    bars  = ax.barh(top10["web_name"], top10["rag_score"], color=bar_c, edgecolor=FPL_PURPLE)
    for bar, val in zip(bars, top10["rag_score"]):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                f"{val:.1f}", va="center", color=ACCENT, fontsize=8)
    ax.set_title(pos)
    ax.set_xlabel("RAG Score")

plt.tight_layout()
plt.savefig(IMGS_DIR / "viz9_top_players_rag.png", dpi=150, bbox_inches="tight", facecolor=FPL_PURPLE)
plt.close()
print("  Saved: viz9_top_players_rag.png")

print("\n  All 9 visualizations saved.")


# Position-wise top picks
print("\nTop 5 by position:")
for pos in ["GKP", "DEF", "MID", "FWD"]:
    top = enriched[enriched["position_name"] == pos].head(5)
    print(f"\n  {pos}:")
    print(top[["web_name","short_name","price","form","predicted_points","value_score","rag_score"]].to_string(index=False))

# ─────────────────────────────────────────────
# 9. GENERATE PLAYER PREDICTIONS + RAG CSV
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("7. GENERATING PLAYER PREDICTIONS")
print("=" * 60)
 
X_all      = X_imp  # already imputed full dataset
y_pred_all = best_model.predict(pd.DataFrame(X_all, columns=FEATURE_COLS))
 
enriched = model_df[["web_name", "position_name", "short_name", "price"]].copy().reset_index(drop=True)
enriched = enriched.join(active[["team_id","ownership","form","minutes",
                                  "goals_scored","assists","clean_sheets",
                                  "bonus","ict_index","has_injury_news",
                                  "is_available","pts_per_90","pts_per_million",
                                  "gc_per_90","appearance_rate",
                                  "fixture_ease","avg_opp_strength","fixture_count",
                                  "team_avg_goals_scored","team_avg_goals_conceded",
                                  "form_tier","ownership_tier","total_points"]].reset_index(drop=True))
 
enriched["predicted_points"] = np.round(y_pred_all, 2)
enriched["value_score"]      = np.round(enriched["predicted_points"] / enriched["price"], 3)
 
# Compute a composite RAG recommendation score (0-100)
# Weights: predicted pts 40%, value 20%, form 15%, fixture ease 15%, availability 10%
def normalize(s):
    mn, mx = s.min(), s.max()
    return (s - mn) / (mx - mn + 1e-9)
 
enriched["rag_score"] = (
    normalize(enriched["predicted_points"]) * 40 +
    normalize(enriched["value_score"])      * 20 +
    normalize(enriched["form"])             * 15 +
    normalize(enriched["fixture_ease"])     * 15 +
    enriched["is_available"].astype(float)  * 10
).round(2)
 
enriched = enriched.sort_values("rag_score", ascending=False)
 
# Position-wise top picks
print("\nTop 5 by position:")
for pos in ["GKP", "DEF", "MID", "FWD"]:
    top = enriched[enriched["position_name"] == pos].head(5)
    print(f"\n  {pos}:")
    print(top[["web_name","short_name","price","form","predicted_points","value_score","rag_score"]].to_string(index=False))
 
# ─────────────────────────────────────────────
# 10. EXPORT
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("8. EXPORTING ARTIFACTS")
print("=" * 60)
 
# XGBoost model — pickle
model_path = Model_DIR / "fpl_xgboost.pkl"
with open(model_path, "wb") as f:
    pickle.dump({
        "model":        trained_models["XGBoost"],
        "model_name":   "XGBoost",
        "feature_cols": FEATURE_COLS,
        "imputer":      imp,
        "metrics":      results["XGBoost"],
    }, f)
print(f"  Model saved   : {model_path}")

# RandomForest model — pickle
model_path = Model_DIR / "fpl_randomForest.pkl"
with open(model_path, "wb") as f:
    pickle.dump({
        "model":        trained_models["RandomForest"],
        "model_name":   "RandomForest",
        "feature_cols": FEATURE_COLS,
        "imputer":      imp,
        "metrics":      results["RandomForest"],
    }, f)
print(f"  Model saved   : {model_path}")

# (c) Enriched player CSV (for RAG)
csv_path = REPORT_DIR / "fpl_players_enriched.csv"
enriched.to_csv(csv_path, index=False)
print(f"  Player CSV    : {csv_path}")
 
# (d) Feature importance CSV
fi_path = REPORT_DIR / "fpl_feature_importance.csv"
importances.to_csv(fi_path, index=False)
print(f"  Feature imp.  : {fi_path}")
 
# (e) Evaluation report JSON
report = {
    "best_model": best_name,
    "metrics": {k: {m: round(v, 4) for m, v in r.items()} for k, r in results.items()},
    "top_features": importances.head(10).to_dict(orient="records"),
    "dataset": {
        "active_players": len(active),
        "features": len(FEATURE_COLS),
        "train_size": len(X_train),
        "test_size": len(X_test),
    }
}
report_path = REPORT_DIR / "fpl_model_report.json"
with open(report_path, "w") as f:
    json.dump(report, f, indent=2)
print(f"  Report JSON   : {report_path}")
 
print("\n" + "=" * 60)
print("PIPELINE COMPLETE ✓")
print("=" * 60)
print(f"\nBest model : {best_name}")
print(f"MAE        : {results[best_name]['MAE']:.3f} points")
print(f"RMSE       : {results[best_name]['RMSE']:.3f} points")
print(f"R²         : {results[best_name]['R2']:.3f}")