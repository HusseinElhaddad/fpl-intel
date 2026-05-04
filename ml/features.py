    # inputs: fpl_intel.db(database), database path as argument
    # outputs: fpl_features.csv(features), model_df.csv()

import sqlite3
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

def feature_engineering(player_name = ""):

    # ─────────────────────────────────────────────
    # 1. LOAD RAW DATA
    # ─────────────────────────────────────────────
    print("=" * 60)
    print("1. LOADING DATA")
    print("=" * 60)
    

    DB_PATH = "fpl_intel.db"
    conn = sqlite3.connect(DB_PATH)
    
    if player_name == "":
        players  = pd.read_sql("SELECT * FROM players",  conn)
        fixtures = pd.read_sql("SELECT * FROM fixtures", conn)
        teams    = pd.read_sql("SELECT * FROM teams",    conn)
    else:
        players  = pd.read_sql("SELECT * FROM players WHERE web_name = ?", conn, params=[player_name])

        team_id  = int(players["team_id"][0])

        fixtures = pd.read_sql("SELECT * FROM fixtures WHERE team_h = ? OR team_a = ?", conn, params=[team_id, team_id])
        teams    = pd.read_sql("SELECT * FROM teams WHERE id = ?",    conn, params=[team_id])
        
    conn.close()
    
    print(type(players))
    print(f"Players : {players.shape}")
    print(f"Fixtures: {fixtures.shape}")
    print(f"Teams   : {teams.shape}")
    
    # ─────────────────────────────────────────────
    # 2. PREPROCESSING
    # ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("2. PREPROCESSING")
    print("=" * 60)
    
    # --- Position mapping ---
    POS_MAP = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
    players["position_name"] = players["position"].map(POS_MAP)
    
    # --- Cost: FPL stores cost * 10 ---
    players["price"] = players["now_cost"] / 10.0
    
    # --- Availability flag ---
    players["is_available"] = players["chance_of_playing_next_round"].fillna(100) >= 75
    
    # --- Injury flag from news ---
    injury_keywords = ["injury", "injur", "doubt", "ill", "operation", "surgery",
                    "knock", "strain", "hamstring", "suspended", "ban"]
    players["has_injury_news"] = players["news"].str.lower().str.contains(
        "|".join(injury_keywords), na=False
    ).astype(int)
    
    # --- Loan flag ---
    players["on_loan"] = players["news"].str.lower().str.contains("loan", na=False).astype(int)
    
    # --- Active player (played at least 1 min) ---
    players["is_active"] = (players["minutes"] > 0).astype(int)
    
    # --- Drop blanks (loan / inactive with 0 pts and 0 mins) ---
    active = players[(players["minutes"] > 0) | (players["total_points"] > 0)].copy()
    print(f"Active players (mins > 0 or pts > 0): {len(active)}")
    
    # ─────────────────────────────────────────────
    # 3. FEATURE ENGINEERING
    # ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("3. FEATURE ENGINEERING")
    print("=" * 60)
    
    # --- Merge team info ---
    active = active.merge(
        teams[["id", "name", "short_name", "strength"]],
        left_on="team_id", right_on="id", suffixes=("", "_team")
    )
    
    # --- Appearance rate (proxy for consistency) ---
    # GW range: full season = 38 GW, max possible minutes per GW = 90
    MAX_MINUTES = 38 * 90
    active["appearance_rate"] = active["minutes"] / MAX_MINUTES          # 0-1 scale
    active["avg_minutes_per_gw"] = active["minutes"] / 38               # avg mins per GW
    
    # --- Points efficiency ---
    active["pts_per_90"]        = np.where(
        active["minutes"] > 0,
        active["total_points"] / (active["minutes"] / 90),
        0
    )
    active["pts_per_million"]   = active["total_points"] / active["price"]
    
    # --- Goal contributions ---
    active["goal_contributions"] = active["goals_scored"] + active["assists"]
    active["gc_per_90"]          = np.where(
        active["minutes"] > 0,
        active["goal_contributions"] / (active["minutes"] / 90),
        0
    )
    active["goals_per_90"]       = np.where(
        active["minutes"] > 0,
        active["goals_scored"] / (active["minutes"] / 90),
        0
    )
    active["assists_per_90"]     = np.where(
        active["minutes"] > 0,
        active["assists"] / (active["minutes"] / 90),
        0
    )
    
    # --- Bonus points rate ---
    active["bonus_per_90"] = np.where(
        active["minutes"] > 0,
        active["bonus"] / (active["minutes"] / 90),
        0
    )
    
    # --- Clean sheet rate (defenders / keepers) ---
    active["cs_per_90"] = np.where(
        active["minutes"] > 0,
        active["clean_sheets"] / (active["minutes"] / 90),
        0
    )
    
    # --- ICT breakdown proxy (FPL ICT is composite: Influence, Creativity, Threat) ---
    # Threat correlates with goals for FWDs/MIDs, Creativity with assists
    # We can't split ICT without the raw endpoint, but we can normalize it
    active["ict_per_90"] = np.where(
        active["minutes"] > 0,
        active["ict_index"] / (active["minutes"] / 90),
        0
    )
    
    # --- Form signal (recent 6-GW rolling avg provided by FPL API) ---
    # Encode form as numeric; handle potential string values
    active["form"] = pd.to_numeric(active["form"], errors="coerce").fillna(0)
    active["form_tier"] = pd.cut(
        active["form"],
        bins=[-0.01, 2, 4, 7, 100],
        labels=["cold", "average", "hot", "elite"]
    )
    
    # --- Ownership category ---
    active["ownership"] = pd.to_numeric(active["selected_by_pct"], errors="coerce").fillna(0)
    active["ownership_tier"] = pd.cut(
        active["ownership"],
        bins=[-0.01, 5, 15, 30, 100],
        labels=["differential", "medium", "popular", "template"]
    )
    
    # ─── FIXTURE DIFFICULTY FEATURES ───────────────────────────────────────────────
    
    # Compute team-level fixture stats from finished fixtures
    finished = fixtures[fixtures["finished"] == 1].copy()
    upcoming = fixtures[fixtures["finished"] == 0].copy()
    
    # Goals scored / conceded per team (from finished GWs)
    home_scored  = finished.groupby("team_h")["team_h_score"].sum().rename("home_scored")
    away_scored  = finished.groupby("team_a")["team_a_score"].sum().rename("away_scored")
    home_conceded = finished.groupby("team_h")["team_a_score"].sum().rename("home_conceded")
    away_conceded = finished.groupby("team_a")["team_h_score"].sum().rename("away_conceded")
    
    team_goals_for = (home_scored.add(away_scored, fill_value=0)).rename("team_goals_for")
    team_goals_against = (home_conceded.add(away_conceded, fill_value=0)).rename("team_goals_against")
    team_games_played = (
        finished.groupby("team_h")["id"].count().add(
            finished.groupby("team_a")["id"].count(), fill_value=0
        )
    ).rename("games_played")
    
    team_stats = pd.concat([team_goals_for, team_goals_against, team_games_played], axis=1).fillna(0)
    team_stats["avg_goals_scored"]   = team_stats["team_goals_for"]    / team_stats["games_played"].clip(lower=1)
    team_stats["avg_goals_conceded"] = team_stats["team_goals_against"] / team_stats["games_played"].clip(lower=1)
    
    active = active.merge(
        team_stats[["avg_goals_scored", "avg_goals_conceded"]].add_prefix("team_"),
        left_on="team_id", right_index=True, how="left"
    )
    
    # Upcoming fixture difficulty for each team (average opponent strength)
    def get_fixture_difficulty(upcoming_df, teams_df):
        rows = []
        for _, fix in upcoming_df.iterrows():
            h_strength = teams_df.loc[teams_df["id"] == fix["team_h"], "strength"].values
            a_strength = teams_df.loc[teams_df["id"] == fix["team_a"], "strength"].values
            if len(h_strength) and len(a_strength):
                rows.append({"team_id": fix["team_h"], "opp_strength": a_strength[0]})
                rows.append({"team_id": fix["team_a"], "opp_strength": h_strength[0]})
        if not rows:
            return pd.DataFrame(columns=["team_id", "avg_opp_strength", "fixture_count"])
        df = pd.DataFrame(rows)
        return df.groupby("team_id")["opp_strength"].agg(
            avg_opp_strength="mean", fixture_count="count"
        ).reset_index()
    
    fix_diff = get_fixture_difficulty(upcoming, teams)
    active = active.merge(fix_diff, on="team_id", how="left")
    active["avg_opp_strength"] = active["avg_opp_strength"].fillna(active["strength"])
    active["fixture_count"]    = active["fixture_count"].fillna(0)
    
    # Fixture difficulty score: low opponent strength = easy = lower number = good
    active["fixture_ease"] = (6 - active["avg_opp_strength"]).clip(lower=0)  # 1=hard, 4=easy
    
    # --- Position dummies ---
    pos_dummies = pd.get_dummies(active["position_name"], prefix="pos")
    active = pd.concat([active, pos_dummies], axis=1)
    
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
    TARGET="total_points"

        # Ensure all pos_ cols exist
    for col in ["pos_DEF", "pos_FWD", "pos_GKP", "pos_MID"]:
        if col not in active.columns:
            active[col] = 0

    _extra = [c for c in [TARGET, "web_name", "position_name", "short_name", "price"] if c not in FEATURE_COLS]
    model_df = active[FEATURE_COLS + _extra].copy().reset_index(drop=True)
    for _col in FEATURE_COLS:
        model_df[_col] = pd.to_numeric(model_df[_col], errors="coerce")
    
    print("Engineered features added.")
    print(f"Dataset shape: {active.shape}")
    if player_name == "":
        features_csv = Path("data/fpl_features.csv")
        model_df_csv = Path("data/model_df.csv")
        active.to_csv(features_csv, index=False)
        model_df.to_csv(model_df_csv, index=False)
        print(f"  Saved: {features_csv}")
        print(f"  Saved: {model_df_csv}")
    else:
        return model_df

if __name__ == "__main__":
    feature_engineering()