"""
FPL Intel — FPL API data pipeline

Fetches players, teams, fixtures (with FDR), and per-player gameweek history
from the official Fantasy Premier League API and stores/updates the SQLite DB.

Tables written:
  players          – season-aggregate stats (upsert)
  teams            – club metadata (upsert)
  fixtures         – schedule + results + FDR (upsert)
  player_history   – per-player per-gameweek stats (upsert)

Usage:
  python -m pipeline.fpl_api                  # uses default DB path
  python -m pipeline.fpl_api path/to/fpl.db   # custom DB path
"""

import argparse
import sqlite3
import time
from datetime import datetime, timezone

import requests

# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

BASE = "https://fantasy.premierleague.com/api"
HEADERS = {"User-Agent": "FPL-Intel/1.0 (educational project)"}
SESSION = requests.Session()
SESSION.headers.update(HEADERS)


def _get(url: str) -> dict:
    """GET with basic retry logic."""
    for attempt in range(3):
        try:
            resp = SESSION.get(url, timeout=15)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            if attempt == 2:
                raise
            print(f"  Retrying {url} ({exc})")
            time.sleep(2 ** attempt)


# ---------------------------------------------------------------------------
# Schema bootstrap
# ---------------------------------------------------------------------------

_DDL = """
CREATE TABLE IF NOT EXISTS players (
    id                              INTEGER PRIMARY KEY,
    web_name                        TEXT,
    team_id                         INTEGER,
    position                        INTEGER,
    total_points                    INTEGER,
    now_cost                        INTEGER,
    selected_by_pct                 REAL,
    minutes                         INTEGER,
    goals_scored                    INTEGER,
    assists                         INTEGER,
    clean_sheets                    INTEGER,
    bonus                           INTEGER,
    form                            REAL,
    ict_index                       REAL,
    chance_of_playing_next_round    INTEGER,
    chance_of_playing_this_round    INTEGER,
    news                            TEXT,
    xg_scored                       REAL,
    xa                              REAL,
    yellow_cards                    INTEGER,
    red_cards                       INTEGER,
    saves                           INTEGER,
    bps                             INTEGER,
    transfers_in_event              INTEGER,
    transfers_out_event             INTEGER,
    value_season                    REAL,
    fetched_at                      TEXT
);

CREATE TABLE IF NOT EXISTS teams (
    id          INTEGER PRIMARY KEY,
    name        TEXT,
    short_name  TEXT,
    strength    INTEGER,
    fetched_at  TEXT
);

CREATE TABLE IF NOT EXISTS fixtures (
    id                  INTEGER PRIMARY KEY,
    gameweek            INTEGER,
    kickoff_time        TEXT,
    team_h              INTEGER,
    team_a              INTEGER,
    team_h_score        INTEGER,
    team_a_score        INTEGER,
    finished            INTEGER,
    team_h_difficulty   INTEGER,
    team_a_difficulty   INTEGER,
    fetched_at          TEXT
);

CREATE TABLE IF NOT EXISTS player_history (
    player_id       INTEGER,
    round           INTEGER,
    total_points    INTEGER,
    goals_scored    INTEGER,
    assists         INTEGER,
    clean_sheets    INTEGER,
    minutes         INTEGER,
    bonus           INTEGER,
    bps             INTEGER,
    xg_scored       REAL,
    xa              REAL,
    saves           INTEGER,
    yellow_cards    INTEGER,
    red_cards       INTEGER,
    fetched_at      TEXT,
    PRIMARY KEY (player_id, round)
);
"""

_ALTER_PLAYERS = [
    "ALTER TABLE players ADD COLUMN chance_of_playing_this_round INTEGER",
    "ALTER TABLE players ADD COLUMN xg_scored REAL",
    "ALTER TABLE players ADD COLUMN xa REAL",
    "ALTER TABLE players ADD COLUMN yellow_cards INTEGER",
    "ALTER TABLE players ADD COLUMN red_cards INTEGER",
    "ALTER TABLE players ADD COLUMN saves INTEGER",
    "ALTER TABLE players ADD COLUMN bps INTEGER",
    "ALTER TABLE players ADD COLUMN transfers_in_event INTEGER",
    "ALTER TABLE players ADD COLUMN transfers_out_event INTEGER",
    "ALTER TABLE players ADD COLUMN value_season REAL",
]

_ALTER_FIXTURES = [
    "ALTER TABLE fixtures ADD COLUMN team_h_difficulty INTEGER",
    "ALTER TABLE fixtures ADD COLUMN team_a_difficulty INTEGER",
]


def _migrate(conn: sqlite3.Connection) -> None:
    """Apply DDL and any missing ALTER TABLE statements idempotently."""
    conn.executescript(_DDL)
    conn.commit()
    cur = conn.cursor()
    for stmt in _ALTER_PLAYERS + _ALTER_FIXTURES:
        try:
            cur.execute(stmt)
        except sqlite3.OperationalError:
            pass  # column already exists
    conn.commit()


# ---------------------------------------------------------------------------
# Fetch & store bootstrap data (players / teams / fixtures)
# ---------------------------------------------------------------------------

def _fetch_bootstrap(conn: sqlite3.Connection) -> list[dict]:
    """Fetch bootstrap-static, upsert players/teams/fixtures. Returns raw elements."""
    now = datetime.now(timezone.utc).isoformat()
    print("Fetching bootstrap-static …")
    data = _get(f"{BASE}/bootstrap-static/")

    # --- teams ---
    conn.executemany(
        """INSERT OR REPLACE INTO teams (id, name, short_name, strength, fetched_at)
           VALUES (:id, :name, :short_name, :strength, :fetched_at)""",
        [
            {
                "id": t["id"],
                "name": t["name"],
                "short_name": t["short_name"],
                "strength": t["strength"],
                "fetched_at": now,
            }
            for t in data["teams"]
        ],
    )
    print(f"  Teams upserted: {len(data['teams'])}")

    # --- players ---
    def _float(val):
        try:
            return float(val)
        except (TypeError, ValueError):
            return None

    def _int(val):
        try:
            return int(val)
        except (TypeError, ValueError):
            return None

    players = [
        {
            "id": e["id"],
            "web_name": e["web_name"],
            "team_id": e["team"],
            "position": e["element_type"],
            "total_points": _int(e.get("total_points")),
            "now_cost": _int(e.get("now_cost")),
            "selected_by_pct": _float(e.get("selected_by_percent", e.get("selected_by_pct"))),
            "minutes": _int(e.get("minutes")),
            "goals_scored": _int(e.get("goals_scored")),
            "assists": _int(e.get("assists")),
            "clean_sheets": _int(e.get("clean_sheets")),
            "bonus": _int(e.get("bonus")),
            "form": _float(e.get("form")),
            "ict_index": _float(e.get("ict_index")),
            "chance_of_playing_next_round": _int(e.get("chance_of_playing_next_round")),
            "chance_of_playing_this_round": _int(e.get("chance_of_playing_this_round")),
            "news": e.get("news") or "",
            "xg_scored": _float(e.get("expected_goals")),
            "xa": _float(e.get("expected_assists")),
            "yellow_cards": _int(e.get("yellow_cards")),
            "red_cards": _int(e.get("red_cards")),
            "saves": _int(e.get("saves")),
            "bps": _int(e.get("bps")),
            "transfers_in_event": _int(e.get("transfers_in_event")),
            "transfers_out_event": _int(e.get("transfers_out_event")),
            "value_season": _float(e.get("value_season")),
            "fetched_at": now,
        }
        for e in data["elements"]
    ]
    conn.executemany(
        """INSERT OR REPLACE INTO players (
               id, web_name, team_id, position, total_points, now_cost,
               selected_by_pct, minutes, goals_scored, assists, clean_sheets,
               bonus, form, ict_index, chance_of_playing_next_round,
               chance_of_playing_this_round, news,
               xg_scored, xa, yellow_cards, red_cards, saves, bps,
               transfers_in_event, transfers_out_event, value_season, fetched_at
           ) VALUES (
               :id, :web_name, :team_id, :position, :total_points, :now_cost,
               :selected_by_pct, :minutes, :goals_scored, :assists, :clean_sheets,
               :bonus, :form, :ict_index, :chance_of_playing_next_round,
               :chance_of_playing_this_round, :news,
               :xg_scored, :xa, :yellow_cards, :red_cards, :saves, :bps,
               :transfers_in_event, :transfers_out_event, :value_season, :fetched_at
           )""",
        players,
    )
    print(f"  Players upserted: {len(players)}")

    # --- fixtures ---
    fixtures_raw = _get(f"{BASE}/fixtures/")
    fixtures = [
        {
            "id": f["id"],
            "gameweek": f.get("event"),
            "kickoff_time": f.get("kickoff_time"),
            "team_h": f.get("team_h"),
            "team_a": f.get("team_a"),
            "team_h_score": f.get("team_h_score"),
            "team_a_score": f.get("team_a_score"),
            "finished": int(bool(f.get("finished"))),
            "team_h_difficulty": f.get("team_h_difficulty"),
            "team_a_difficulty": f.get("team_a_difficulty"),
            "fetched_at": now,
        }
        for f in fixtures_raw
    ]
    conn.executemany(
        """INSERT OR REPLACE INTO fixtures (
               id, gameweek, kickoff_time, team_h, team_a,
               team_h_score, team_a_score, finished,
               team_h_difficulty, team_a_difficulty, fetched_at
           ) VALUES (
               :id, :gameweek, :kickoff_time, :team_h, :team_a,
               :team_h_score, :team_a_score, :finished,
               :team_h_difficulty, :team_a_difficulty, :fetched_at
           )""",
        fixtures,
    )
    print(f"  Fixtures upserted: {len(fixtures)}")
    conn.commit()
    return data["elements"]


# ---------------------------------------------------------------------------
# Fetch per-player gameweek history
# ---------------------------------------------------------------------------

def _fetch_player_histories(conn: sqlite3.Connection, elements: list[dict]) -> None:
    """Fetch element-summary for every player and upsert gameweek history."""
    now = datetime.now(timezone.utc).isoformat()
    total = len(elements)
    inserted = 0

    def _float(val):
        try:
            return float(val)
        except (TypeError, ValueError):
            return None

    def _int(val):
        try:
            return int(val)
        except (TypeError, ValueError):
            return None

    for i, element in enumerate(elements, 1):
        pid = element["id"]
        try:
            detail = _get(f"{BASE}/element-summary/{pid}/")
        except requests.RequestException as exc:
            print(f"  Skipping player {pid}: {exc}")
            continue

        rows = []
        for h in detail.get("history", []):
            rows.append({
                "player_id": pid,
                "round": _int(h.get("round")),
                "total_points": _int(h.get("total_points")),
                "goals_scored": _int(h.get("goals_scored")),
                "assists": _int(h.get("assists")),
                "clean_sheets": _int(h.get("clean_sheets")),
                "minutes": _int(h.get("minutes")),
                "bonus": _int(h.get("bonus")),
                "bps": _int(h.get("bps")),
                "xg_scored": _float(h.get("expected_goals")),
                "xa": _float(h.get("expected_assists")),
                "saves": _int(h.get("saves")),
                "yellow_cards": _int(h.get("yellow_cards")),
                "red_cards": _int(h.get("red_cards")),
                "fetched_at": now,
            })

        if rows:
            conn.executemany(
                """INSERT OR REPLACE INTO player_history (
                       player_id, round, total_points, goals_scored, assists,
                       clean_sheets, minutes, bonus, bps, xg_scored, xa,
                       saves, yellow_cards, red_cards, fetched_at
                   ) VALUES (
                       :player_id, :round, :total_points, :goals_scored, :assists,
                       :clean_sheets, :minutes, :bonus, :bps, :xg_scored, :xa,
                       :saves, :yellow_cards, :red_cards, :fetched_at
                   )""",
                rows,
            )
            inserted += len(rows)

        if i % 100 == 0 or i == total:
            conn.commit()
            print(f"  Player history: {i}/{total} players processed ({inserted} rows)")

        # Be polite to the API
        time.sleep(0.05)

    conn.commit()
    print(f"  Player history upserted: {inserted} total rows")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run(db_path: str = "fpl_intel.db") -> None:
    """Fetch all FPL data and write to *db_path*."""
    print(f"=== FPL API pipeline — {datetime.now(timezone.utc).isoformat()} ===")
    conn = sqlite3.connect(db_path)
    try:
        _migrate(conn)
        elements = _fetch_bootstrap(conn)
        _fetch_player_histories(conn, elements)
    finally:
        conn.close()
    print("=== Done ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FPL Intel — fetch FPL API data")
    parser.add_argument("db", nargs="?", default="fpl_intel.db", help="Path to SQLite DB")
    args = parser.parse_args()
    run(args.db)
