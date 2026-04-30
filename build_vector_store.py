"""
FPL Intel — Vector store builder (incremental)

Builds or incrementally updates the ChromaDB vector store from the SQLite DB.

Incremental logic:
  * News articles: tracks highest ``id`` already embedded in a local
    ``_vector_store_state`` table.  Only new rows are ingested.
  * Players: existing player documents are deleted by ``name`` metadata and
    re-inserted so stats stay current.

Full rebuild:
  python build_vector_store.py --full      # wipe & rebuild from scratch

Incremental update (default):
  python build_vector_store.py             # embed only new/changed data
"""

import argparse
import sqlite3

import pandas as pd
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

VECTOR_DIR = "./fpl_vector_db"
COLLECTION = "fpl_rag"
EMBED_MODEL = "all-MiniLM-L6-v2"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)


def _get_vector_db(embeddings: HuggingFaceEmbeddings) -> Chroma:
    return Chroma(
        persist_directory=VECTOR_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION,
    )


def _ensure_state_table(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS _vector_store_state (
            key   TEXT PRIMARY KEY,
            value TEXT
        )
    """)
    conn.commit()


def _get_state(conn: sqlite3.Connection, key: str, default: str = "0") -> str:
    row = conn.execute(
        "SELECT value FROM _vector_store_state WHERE key = ?", (key,)
    ).fetchone()
    return row[0] if row else default


def _set_state(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO _vector_store_state (key, value) VALUES (?, ?)",
        (key, value),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Document builders
# ---------------------------------------------------------------------------

_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)


def _build_news_docs(news_df: pd.DataFrame) -> list:
    docs = []
    for _, row in news_df.iterrows():
        full_text = (
            f"Source: {row['source']}\n"
            f"Title: {row['title']}\n"
            f"Content: {row['body']}"
        )
        for chunk in _splitter.split_text(full_text):
            docs.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "type": "news",
                        "article_id": int(row["id"]),
                        "source": row["source"],
                    },
                )
            )
    return docs


def _build_player_docs(players_df: pd.DataFrame) -> list:
    docs = []
    for _, row in players_df.iterrows():
        def _fmt(val, fmt=".2f"):
            try:
                import math
                if val is None or (isinstance(val, float) and math.isnan(val)):
                    return "N/A"
                return format(float(val), fmt)
            except (TypeError, ValueError):
                return "N/A"

        content = (
            f"Player: {row['web_name']}\n"
            f"Total Points: {row['total_points']}\n"
            f"Cost: £{row['now_cost'] / 10:.1f}m\n"
            f"Goals: {row['goals_scored']}\n"
            f"Assists: {row['assists']}\n"
            f"Clean Sheets: {row['clean_sheets']}\n"
            f"xG: {_fmt(row.get('xg_scored'))}\n"
            f"xA: {_fmt(row.get('xa'))}\n"
            f"Yellow Cards: {row.get('yellow_cards', 'N/A')}\n"
            f"Red Cards: {row.get('red_cards', 'N/A')}\n"
            f"Saves: {row.get('saves', 'N/A')}\n"
            f"BPS: {row.get('bps', 'N/A')}\n"
            f"Transfers In (GW): {row.get('transfers_in_event', 'N/A')}\n"
            f"Transfers Out (GW): {row.get('transfers_out_event', 'N/A')}\n"
            f"Form: {row['form']}\n"
            f"News: {row.get('news') or 'No news'}"
        )
        docs.append(
            Document(
                page_content=content,
                metadata={"type": "player", "name": row["web_name"]},
            )
        )
    return docs


# ---------------------------------------------------------------------------
# Full rebuild
# ---------------------------------------------------------------------------

def rebuild_full(db_path: str = "fpl_intel.db") -> None:
    """Wipe the collection and rebuild everything from scratch."""
    print("=== Full vector store rebuild ===")
    conn = sqlite3.connect(db_path)
    try:
        _ensure_state_table(conn)
        embeddings = _get_embeddings()

        # Delete existing collection
        db = _get_vector_db(embeddings)
        try:
            db.delete_collection()
            print("  Existing collection deleted.")
        except Exception:
            pass

        news_df = pd.read_sql_query("SELECT id, title, body, source FROM news_articles", conn)
        players_df = pd.read_sql_query(
            """SELECT web_name, total_points, now_cost, goals_scored, assists,
                      clean_sheets, form, news,
                      xg_scored, xa, yellow_cards, red_cards, saves, bps,
                      transfers_in_event, transfers_out_event
               FROM players""",
            conn,
        )

        news_docs = _build_news_docs(news_df)
        player_docs = _build_player_docs(players_df)
        all_docs = news_docs + player_docs

        print(f"  News chunks: {len(news_docs)} | Player docs: {len(player_docs)}")
        print("  Building vector database ...")
        Chroma.from_documents(
            documents=all_docs,
            embedding=embeddings,
            persist_directory=VECTOR_DIR,
            collection_name=COLLECTION,
        )

        # Record watermarks
        if not news_df.empty:
            _set_state(conn, "last_news_id", str(int(news_df["id"].max())))
        _set_state(conn, "players_embedded", "1")
        print("=== Done ===")
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Incremental update
# ---------------------------------------------------------------------------

def update_incremental(db_path: str = "fpl_intel.db") -> None:
    """Embed only new news articles and refresh all player documents."""
    print("=== Incremental vector store update ===")
    conn = sqlite3.connect(db_path)
    try:
        _ensure_state_table(conn)
        embeddings = _get_embeddings()
        vector_db = _get_vector_db(embeddings)

        # --- New news articles ---
        last_news_id = int(_get_state(conn, "last_news_id", "0"))
        new_news_df = pd.read_sql_query(
            "SELECT id, title, body, source FROM news_articles WHERE id > ?",
            conn,
            params=(last_news_id,),
        )
        if not new_news_df.empty:
            news_docs = _build_news_docs(new_news_df)
            print(f"  Adding {len(news_docs)} new news chunks (from {len(new_news_df)} articles) ...")
            vector_db.add_documents(news_docs)
            _set_state(conn, "last_news_id", str(int(new_news_df["id"].max())))
        else:
            print("  No new news articles.")

        # --- Refresh player docs (delete old, re-insert with latest stats) ---
        players_df = pd.read_sql_query(
            """SELECT web_name, total_points, now_cost, goals_scored, assists,
                      clean_sheets, form, news,
                      xg_scored, xa, yellow_cards, red_cards, saves, bps,
                      transfers_in_event, transfers_out_event
               FROM players""",
            conn,
        )
        if not players_df.empty:
            print(f"  Refreshing {len(players_df)} player documents ...")
            for name in players_df["web_name"].tolist():
                try:
                    existing = vector_db.get(where={"name": name})
                    ids_to_delete = existing.get("ids", [])
                    if ids_to_delete:
                        vector_db.delete(ids=ids_to_delete)
                except Exception:
                    pass
            player_docs = _build_player_docs(players_df)
            vector_db.add_documents(player_docs)
            _set_state(conn, "players_embedded", "1")

        print("=== Done ===")
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FPL Intel — build/update vector store")
    parser.add_argument("db", nargs="?", default="fpl_intel.db", help="Path to SQLite DB")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Full rebuild (wipe + rebuild from scratch)",
    )
    args = parser.parse_args()

    if args.full:
        rebuild_full(args.db)
    else:
        update_incremental(args.db)
