"""
FPL Intel — RAG ingestion

Thin wrapper around ``build_vector_store`` that the orchestrator / scheduler
can call without importing the top-level script directly.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from build_vector_store import rebuild_full, update_incremental


def ingest_full(db_path: str = "fpl_intel.db") -> None:
    """Full rebuild of the vector store."""
    rebuild_full(db_path)


def ingest_incremental(db_path: str = "fpl_intel.db") -> None:
    """Incremental update — only new/changed data."""
    update_incremental(db_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="FPL Intel — RAG ingest")
    parser.add_argument("db", nargs="?", default="fpl_intel.db", help="Path to SQLite DB")
    parser.add_argument("--full", action="store_true", help="Full rebuild")
    args = parser.parse_args()

    if args.full:
        ingest_full(args.db)
    else:
        ingest_incremental(args.db)
