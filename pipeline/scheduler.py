"""
FPL Intel — Background scheduler

Runs the FPL API pipeline and news scraper on a fixed schedule using
APScheduler so the database and vector store stay current throughout the season.

Schedule:
  • FPL API (players / fixtures / history)  — every 6 hours
  • News scraper                             — every 3 hours
  • Vector store incremental rebuild         — after each DB update

Usage:
  python -m pipeline.scheduler                   # uses default DB path
  python -m pipeline.scheduler path/to/fpl.db    # custom DB path

  The process runs until interrupted (Ctrl-C / SIGTERM).
"""

import argparse
import logging
import os
import sys

# Ensure project root is on sys.path when run as a script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.interval import IntervalTrigger

from pipeline.fpl_api import run as run_fpl_api
from pipeline.news_scraper import run as run_news_scraper

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("fpl_intel.scheduler")

# ---------------------------------------------------------------------------
# Job wrappers
# ---------------------------------------------------------------------------

_DB_PATH = "fpl_intel.db"


def _job_fpl_api() -> None:
    log.info("Starting FPL API refresh …")
    try:
        run_fpl_api(_DB_PATH)
        log.info("FPL API refresh complete — triggering vector store update")
        _job_vector_store()
    except Exception as exc:
        log.error("FPL API refresh failed: %s", exc, exc_info=True)


def _job_news() -> None:
    log.info("Starting news scrape …")
    try:
        inserted = run_news_scraper(_DB_PATH)
        log.info("News scrape complete — %d new articles", inserted)
        if inserted > 0:
            log.info("New articles found — triggering vector store update")
            _job_vector_store()
    except Exception as exc:
        log.error("News scrape failed: %s", exc, exc_info=True)


def _job_vector_store() -> None:
    """Incremental vector store update — import lazily to avoid loading heavy
    ML dependencies unless actually needed."""
    try:
        import importlib
        vstore = importlib.import_module("build_vector_store")
        if hasattr(vstore, "update_incremental"):
            vstore.update_incremental(_DB_PATH)
        else:
            log.warning(
                "build_vector_store has no update_incremental(); skipping vector store update"
            )
    except Exception as exc:
        log.error("Vector store update failed: %s", exc, exc_info=True)


# ---------------------------------------------------------------------------
# Scheduler setup
# ---------------------------------------------------------------------------

def run(db_path: str = "fpl_intel.db") -> None:
    global _DB_PATH
    _DB_PATH = db_path

    scheduler = BlockingScheduler(timezone="UTC")

    # FPL API every 6 hours
    scheduler.add_job(
        _job_fpl_api,
        trigger=IntervalTrigger(hours=6),
        id="fpl_api",
        name="FPL API refresh",
        replace_existing=True,
        max_instances=1,
        misfire_grace_time=300,
    )

    # News scraper every 3 hours
    scheduler.add_job(
        _job_news,
        trigger=IntervalTrigger(hours=3),
        id="news_scraper",
        name="News scraper",
        replace_existing=True,
        max_instances=1,
        misfire_grace_time=300,
    )

    log.info(
        "Scheduler started. FPL API: every 6 h | News: every 3 h | DB: %s",
        db_path,
    )
    log.info("Press Ctrl-C to stop.")

    # Run both jobs immediately on startup so the DB is fresh
    log.info("Running initial FPL API fetch …")
    _job_fpl_api()
    log.info("Running initial news scrape …")
    _job_news()

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        log.info("Scheduler stopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FPL Intel — background scheduler")
    parser.add_argument("db", nargs="?", default="fpl_intel.db", help="Path to SQLite DB")
    args = parser.parse_args()
    run(args.db)
