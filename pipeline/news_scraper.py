"""
FPL Intel — News scraper

Collects FPL-related news from multiple public sources and stores them in the
``news_articles`` table of the SQLite DB.

Sources:
  1. FPL bootstrap-static player news field (official, authoritative)
  2. BBC Sport Fantasy Football RSS feed
  3. Sky Sports Fantasy Football RSS feed
  4. FPL Injuries RSS feed (fplfixtures.com)
  5. Planet FPL RSS feed

Deduplication is handled by the ``url UNIQUE`` constraint — duplicate inserts
are silently ignored.

Usage:
  python -m pipeline.news_scraper                   # uses default DB path
  python -m pipeline.news_scraper path/to/fpl.db    # custom DB path
"""

import argparse
import sqlite3
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from html.parser import HTMLParser

import requests

BASE = "https://fantasy.premierleague.com/api"
HEADERS = {"User-Agent": "FPL-Intel/1.0 (educational project)"}
SESSION = requests.Session()
SESSION.headers.update(HEADERS)

# ---------------------------------------------------------------------------
# RSS sources
# ---------------------------------------------------------------------------

RSS_FEEDS = [
    {
        "name": "BBC Sport Fantasy Football",
        "url": "https://feeds.bbci.co.uk/sport/football/rss.xml",
    },
    {
        "name": "Sky Sports Football",
        "url": "https://www.skysports.com/rss/12040",
    },
    {
        "name": "FPL Fixtures / Injuries",
        "url": "https://www.fplfixtures.com/feed/",
    },
    {
        "name": "Planet FPL",
        "url": "https://www.planetfpl.com/feed/",
    },
    {
        "name": "FPL Review",
        "url": "https://fplreview.com/feed/",
    },
    {
        "name": "Fantasy Football Scout",
        "url": "https://www.fantasyfootballscout.co.uk/feed/",
    },
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _HTMLStripper(HTMLParser):
    """Minimal HTML → plain-text converter."""

    def __init__(self):
        super().__init__()
        self._parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self._parts.append(data)

    def get_text(self) -> str:
        return " ".join(self._parts).strip()


def _strip_html(text: str) -> str:
    p = _HTMLStripper()
    try:
        p.feed(text or "")
        return p.get_text()
    except Exception:
        return text or ""


def _get(url: str, timeout: int = 15) -> requests.Response | None:
    for attempt in range(3):
        try:
            resp = SESSION.get(url, timeout=timeout)
            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:
            if attempt == 2:
                print(f"  Failed to fetch {url}: {exc}")
                return None
            time.sleep(2 ** attempt)
    return None


# ---------------------------------------------------------------------------
# Scrapers
# ---------------------------------------------------------------------------


def _parse_rss(xml_text: str, source: str) -> list[dict]:
    """Parse RSS/Atom XML and return a list of article dicts."""
    articles: list[dict] = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return articles

    # Handle both RSS <item> and Atom <entry>
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    items = root.findall(".//item") or root.findall(".//atom:entry", ns)

    for item in items:
        def _text(tag: str) -> str:
            el = item.find(tag) or item.find(f"atom:{tag}", ns)
            return (el.text or "").strip() if el is not None else ""

        title = _text("title")
        url = _text("link") or _text("guid")
        # Atom uses <id> for URL
        if not url:
            el = item.find("atom:id", ns)
            url = (el.text or "").strip() if el is not None else ""

        body = _strip_html(_text("description") or _text("summary") or _text("content"))

        if title and url:
            articles.append({
                "source": source,
                "title": title,
                "url": url,
                "body": body,
            })

    return articles


def _scrape_rss_feeds() -> list[dict]:
    """Fetch all configured RSS feeds and return combined article list."""
    all_articles: list[dict] = []
    for feed in RSS_FEEDS:
        resp = _get(feed["url"])
        if resp is None:
            continue
        articles = _parse_rss(resp.text, feed["name"])
        print(f"  {feed['name']}: {len(articles)} articles")
        all_articles.extend(articles)
    return all_articles


def _scrape_fpl_player_news(db_path: str) -> list[dict]:
    """
    Extract news snippets from the FPL bootstrap-static endpoint for every
    player that has a non-empty ``news`` field.  Each player gets one article
    with a synthetic URL so deduplication works across runs.
    """
    resp = _get(f"{BASE}/bootstrap-static/")
    if resp is None:
        return []

    data = resp.json()
    articles: list[dict] = []
    for e in data.get("elements", []):
        news_text = (e.get("news") or "").strip()
        if not news_text:
            continue
        player_name = e.get("web_name", str(e["id"]))
        # Use a stable synthetic URL so the UNIQUE constraint deduplicates
        synthetic_url = f"fpl://player-news/{e['id']}"
        articles.append({
            "source": "FPL Official",
            "title": f"{player_name} — FPL Update",
            "url": synthetic_url,
            "body": news_text,
        })

    print(f"  FPL Official player news: {len(articles)} articles")
    return articles


# ---------------------------------------------------------------------------
# DB write
# ---------------------------------------------------------------------------


def _upsert_articles(conn: sqlite3.Connection, articles: list[dict]) -> int:
    """Insert articles, ignoring duplicates. Returns count of new rows."""
    now = datetime.now(timezone.utc).isoformat()
    inserted = 0
    for art in articles:
        try:
            conn.execute(
                """INSERT INTO news_articles (source, title, url, body, scraped_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (art["source"], art["title"], art["url"], art["body"], now),
            )
            inserted += 1
        except sqlite3.IntegrityError:
            pass  # duplicate URL — skip
    conn.commit()
    return inserted


def _ensure_table(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS news_articles (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            source      TEXT,
            title       TEXT,
            url         TEXT UNIQUE,
            body        TEXT,
            scraped_at  TEXT
        )
    """)
    conn.commit()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run(db_path: str = "fpl_intel.db") -> int:
    """
    Scrape all news sources and persist to *db_path*.
    Returns the number of newly inserted articles.
    """
    print(f"=== News scraper — {datetime.now(timezone.utc).isoformat()} ===")
    conn = sqlite3.connect(db_path)
    try:
        _ensure_table(conn)
        articles: list[dict] = []
        articles.extend(_scrape_fpl_player_news(db_path))
        articles.extend(_scrape_rss_feeds())
        inserted = _upsert_articles(conn, articles)
        total = conn.execute("SELECT COUNT(*) FROM news_articles").fetchone()[0]
        print(f"  New articles inserted: {inserted}  |  Total in DB: {total}")
    finally:
        conn.close()
    print("=== Done ===")
    return inserted


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FPL Intel — scrape news")
    parser.add_argument("db", nargs="?", default="fpl_intel.db", help="Path to SQLite DB")
    args = parser.parse_args()
    run(args.db)
