"""Incremental GDELT DOC 2.0 poller.
Duplicate from dgelt tempalte
"""
import hashlib
import re
import sys
import time
import os
from datetime import UTC, datetime, timedelta
from urllib.parse import urlparse

from pipeline import db
from pipeline.http import throttled_get

API_URL = "https://newsapi.org/v2/everything"
MAX_RECORDS = 100          # NewsAPI's pageSize max
OVERLAP = timedelta(minutes=15)
FIRST_RUN_LOOKBACK = timedelta(hours=72)
MIN_WINDOW = timedelta(minutes=1)   # bisect guard

NEWSAPI_KEY = os.environ["NEWSAPI_KEY"]


def _fmt(dt: datetime) -> str:
    # NewsAPI wants ISO 8601, e.g. 2026-07-15T14:30:00
    return dt.strftime("%Y-%m-%dT%H:%M:%S")


def fetch_window(query: str, start: datetime, end: datetime) -> list[dict]:
    resp = throttled_get(API_URL, label="NEWSAPI", throttle_s=1.0, params={
        "q": query,
        "from": _fmt(start),
        "to": _fmt(end),
        "sortBy": "publishedAt",
        "pageSize": MAX_RECORDS,
        "language": "en",
        "apiKey": NEWSAPI_KEY,
    })
    try:
        payload = resp.json()
    except ValueError:
        raise RuntimeError(f"NEWSAPI non-JSON response: {resp.text[:200]}")

    if payload.get("status") != "ok":
        # NewsAPI returns {"status":"error","code":...,"message":...} with HTTP 200 sometimes
        raise RuntimeError(f"NEWSAPI error response: {payload}")

    articles = payload.get("articles", [])
    total_results = payload.get("totalResults", len(articles))

    if total_results > MAX_RECORDS:
        if (end - start) > MIN_WINDOW:
            mid = start + (end - start) / 2
            return fetch_window(query, start, mid) + fetch_window(query, mid, end)

        print(f"WARNING: still {total_results} results at minimum window "
              f"{_fmt(start)}-{_fmt(end)}; articles beyond the newest "
              f"{MAX_RECORDS} are dropped (query: {query[:80]})",
              file=sys.stderr)
    return articles


def normalize_title_hash(title: str) -> str:
    normalized = re.sub(r"[\W_]+", " ", title.lower()).strip()
    return hashlib.sha1(normalized.encode()).hexdigest()


def _parse_published_at(value: str) -> datetime | None:
    # NewsAPI's publishedAt is ISO 8601, e.g. "2026-07-15T14:30:00Z"
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
    except (ValueError, TypeError):
        return None


def _domain(url: str | None) -> str | None:
    if not url:
        return None
    try:
        return urlparse(url).netloc or None
    except ValueError:
        return None


def to_rows(bank_id: str, articles: list[dict]) -> list[dict]:
    """Fold syndication duplicates (same normalized title) into one row each."""
    by_hash: dict[str, dict] = {}
    seen_urls: set[str] = set()
    for a in articles:
        url = a.get("url")
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)
        title_hash = normalize_title_hash(a.get("title") or url)
        if title_hash in by_hash:
            by_hash[title_hash]["n_duplicates"] += 1
            continue
        by_hash[title_hash] = {
            "source": "newsapi",
            "external_id": url,
            "bank_id": bank_id,
            "published_at": _parse_published_at(a.get("publishedAt")),
            "title": a.get("title"),
            "url": url,
            "domain": _domain(url),
            "text_excerpt": a.get("description"),
            "title_hash": title_hash,
            "n_duplicates": 0,
            "meta": {
                "author": a.get("author"),
                "outlet": (a.get("source") or {}).get("name"),
                "url_to_image": a.get("urlToImage"),
                "content_snippet": a.get("content"),
            },
        }
    return list(by_hash.values())


def main() -> None:
    started = time.monotonic()
    seen = inserted = 0
    failed: list[str] = []
    conn = db.connect()
    try:
        for bank in db.get_live_banks(conn):
            if not bank["aliases"]:
                continue
            try:
                run_start = datetime.now(UTC)
                watermark = db.get_watermark(conn, "newsapi", bank["bank_id"])
                window_start = (watermark or run_start - FIRST_RUN_LOOKBACK) - OVERLAP
                query = " OR ".join(bank["aliases"]) if isinstance(bank["aliases"], list) else bank["aliases"]
                articles = fetch_window(query, window_start, run_start)
                rows = to_rows(bank["bank_id"], articles)
                # The overlap re-fetch can pick a different representative URL
                # for a story already stored (UNIQUE on external_id wouldn't
                # catch it) — drop rows whose title we already have.
                known = db.existing_title_hashes(
                    conn, "newsapi", bank["bank_id"], [r["title_hash"] for r in rows])
                rows = [r for r in rows if r["title_hash"] not in known]
                n = db.upsert_raw_items(conn, rows)
                db.set_watermark(conn, "newsapi", bank["bank_id"], run_start)
                seen += len(articles)
                inserted += n
                print(f"{bank['bank_id']}: {len(articles)} seen, {n} inserted")
            except Exception as exc:
                # One bank's bad query or API failure must not starve the
                # banks after it. Its watermark stays put, so the missed
                # window is retried next run.
                conn.rollback()
                failed.append(bank["bank_id"])
                print(f"{bank['bank_id']}: FAILED: {exc}", file=sys.stderr)
        db.write_heartbeat(conn, "poll_newsapi", seen, inserted,
                           time.monotonic() - started, not failed)
    except Exception:
        try:
            conn.rollback()
            db.write_heartbeat(conn, "poll_newsapi", seen, inserted, time.monotonic() - started, False)
        except Exception:
            pass  # never mask the original failure
        raise
    finally:
        conn.close()
    if failed:
        sys.exit(f"poll_newsapi: failed banks: {', '.join(failed)}")


if __name__ == "__main__":
    sys.exit(main())