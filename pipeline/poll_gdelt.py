"""Incremental GDELT DOC 2.0 poller.

Per live bank: fetch artlist for the window [watermark - 15 min, now].
A full page (exactly 250 rows) means the window overflowed — bisect it
recursively. Syndication duplicates are folded before insert via a
normalized-title hash. artlist has no per-article tone; we store only
what it returns.
"""

import hashlib
import re
import sys
import time
from datetime import UTC, datetime, timedelta

import requests

from pipeline import db

API_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
MAX_RECORDS = 250
OVERLAP = timedelta(minutes=15)
FIRST_RUN_LOOKBACK = timedelta(hours=72)
MIN_WINDOW = timedelta(minutes=1)   # bisect guard
THROTTLE_S = 1.0
MAX_RETRIES = 5

_last_request_at = 0.0


def _throttled_get(params: dict) -> requests.Response:
    global _last_request_at
    for attempt in range(MAX_RETRIES):
        wait = THROTTLE_S - (time.monotonic() - _last_request_at)
        if wait > 0:
            time.sleep(wait)
        _last_request_at = time.monotonic()
        resp = requests.get(API_URL, params=params, timeout=60)
        if resp.status_code == 429 or resp.status_code >= 500:
            time.sleep(2 ** attempt)
            continue
        resp.raise_for_status()
        return resp
    raise RuntimeError(f"GDELT still failing after {MAX_RETRIES} retries: {resp.status_code}")


def _fmt(dt: datetime) -> str:
    return dt.strftime("%Y%m%d%H%M%S")


def fetch_window(query: str, start: datetime, end: datetime) -> list[dict]:
    resp = _throttled_get({
        "query": f"({query})",
        "mode": "artlist",
        "format": "json",
        "maxrecords": MAX_RECORDS,
        "startdatetime": _fmt(start),
        "enddatetime": _fmt(end),
        "sort": "datedesc",
    })
    try:
        articles = resp.json().get("articles", [])
    except ValueError:
        # GDELT returns plain-text errors (e.g. malformed query) with HTTP 200
        raise RuntimeError(f"GDELT non-JSON response: {resp.text[:200]}")
    if len(articles) >= MAX_RECORDS and (end - start) > MIN_WINDOW:
        mid = start + (end - start) / 2
        return fetch_window(query, start, mid) + fetch_window(query, mid, end)
    return articles


def normalize_title_hash(title: str) -> str:
    normalized = re.sub(r"[\W_]+", " ", title.lower()).strip()
    return hashlib.sha1(normalized.encode()).hexdigest()


def _parse_seendate(value: str) -> datetime | None:
    try:
        return datetime.strptime(value, "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)
    except (ValueError, TypeError):
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
            "source": "gdelt",
            "external_id": url,
            "bank_id": bank_id,
            "published_at": _parse_seendate(a.get("seendate")),
            "title": a.get("title"),
            "url": url,
            "domain": a.get("domain"),
            "text_excerpt": None,
            "title_hash": title_hash,
            "n_duplicates": 0,
            "meta": {"language": a.get("language")},
        }
    return list(by_hash.values())


def main() -> None:
    started = time.monotonic()
    seen = inserted = 0
    conn = db.connect()
    try:
        for bank in db.get_live_banks(conn):
            if not bank["gdelt_query"]:
                continue
            run_start = datetime.now(UTC)
            watermark = db.get_watermark(conn, "gdelt", bank["bank_id"])
            window_start = (watermark or run_start - FIRST_RUN_LOOKBACK) - OVERLAP
            articles = fetch_window(bank["gdelt_query"], window_start, run_start)
            rows = to_rows(bank["bank_id"], articles)
            n = db.upsert_raw_items(conn, rows)
            db.set_watermark(conn, "gdelt", bank["bank_id"], run_start)
            seen += len(articles)
            inserted += n
            print(f"{bank['bank_id']}: {len(articles)} seen, {n} inserted")
        db.write_heartbeat(conn, "poll_gdelt", seen, inserted, time.monotonic() - started, True)
    except Exception:
        try:
            conn.rollback()
            db.write_heartbeat(conn, "poll_gdelt", seen, inserted, time.monotonic() - started, False)
        except Exception:
            pass  # never mask the original failure
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
