"""Incremental GDELT DOC 2.0 poller.

Per live bank: fetch artlist for the window [watermark - 15 min, now].
A full page (exactly 250 rows) means the window overflowed — bisect it
recursively. Syndication duplicates are folded before insert via a
normalized-title hash. artlist has no per-article tone; we store only
what it returns.

This file doubles as the TEMPLATE for new incremental pollers: copy the
main() skeleton (get_live_banks → get_watermark → fetch → to_rows →
upsert_raw_items → set_watermark → write_heartbeat) and swap only the
fetch/transform parts. Full checklist: RUNBOOK.md §6.
"""

import hashlib
import re
import sys
import time
from datetime import UTC, datetime, timedelta

from pipeline import db
from pipeline.http import throttled_get

API_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
MAX_RECORDS = 250
OVERLAP = timedelta(minutes=15)
FIRST_RUN_LOOKBACK = timedelta(hours=72)
MIN_WINDOW = timedelta(minutes=1)   # bisect guard


def _fmt(dt: datetime) -> str:
    return dt.strftime("%Y%m%d%H%M%S")


def fetch_window(query: str, start: datetime, end: datetime) -> list[dict]:
    # GDELT rejects parentheses around anything that is not an OR list
    # ("Parentheses may only be used around OR'd statements").
    wrapped = f"({query})" if " OR " in query else query
    resp = throttled_get(API_URL, label="GDELT", throttle_s=5.0, params={
        "query": wrapped,
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
    if len(articles) >= MAX_RECORDS:
        if (end - start) > MIN_WINDOW:
            mid = start + (end - start) / 2
            return fetch_window(query, start, mid) + fetch_window(query, mid, end)
        # GDELT ingests in 15-minute batches, so seendates cluster on batch
        # boundaries: a full page at the minimum window cannot be split
        # further, and the API has no cursor to page past 250. Anything older
        # than the newest 250 in this window is unavailable — say so loudly
        # instead of dropping it silently.
        print(f"WARNING: still {len(articles)} articles at minimum window "
              f"{_fmt(start)}-{_fmt(end)}; articles beyond the newest "
              f"{MAX_RECORDS} are dropped (query: {query[:80]})",
              file=sys.stderr)
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
    failed: list[str] = []
    conn = db.connect()
    try:
        for bank in db.get_live_banks(conn):
            if not bank["gdelt_query"]:
                continue
            try:
                run_start = datetime.now(UTC)
                watermark = db.get_watermark(conn, "gdelt", bank["bank_id"])
                window_start = (watermark or run_start - FIRST_RUN_LOOKBACK) - OVERLAP
                articles = fetch_window(bank["gdelt_query"], window_start, run_start)
                rows = to_rows(bank["bank_id"], articles)
                # The overlap re-fetch can pick a different representative URL
                # for a story already stored (UNIQUE on external_id wouldn't
                # catch it) — drop rows whose title we already have.
                known = db.existing_title_hashes(
                    conn, "gdelt", bank["bank_id"], [r["title_hash"] for r in rows])
                rows = [r for r in rows if r["title_hash"] not in known]
                n = db.upsert_raw_items(conn, rows)
                db.set_watermark(conn, "gdelt", bank["bank_id"], run_start)
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
        db.write_heartbeat(conn, "poll_gdelt", seen, inserted,
                           time.monotonic() - started, not failed)
    except Exception:
        try:
            conn.rollback()
            db.write_heartbeat(conn, "poll_gdelt", seen, inserted, time.monotonic() - started, False)
        except Exception:
            pass  # never mask the original failure
        raise
    finally:
        conn.close()
    if failed:
        sys.exit(f"poll_gdelt: failed banks: {', '.join(failed)}")


if __name__ == "__main__":
    sys.exit(main())
