import os
import sys
import time
from datetime import UTC, datetime, timedelta

from pipeline import db
from pipeline.http import throttled_get
from pipeline.poll_gdelt import normalize_title_hash

API_URL = "https://www.alphavantage.co/query"
API_KEY = os.environ["ALPHA_VANTAGE_API_KEY"]  # fail fast at import if unset

# Free tier is 25 requests/day (as of mid-2026) rather than a per-minute cap,
# so throttle_s here is mostly about not hammering the API within one run,
# not about satisfying a per-minute limit. Tune to your actual key's tier.
THROTTLE_S = 15.0
MAX_RECORDS = 1000          # NEWS_SENTIMENT `limit`; premium keys allow up to 1000
OVERLAP = timedelta(minutes=15)
FIRST_RUN_LOOKBACK = timedelta(hours=72)
MIN_WINDOW = timedelta(minutes=1)   # bisect guard — matches time_from/time_to's minute granularity


def _fmt(dt: datetime) -> str:
    # NEWS_SENTIMENT's time_from/time_to take minute-resolution timestamps,
    # no seconds.
    return dt.strftime("%Y%m%dT%H%M")


def fetch_window(ticker: str, start: datetime, end: datetime) -> list[dict]:
    resp = throttled_get(API_URL, label="AlphaVantage", throttle_s=THROTTLE_S, params={
        "function": "NEWS_SENTIMENT",
        "tickers": ticker,
        "sort": "LATEST",
        "limit": MAX_RECORDS,
        "time_from": _fmt(start),
        "time_to": _fmt(end),
        "apikey": API_KEY,
    })
    try:
        payload = resp.json()
    except ValueError:
        raise RuntimeError(f"Alpha Vantage non-JSON response: {resp.text[:200]}")

    # Alpha Vantage returns HTTP 200 for rate-limit / bad-param errors too;
    # they show up as top-level keys instead of an HTTP status.
    if "Error Message" in payload:
        raise RuntimeError(f"Alpha Vantage error for {ticker}: {payload['Error Message'][:200]}")
    if "Information" in payload or "Note" in payload:
        # Typically a rate-limit notice. Treat as a failure for this bank;
        # the per-bank try/except leaves the watermark untouched so this
        # window is retried next run instead of being silently skipped.
        raise RuntimeError(f"Alpha Vantage throttled for {ticker}: "
                            f"{(payload.get('Information') or payload.get('Note'))[:200]}")

    articles = payload.get("feed", [])
    if len(articles) >= MAX_RECORDS:
        if (end - start) > MIN_WINDOW:
            mid = start + (end - start) / 2
            return fetch_window(ticker, start, mid) + fetch_window(ticker, mid, end)
        print(f"WARNING: still {len(articles)} articles at minimum window "
              f"{_fmt(start)}-{_fmt(end)}; articles beyond the newest "
              f"{MAX_RECORDS} are dropped (ticker: {ticker})",
              file=sys.stderr)
    return articles


def _parse_time_published(value: str) -> datetime | None:
    try:
        return datetime.strptime(value, "%Y%m%dT%H%M%S").replace(tzinfo=UTC)
    except (ValueError, TypeError):
        return None


def _ticker_sentiment(article: dict, ticker: str) -> dict:
    """Pull out the sentiment block specific to this bank's ticker, if present."""
    for entry in article.get("ticker_sentiment", []):
        if entry.get("ticker") == ticker:
            return entry
    return {}


def to_rows(bank_id: str, ticker: str, articles: list[dict]) -> list[dict]:
    """Dedup syndicated articles (same URL) — Alpha Vantage aggregates from
    many outlets and the same wire story can appear more than once."""
    seen_urls: set[str] = set()
    rows: list[dict] = []
    for a in articles:
        url = a.get("url")
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)
        ts = _ticker_sentiment(a, ticker)
        rows.append({
            "source": "alpha_vantage",
            "external_id": url,
            "bank_id": bank_id,
            "published_at": _parse_time_published(a.get("time_published")),
            "title": a.get("title"),
            "url": url,
            "domain": a.get("source_domain"),
            "text_excerpt": a.get("summary"),
            "title_hash": None,  # set below, mirrors GDELT's normalize step
            "n_duplicates": 0,
            "meta": {
                "source_name": a.get("source"),
                "authors": a.get("authors"),
                "topics": a.get("topics"),
                "overall_sentiment_score": a.get("overall_sentiment_score"),
                "overall_sentiment_label": a.get("overall_sentiment_label"),
                "ticker_relevance_score": ts.get("relevance_score"),
                "ticker_sentiment_score": ts.get("ticker_sentiment_score"),
                "ticker_sentiment_label": ts.get("ticker_sentiment_label"),
            },
        })

    # Reuse GDELT's title-hash approach for the pre-insert dedup step required
    # below (existing_title_hashes), since Alpha Vantage syndicates too.
    for r in rows:
        r["title_hash"] = normalize_title_hash(r["title"] or r["url"])
    return rows


def main() -> None:
    started = time.monotonic()
    seen = inserted = 0
    failed: list[str] = []
    conn = db.connect()
    try:
        for bank in db.get_live_banks(conn):
            # `ticker` is a dedicated column, distinct from the existing
            # `aliases` list (name variants used for text matching elsewhere).
            # NEWS_SENTIMENT needs a real stock symbol, which isn't reliably
            # derivable from aliases (private/subsidiary banks have aliases
            # but no ticker; an alias string isn't guaranteed to be a valid
            # symbol even when it happens to look like one, e.g. 'pnc').
            if not bank["ticker"]:
                continue
            try:
                run_start = datetime.now(UTC)
                watermark = db.get_watermark(conn, "alpha_vantage", bank["bank_id"])
                window_start = (watermark or run_start - FIRST_RUN_LOOKBACK) - OVERLAP
                articles = fetch_window(bank["ticker"], window_start, run_start)
                rows = to_rows(bank["bank_id"], bank["ticker"], articles)
                known = db.existing_title_hashes(
                    conn, "alpha_vantage", bank["bank_id"], [r["title_hash"] for r in rows])
                rows = [r for r in rows if r["title_hash"] not in known]
                n = db.upsert_raw_items(conn, rows)
                db.set_watermark(conn, "alpha_vantage", bank["bank_id"], run_start)
                seen += len(articles)
                inserted += n
                print(f"{bank['bank_id']}: {len(articles)} seen, {n} inserted")
            except Exception as exc:
                conn.rollback()
                failed.append(bank["bank_id"])
                print(f"{bank['bank_id']}: FAILED: {exc}", file=sys.stderr)
        db.write_heartbeat(conn, "poll_alpha_vantage", seen, inserted,
                           time.monotonic() - started, not failed)
    except Exception:
        try:
            conn.rollback()
            db.write_heartbeat(conn, "poll_alpha_vantage", seen, inserted, time.monotonic() - started, False)
        except Exception:
            pass  # never mask the original failure
        raise
    finally:
        conn.close()
    if failed:
        sys.exit(f"poll_alpha_vantage: failed banks: {', '.join(failed)}")


if __name__ == "__main__":
    sys.exit(main())