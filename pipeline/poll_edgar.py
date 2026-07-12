"""Incremental SEC EDGAR poller.

Per bank CIK: GET data.sec.gov/submissions/CIK##########.json (SEC blocks
requests without a User-Agent contact email). The response's filings.recent
is COLUMNAR parallel arrays — zip by index. The API returns full history, so
incrementality is client-side: keep filings newer than the watermark only.

Forms kept: 8-K, 10-Q, 10-K (+ /A). For 8-Ks only, the primary document is
fetched, HTML-stripped, and the first ~4000 chars stored as text_excerpt.

New incremental pollers: copy the main() skeleton from poll_gdelt.py (the
canonical template) rather than this file — EDGAR's client-side date filter
is a source-specific quirk. Full checklist: RUNBOOK.md §6.
"""

import re
import sys
import time
from datetime import UTC, date, datetime, timedelta
from html.parser import HTMLParser

import requests

from pipeline import db

SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_nodash}/{doc}"
FORMS = {"8-K", "8-K/A", "10-Q", "10-Q/A", "10-K", "10-K/A"}
# EDGAR first run looks back further than GDELT's 72h: big banks may not file
# anything in 3 days, and the DoD needs a non-empty first run.
FIRST_RUN_LOOKBACK = timedelta(days=90)
EXCERPT_CHARS = 4000
THROTTLE_S = 1.0
MAX_RETRIES = 5

_last_request_at = 0.0


def _headers() -> dict:
    import os
    email = os.environ["SEC_USER_AGENT_EMAIL"]
    return {"User-Agent": f"PNC-Capstone academic research ({email})"}


def _throttled_get(url: str) -> requests.Response:
    global _last_request_at
    for attempt in range(MAX_RETRIES):
        wait = THROTTLE_S - (time.monotonic() - _last_request_at)
        if wait > 0:
            time.sleep(wait)
        _last_request_at = time.monotonic()
        resp = requests.get(url, headers=_headers(), timeout=60)
        if resp.status_code in (403, 429) or resp.status_code >= 500:
            time.sleep(2 ** attempt)
            continue
        resp.raise_for_status()
        return resp
    raise RuntimeError(f"SEC still failing after {MAX_RETRIES} retries: {resp.status_code} {url}")


class _TextExtractor(HTMLParser):
    _SKIP = {"script", "style"}

    def __init__(self):
        super().__init__()
        self._chunks: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in self._SKIP:
            self._skip_depth += 1

    def handle_endtag(self, tag):
        if tag in self._SKIP and self._skip_depth:
            self._skip_depth -= 1

    def handle_data(self, data):
        if not self._skip_depth:
            self._chunks.append(data)

    def text(self) -> str:
        return re.sub(r"\s+", " ", " ".join(self._chunks)).strip()


def strip_html(html: str) -> str:
    parser = _TextExtractor()
    parser.feed(html)
    return parser.text()


def fetch_excerpt(cik: str, accession: str, primary_doc: str) -> str | None:
    url = ARCHIVES_URL.format(
        cik_int=int(cik), acc_nodash=accession.replace("-", ""), doc=primary_doc)
    text = strip_html(_throttled_get(url).text)
    return text[:EXCERPT_CHARS] or None


def iter_recent_filings(submissions: dict):
    """Yield dicts from the columnar parallel arrays in filings.recent."""
    recent = submissions.get("filings", {}).get("recent", {})
    keys = ("accessionNumber", "filingDate", "form", "primaryDocument",
            "items", "acceptanceDateTime")
    columns = [recent.get(k, []) for k in keys]
    for values in zip(*columns):
        yield dict(zip(keys, values))


def to_row(bank: dict, filing: dict) -> dict:
    cik = bank["cik"]
    accession = filing["accessionNumber"]
    form = filing["form"]
    filing_date = date.fromisoformat(filing["filingDate"])
    excerpt = None
    if form.startswith("8-K") and filing["primaryDocument"]:
        excerpt = fetch_excerpt(cik, accession, filing["primaryDocument"])
    return {
        "source": "edgar",
        "external_id": accession,
        "bank_id": bank["bank_id"],
        "published_at": datetime(filing_date.year, filing_date.month, filing_date.day, tzinfo=UTC),
        "title": f"{bank['holding_name']} {form}",
        "url": ARCHIVES_URL.format(
            cik_int=int(cik), acc_nodash=accession.replace("-", ""),
            doc=f"{accession}-index.htm"),
        "domain": "sec.gov",
        "text_excerpt": excerpt,
        "title_hash": None,
        "n_duplicates": 0,
        "meta": {
            "form": form,
            "items": [i.strip() for i in (filing["items"] or "").split(",") if i.strip()],
            "acceptanceDateTime": filing["acceptanceDateTime"],
            "primaryDocument": filing["primaryDocument"],
        },
    }


def main() -> None:
    started = time.monotonic()
    seen = inserted = 0
    conn = db.connect()
    try:
        for bank in db.get_live_banks(conn):
            if not bank["cik"]:
                continue
            run_start = datetime.now(UTC)
            watermark = db.get_watermark(conn, "edgar", bank["bank_id"])
            cutoff = (watermark or run_start - FIRST_RUN_LOOKBACK).date()
            submissions = _throttled_get(SUBMISSIONS_URL.format(cik=bank["cik"])).json()
            rows = [
                to_row(bank, f) for f in iter_recent_filings(submissions)
                if f["form"] in FORMS and date.fromisoformat(f["filingDate"]) >= cutoff
            ]
            n = db.upsert_raw_items(conn, rows)
            db.set_watermark(conn, "edgar", bank["bank_id"], run_start)
            seen += len(rows)
            inserted += n
            print(f"{bank['bank_id']}: {len(rows)} seen, {n} inserted")
        db.write_heartbeat(conn, "poll_edgar", seen, inserted, time.monotonic() - started, True)
    except Exception:
        try:
            conn.rollback()
            db.write_heartbeat(conn, "poll_edgar", seen, inserted, time.monotonic() - started, False)
        except Exception:
            pass  # never mask the original failure
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
