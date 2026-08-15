"""SEC EDGAR 2020-2024 backfill to CSV.

The live poller (poll_edgar.py) reads filings.recent, which caps at the
newest ~1000 filings per CIK — for the big banks that does not reach back
to 2020. This module also walks filings.files, the paginated archive JSONs
under data.sec.gov/submissions/, and concatenates their filing arrays.

Output is a CSV, not DB rows: the backtest corpus is file-based because the
GKG backfill pushed Supabase past its free-tier quota and the historical
rows were deleted as reproducible (scoring/DESIGN.md, 2026-08-14 entries).
`attributed` is the literal "true" for every row — filings are the bank's
own submissions, the same reasoning that puts "edgar" in
attribute_items.ATTRIBUTED_AT_INGEST.

Needs SEC_USER_AGENT_EMAIL set (SEC blocks anonymous clients) and DB access
for the bank list only. Resume: if the output CSV exists, already-fetched
(bank_id, url) pairs are kept and skipped, so an interrupted run continues.

    SEC_USER_AGENT_EMAIL=you@example.edu \
        python -m pipeline.backfill_edgar --to-csv corpus/
"""

import argparse
import csv
import os
import sys
from datetime import UTC, date, datetime

import requests

from pipeline import db
from pipeline.http import throttled_get
from pipeline.poll_edgar import (
    ARCHIVES_URL,
    EXCERPT_CHARS,
    FORMS,
    SUBMISSIONS_URL,
    _headers,
    iter_recent_filings,
    strip_html,
)

# poll_edgar._get spaces requests 1 s apart (pipeline.http default), sized
# for a ~104-request incremental run. This pull is thousands of requests;
# SEC allows 10/s, so space at ~0.12 s instead. Same retry set otherwise.
SLEEP_S = 0.12


def _get(url: str) -> requests.Response:
    """poll_edgar._get with the backfill's pacing; see SLEEP_S above."""
    return throttled_get(
        url,
        headers=_headers(),
        retry_statuses=(403, 429),
        throttle_s=SLEEP_S,
        label="SEC",
    )


def fetch_excerpt(cik: str, accession: str, primary_doc: str) -> str | None:
    """poll_edgar.fetch_excerpt, but through the faster-paced _get above."""
    url = ARCHIVES_URL.format(
        cik_int=int(cik), acc_nodash=accession.replace("-", ""), doc=primary_doc
    )
    return strip_html(_get(url).text)[:EXCERPT_CHARS] or None


def iter_columnar_filings(page: dict):
    """Zip one columnar filing block (filings.recent or an archive page).

    The archive JSONs carry the same parallel arrays at top level that the
    live poller reads under filings.recent; poll_edgar.iter_recent_filings
    is recent-only in name but its zip/length-check logic is exactly this,
    so reuse it by wrapping.
    """
    return iter_recent_filings({"filings": {"recent": page}})


def pages_in_range(files: list[dict], since: date, until: date) -> list[str]:
    """Archive page names whose [filingFrom, filingTo] overlaps [since, until).

    Pure. filings.files metadata carries the date span of each page, so pages
    entirely outside the window are never fetched.
    """
    return [
        f["name"]
        for f in files
        if date.fromisoformat(f["filingFrom"]) < until
        and date.fromisoformat(f["filingTo"]) >= since
    ]


def select_filings(filings, since: date, until: date):
    """Yield filings with form in FORMS and since <= filingDate < until,
    deduped by accession. Pure.

    Called per bank, so the accession dedup is effectively (bank_id,
    accession) — recent and archive pages can overlap at the seam.
    """
    seen: set[str] = set()
    for f in filings:
        if f["form"] not in FORMS:
            continue
        if not (since <= date.fromisoformat(f["filingDate"]) < until):
            continue
        if f["accessionNumber"] in seen:
            continue
        seen.add(f["accessionNumber"])
        yield f


CSV_COLUMNS = ["bank_id", "published_at", "title", "url", "text_excerpt", "attributed"]


def csv_row(bank: dict, filing: dict, excerpt: str | None) -> dict:
    """One CSV row, shaped like poll_edgar.to_row shapes its DB row. Pure."""
    d = date.fromisoformat(filing["filingDate"])
    accession = filing["accessionNumber"]
    return {
        "bank_id": bank["bank_id"],
        # filingDate is a date; midnight UTC, as the live poller stores it.
        "published_at": datetime(d.year, d.month, d.day, tzinfo=UTC).isoformat(),
        "title": f"{bank['holding_name']} {filing['form']}",
        "url": ARCHIVES_URL.format(
            cik_int=int(bank["cik"]),
            acc_nodash=accession.replace("-", ""),
            doc=f"{accession}-index.htm",
        ),
        "text_excerpt": excerpt or "",
        "attributed": "true",
    }


def resume_pairs(rows: list[dict]) -> set[tuple[str, str]]:
    """(bank_id, url) pairs already in the output CSV. Pure."""
    return {(r["bank_id"], r["url"]) for r in rows}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--since", default="2020-01-01")
    ap.add_argument("--until", default="2025-01-01")
    ap.add_argument(
        "--to-csv",
        metavar="DIR",
        required=True,
        help="write edgar_2020_2024.csv to DIR; no DB writes",
    )
    args = ap.parse_args()
    since, until = date.fromisoformat(args.since), date.fromisoformat(args.until)

    os.makedirs(args.to_csv, exist_ok=True)
    path = os.path.join(args.to_csv, "edgar_2020_2024.csv")
    out: list[dict] = []
    if os.path.exists(path):
        with open(path, newline="") as f:
            out = list(csv.DictReader(f))
        print(f"resuming: {len(out)} rows already in {path}", file=sys.stderr)
    done = resume_pairs(out)

    conn = db.connect()
    try:
        banks = [b for b in db.get_live_banks(conn) if b["cik"]]
    finally:
        conn.close()

    submissions_base = SUBMISSIONS_URL.rsplit("/", 1)[0]
    for bank in banks:
        # Same zfill as poll_edgar.main(): company_tickers.json publishes cik
        # as an unpadded integer.
        bank["cik"] = bank["cik"].strip().zfill(10)
        submissions = _get(SUBMISSIONS_URL.format(cik=bank["cik"])).json()
        filings_json = submissions.get("filings", {})
        pages = [filings_json.get("recent", {})]
        for name in pages_in_range(filings_json.get("files", []), since, until):
            pages.append(_get(f"{submissions_base}/{name}").json())
        filings = (f for page in pages for f in iter_columnar_filings(page))
        new = skipped = 0
        for filing in select_filings(filings, since, until):
            row = csv_row(bank, filing, None)
            if (bank["bank_id"], row["url"]) in done:
                skipped += 1
                continue
            # Excerpts exactly as the live poller: 8-K primary docs only, and
            # a 4xx on one document loses the excerpt, not the filing.
            if filing["form"].startswith("8-K") and filing["primaryDocument"]:
                try:
                    row["text_excerpt"] = (
                        fetch_excerpt(
                            bank["cik"],
                            filing["accessionNumber"],
                            filing["primaryDocument"],
                        )
                        or ""
                    )
                except requests.HTTPError as exc:
                    print(
                        f"{bank['bank_id']}: excerpt fetch failed for "
                        f"{filing['accessionNumber']}: {exc}",
                        file=sys.stderr,
                    )
            out.append(row)
            new += 1
        print(f"{bank['bank_id']}: {new} new, {skipped} resumed", file=sys.stderr)
        # Rewrite after every bank so an interrupted run resumes from the
        # last completed bank, not from zero.
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            w.writeheader()
            w.writerows(out)
    print(f"{len(out)} rows -> {path}", file=sys.stderr)


if __name__ == "__main__":
    main()
