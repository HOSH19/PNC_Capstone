"""Bulk CSV loader for CFPB complaints -> cfpb_complaint. Two uses, both
run manually (NOT wired into any workflow):

1. Deep history: poll_cfpb.py keeps only a short recent window; run this once
   to load the trailing year (or more) that the day-by-day API path would be
   slow to walk.
2. Narrative refresh: CFPB publishes each complaint's narrative MONTHS after
   it is received (it scrubs PII first) -- see fill rates below. poll_cfpb
   captures the structured row immediately but with an empty narrative, and it
   never revisits old dates. Re-running this over the trailing ~13 months
   (DO UPDATE upsert) backfills narratives as they appear. Worth doing ~monthly
   if the narrative text matters; the bulk CSV itself only regenerates ~monthly.

     narrative fill rate by complaint age (measured via the API):
       ~6 weeks: 1%   ~3.5 months: 3%   ~9 months: 18%   ~2 years: 31%

It downloads the bulk CSV (files.consumerfinance.gov/ccdb/complaints.csv,
~8.9GB) in one pass. Set how far back to keep via LOOKBACK_DAYS, e.g.::

    LOOKBACK_DAYS=395 python -m pipeline.backfill_cfpb   # ~13 months

The bulk CSV is neither sorted nor Range-filterable by date/company, and it
only regenerates ~monthly (so it is NOT a good recurring source -- that is
why the scheduled job uses the API instead). This walks the ENTIRE file in
100MB Range chunks through pipeline.http.throttled_get (never a raw stream --
throttled_get buffers each response, so chunks stay small enough to be safe),
keeping only rows that (1) fall in the LOOKBACK_DAYS window and (2) match one
of the ~104 curated live banks. Everything else is dropped.

Known limitation: a chunk boundary can fall inside a quoted narrative that
contains a literal newline, corrupting the one row split at that boundary
(~1 row per 89 boundaries out of ~16M) -- such rows are skipped, not fatal.

Writes cfpb_complaint by upsert on complaint_id, so it is safe to re-run and
composes with poll_cfpb (whichever sees a complaint first wins; the other
no-ops). Bank matching mirrors poll_agency_rss.py's word-boundary strategy.
"""

import csv
import os
import re
import sys
import time
from datetime import date, timedelta

from pipeline import db
from pipeline.http import throttled_get

CSV_URL = "https://files.consumerfinance.gov/ccdb/complaints.csv"
CHUNK_BYTES = 100_000_000  # 100MB per Range request -- small enough for throttled_get
LOOKBACK_DAYS = int(os.environ.get("LOOKBACK_DAYS", "365"))  # how far back to keep
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"  # blocked without one

HEADER = ["Date received", "Product", "Sub-product", "Issue", "Sub-issue",
          "Consumer complaint narrative", "Company public response", "Company",
          "State", "ZIP code", "Tags", "Consumer consent provided?",
          "Submitted via", "Date sent to company", "Company response to consumer",
          "Timely response?", "Consumer disputed?", "Complaint ID"]


def get_file_size() -> int:
    resp = throttled_get(CSV_URL, headers={"Range": "bytes=0-0", "User-Agent": UA},
                         throttle_s=1.0, label="CFPB:size")
    content_range = resp.headers.get("Content-Range", "")
    m = re.search(r"/(\d+)$", content_range)
    if not m:
        raise RuntimeError(f"could not parse total size from Content-Range: {content_range!r}")
    return int(m.group(1))


def build_company_index(banks: list[dict]) -> list[tuple[str, list[re.Pattern]]]:
    """Same word-boundary + generic-name-aware strategy as poll_agency_rss.py:
    plain substring matching produces false positives on bare short names and
    on banks flagged "generic" in their seed-data notes."""
    out = []
    for b in banks:
        if b.get("notes") and "generic" in b["notes"].lower():
            names = [b.get("holding_name")]
        else:
            names = [b.get("bank_legal_name"), b.get("holding_name")] + list(b.get("aliases") or [])
        patterns = [re.compile(r"\b" + re.escape(n.lower()) + r"\b") for n in names if n]
        if patterns:
            out.append((b["bank_id"], patterns))
    return out


def match_bank(company: str, index: list[tuple[str, list[re.Pattern]]]) -> str | None:
    company_lower = company.lower()
    for bank_id, patterns in index:
        if any(p.search(company_lower) for p in patterns):
            return bank_id
    return None


def parse_chunk(text: str, index: list[tuple[str, list[re.Pattern]]],
                min_date: str) -> tuple[list[dict], str]:
    """Parse complete CSV rows from `text`; return (rows, leftover_tail).

    The last physical line may be incomplete (chunk boundary) -- held back
    and prepended to the next chunk rather than parsed now.
    """
    lines = text.split("\n")
    complete, leftover = lines[:-1], lines[-1]
    rows = []
    for fields in csv.reader(complete, quotechar='"', doublequote=True):
        if len(fields) != len(HEADER) or not fields[-1].strip().isdigit():
            continue  # boundary artifact or malformed row -- skip, don't crash the run
        row = dict(zip(HEADER, fields))
        if row["Date received"] < min_date:
            continue  # lookback window (ISO dates sort lexically)
        bank_id = match_bank(row["Company"], index)
        if bank_id is None:
            continue  # only store complaints about the curated banks (see docstring)
        rows.append({
            "complaint_id": int(row["Complaint ID"]),
            "bank_id": bank_id,
            "company": row["Company"],
            "date_received": row["Date received"] or None,
            "product": row["Product"] or None,
            "sub_product": row["Sub-product"] or None,
            "issue": row["Issue"] or None,
            "sub_issue": row["Sub-issue"] or None,
            "narrative": (row["Consumer complaint narrative"][:4000]
                         if row["Consumer complaint narrative"] else None),
            "state": row["State"] or None,
            "zip_code": row["ZIP code"] or None,
            "submitted_via": row["Submitted via"] or None,
            "company_response": row["Company response to consumer"] or None,
            "timely_response": (row["Timely response?"] == "Yes") if row["Timely response?"] else None,
            "consumer_disputed": row["Consumer disputed?"] or None,
        })
    return rows, leftover


def upsert_complaints(conn, rows: list[dict]) -> int:
    """Upsert on complaint_id with DO UPDATE so a re-run refreshes rows whose
    narrative CFPB has since published (see module docstring on the lag).
    narrative is COALESCE'd so an empty value never wipes one already stored."""
    if not rows:
        return 0
    cols = list(rows[0].keys())
    col_list = ", ".join(cols)
    params = ", ".join(f"%({c})s" for c in cols)
    sets = ["narrative = COALESCE(EXCLUDED.narrative, cfpb_complaint.narrative)",
            "collected_at = now()"]
    sets += [f"{c} = EXCLUDED.{c}" for c in cols if c not in ("complaint_id", "narrative")]
    with conn.cursor() as cur:
        cur.executemany(
            f"INSERT INTO cfpb_complaint ({col_list}) VALUES ({params}) "
            f"ON CONFLICT (complaint_id) DO UPDATE SET {', '.join(sets)}",
            rows,
        )
        n = cur.rowcount
    conn.commit()
    return n


def main() -> None:
    started = time.monotonic()
    seen = inserted = 0
    conn = db.connect()
    try:
        index = build_company_index(db.get_live_banks(conn))
        min_date = (date.today() - timedelta(days=LOOKBACK_DAYS)).isoformat()
        total_size = get_file_size()
        n_chunks = (total_size + CHUNK_BYTES - 1) // CHUNK_BYTES
        print(f"complaints.csv: {total_size:,} bytes, ~{n_chunks} chunks, keeping >= {min_date}")

        leftover = ""
        start = 0
        first = True
        while start < total_size:
            end = min(start + CHUNK_BYTES - 1, total_size - 1)
            resp = throttled_get(CSV_URL, headers={"Range": f"bytes={start}-{end}", "User-Agent": UA},
                                 throttle_s=1.0, label="CFPB:chunk")
            text = resp.content.decode("utf-8", errors="replace")
            if first:
                text = text.split("\n", 1)[1] if "\n" in text else ""  # drop the header row
                first = False
            rows, leftover = parse_chunk(leftover + text, index, min_date)
            n = upsert_complaints(conn, rows)
            seen += len(rows)
            inserted += n
            pct = (end + 1) / total_size * 100
            print(f"  bytes {start}-{end} ({pct:.1f}%): {len(rows)} parsed, {n} inserted")
            start = end + 1
        db.write_heartbeat(conn, "backfill_cfpb", seen, inserted,
                           time.monotonic() - started, True)
    except Exception:
        try:
            conn.rollback()
            db.write_heartbeat(conn, "backfill_cfpb", seen, inserted,
                               time.monotonic() - started, False)
        except Exception:
            pass  # never mask the original failure
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
