"""Full loader: CFPB Consumer Complaint Database -> Postgres.

Stateless full-loader pattern (RUNBOOK §6). The official Search API
(consumerfinance.gov/.../search/api/v1/) is unreachable -- confirmed both
locally and from a GitHub Actions runner (curl times out, HTTP 000 -- not
a firewall/IP issue on our end, the endpoint itself doesn't respond to
automated requests). Falls back to the bulk CSV export
(files.consumerfinance.gov/ccdb/complaints.csv), which supports Range
requests (confirmed: accept-ranges: bytes) but is ~8.9GB.

Confirmed by hand: the file is NOT sorted by date -- both the first and
last 3KB sampled contain complaints spanning 2018-2024 in no particular
order. There is no "recent" region a smaller Range window could target;
any fixed-offset sample is equally (un)representative of the whole. So
this loader walks the ENTIRE file in fixed-size chunks via repeated Range
requests through pipeline.http.throttled_get -- never a raw streaming
request, since throttled_get loads each response into memory and chunks
must stay small enough for that to be safe (100MB/chunk, ~89 requests for
the full file, ~5-10 min run time).

Known limitation: a chunk boundary can fall inside a quoted narrative
field that itself contains a literal newline, corrupting the row split
exactly at that boundary (~1 row per 89 boundaries, out of ~8M total rows
-- not worth the complexity of a proper quote-aware streaming parser for
this loader). Malformed boundary rows are skipped, not crashed on.

bank_id is resolved by matching the Company field against bank.aliases /
bank_legal_name / holding_name, reusing the same word-boundary +
generic-name-aware strategy as poll_agency_rss.py (see that file's
build_alias_index for why: bare short names and generic industry phrases
produce false matches on plain substring search).

Two filters cut what gets stored (the whole file is still READ either way --
it is neither sorted nor Range-filterable, so we scan all ~16M rows):
  1. date_received >= a rolling LOOKBACK_DAYS window (default ~1 year). The
     scheduled job keeps the trailing year current; older history is a
     separate one-off backfill (run with LOOKBACK_DAYS bumped up), not part
     of the daily run.
  2. Company matches one of the ~104 curated live banks. ~92% of complaints
     are about companies we don't track (credit bureaus, credit unions, debt
     collectors, mortgage servicers) -- dropped.
Result at 1-year window: ~185K rows / ~130MB (matched, narrative kept),
which fits the Supabase free-tier 500MB cap alongside the other tables.
The narrative is deliberately kept: it is the raw text this sentiment-
driven project analyzes, and it is ~63% of per-row bytes. (This loader
never deletes, so the table slowly grows past one year as runs accumulate;
that is fine at ~130MB/yr and revisited if the DB nears its cap.)

Run: python -m pipeline.loaders.load_cfpb
"""

import csv
import re
import sys
import time
from datetime import date, timedelta

from pipeline import db
from pipeline.http import throttled_get

CSV_URL = "https://files.consumerfinance.gov/ccdb/complaints.csv"
CHUNK_BYTES = 100_000_000  # 100MB per Range request -- small enough for throttled_get
LOOKBACK_DAYS = 365  # rolling window kept by the scheduled run; raise it for a backfill
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
            continue  # rolling lookback window (ISO dates sort lexically)
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
    if not rows:
        return 0
    cols = list(rows[0].keys())
    col_list = ", ".join(cols)
    params = ", ".join(f"%({c})s" for c in cols)
    with conn.cursor() as cur:
        cur.executemany(
            f"INSERT INTO cfpb_complaint ({col_list}) VALUES ({params}) "
            "ON CONFLICT (complaint_id) DO NOTHING",
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
        db.write_heartbeat(conn, "load_cfpb", seen, inserted,
                           time.monotonic() - started, True)
    except Exception:
        try:
            conn.rollback()
            db.write_heartbeat(conn, "load_cfpb", seen, inserted,
                               time.monotonic() - started, False)
        except Exception:
            pass  # never mask the original failure
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
