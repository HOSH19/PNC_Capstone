"""Incremental poller: CFPB Consumer Complaint Database -> cfpb_complaint.

Uses the CFPB Search API (date_received_min/max), NOT the 8.9GB bulk CSV.
The API is incremental, date-filterable, and fresher -- the bulk CSV only
regenerates ~monthly, so it is unfit for a recurring job (it lives on in
backfill_cfpb.py for one-off deep-history loads). Each run fetches the days
since the last one and upserts by complaint_id, so re-runs are cheap no-ops.

Watermark: the max(date_received) already in cfpb_complaint, minus an
OVERLAP_DAYS window (complaints are published a few days after they are
received, so the newest day is never complete). The shared `watermark`
table is deliberately NOT used here: its bank_id is a NOT NULL FK to bank,
so it cannot key a global, non-per-bank source like this one. On an empty
table the first run only looks back DEFAULT_LOOKBACK_DAYS -- the ~1-year
history is a separate one-off load (backfill_cfpb.py), not this job.

The API returns the entire matching window in one response with no server
paging (a 3-month query is ~340MB and times out), so this fetches ONE DAY
per request and loops -- each day is ~35MB / a few seconds.

Writes cfpb_complaint (its own table), not raw_item: complaints carry rich
structured fields (product / issue / narrative / response) that raw_item
has no columns for. Bank matching reuses poll_agency_rss.py's word-boundary
+ generic-name strategy. Only complaints matching one of the ~104 curated
live banks are stored (~92% of all complaints are about companies we do not
track -- credit bureaus, debt collectors, etc.).

Run: python -m pipeline.poll_cfpb
"""

import re
import sys
import time
from datetime import date, timedelta

from pipeline import db
from pipeline.http import throttled_get

API_URL = "https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/"
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"  # 403 without a browser UA
DEFAULT_LOOKBACK_DAYS = 7  # first run on an empty table; deep history is backfill_cfpb.py
OVERLAP_DAYS = 3           # re-scan recent days -- complaints publish a few days late


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


def fetch_day(day: date) -> list[dict]:
    """One day of complaints (all companies). date_received_max is the next
    day; overlapping day windows are fine -- upsert on complaint_id dedups."""
    resp = throttled_get(API_URL, params={
        "date_received_min": day.isoformat(),
        "date_received_max": (day + timedelta(days=1)).isoformat(),
        "format": "json",
        "no_aggs": "true",
    }, headers={"User-Agent": UA}, throttle_s=1.0, label=f"CFPB:{day}")
    return resp.json()


def to_rows(hits: list[dict], index: list[tuple[str, list[re.Pattern]]]) -> list[dict]:
    rows = []
    for h in hits:
        s = h.get("_source", {})
        company = s.get("company") or ""
        cid = s.get("complaint_id")
        if not company or not cid:
            continue
        bank_id = match_bank(company, index)
        if bank_id is None:
            continue  # only store complaints about the curated banks
        narrative = s.get("complaint_what_happened") or None
        rows.append({
            "complaint_id": int(cid),
            "bank_id": bank_id,
            "company": company,
            "date_received": (s.get("date_received") or "")[:10] or None,
            "product": s.get("product") or None,
            "sub_product": s.get("sub_product") or None,
            "issue": s.get("issue") or None,
            "sub_issue": s.get("sub_issue") or None,
            "narrative": narrative[:4000] if narrative else None,
            "state": s.get("state") or None,
            "zip_code": s.get("zip_code") or None,
            "submitted_via": s.get("submitted_via") or None,
            "company_response": s.get("company_response") or None,
            "timely_response": (s.get("timely") == "Yes") if s.get("timely") else None,
            "consumer_disputed": None,  # not returned by the API (field retired by CFPB in 2017)
        })
    return rows


def upsert_complaints(conn, rows: list[dict]) -> int:
    """Upsert on complaint_id. DO UPDATE (not DO NOTHING) because a complaint's
    fields change after we first see it: company_response/timely progress as
    the case is worked, and -- critically -- the narrative is published months
    later (CFPB scrubs it before release), long after the structured row lands.
    narrative is COALESCE'd so a later empty response never wipes one we already
    captured; the rest take the freshest value."""
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


def start_day(conn) -> date:
    """Resume from the latest complaint we already have, minus the overlap;
    on an empty table, look back DEFAULT_LOOKBACK_DAYS."""
    with conn.cursor() as cur:
        cur.execute("SELECT max(date_received) AS mx FROM cfpb_complaint")
        mx = cur.fetchone()["mx"]
    if mx is None:
        return date.today() - timedelta(days=DEFAULT_LOOKBACK_DAYS)
    return mx - timedelta(days=OVERLAP_DAYS)


def main() -> None:
    started = time.monotonic()
    seen = inserted = 0
    conn = db.connect()
    try:
        index = build_company_index(db.get_live_banks(conn))
        day = start_day(conn)
        today = date.today()
        print(f"polling CFPB from {day} to {today}")
        while day <= today:
            hits = fetch_day(day)
            rows = to_rows(hits, index)
            n = upsert_complaints(conn, rows)
            seen += len(hits)
            inserted += n
            print(f"{day}: {len(hits)} complaints, {len(rows)} bank-matched, {n} upserted")
            day += timedelta(days=1)
        db.write_heartbeat(conn, "poll_cfpb", seen, inserted,
                           time.monotonic() - started, True)
    except Exception:
        try:
            conn.rollback()
            db.write_heartbeat(conn, "poll_cfpb", seen, inserted,
                               time.monotonic() - started, False)
        except Exception:
            pass  # never mask the original failure
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
