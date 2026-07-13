"""FDIC enforcement orders → raw_item, from a hand-maintained CSV.

The ED&O portal (orders.fdic.gov) is a Salesforce SPA with no export or API,
so per DATA_SOURCES.md the staged approach starts manual: once a month, open
the portal's press-release-orders page (linked from the monthly FDIC press
release) and append that month's orders to db/seed/fdic_enforcement.csv.
This module upserts the whole CSV every run — UNIQUE(source, external_id,
bank_id) makes re-runs no-ops, so no watermark is needed and re-running
after editing the CSV is always safe.

Orders name the operating bank, so matching uses bank_legal_name / aliases /
holding_name (normalized). Orders for banks outside the tracked set are
skipped with a log line — expected, since most FDIC orders target small
community banks.

Run: python -m pipeline.poll_fdic_enforcement
"""

import csv
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from pipeline import db

SEED_CSV = Path(__file__).resolve().parent.parent / "db" / "seed" / "fdic_enforcement.csv"
COLUMNS = ("order_date", "institution_name", "city", "state", "order_type",
           "docket_number", "pdf_url")


def normalize(name: str) -> str:
    n = re.sub(r"[\W_]+", " ", name.lower()).strip()
    # Strip legal boilerplate so "PNC Bank, National Association" matches "PNC Bank".
    return re.sub(r"\b(national association|na|n a|ssb|fsb|inc|corp|corporation|company|co)\b",
                  " ", n).strip()


def build_matcher(banks: list[dict]):
    """Map normalized institution names -> bank_id from every name we know."""
    index: dict[str, str] = {}
    for b in banks:
        names = [b.get("bank_legal_name"), b.get("holding_name")] + list(b.get("aliases") or [])
        for name in names:
            if name:
                index[normalize(name)] = b["bank_id"]
    def match(institution: str) -> str | None:
        return index.get(normalize(institution))
    return match


def load_rows() -> list[dict]:
    with open(SEED_CSV, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != list(COLUMNS):
            sys.exit(f"fdic_enforcement.csv header mismatch:\n  expected: {list(COLUMNS)}\n"
                     f"  found:    {reader.fieldnames}")
        return [{k: v.strip() for k, v in r.items() if k} for r in reader]


def to_row(bank_id: str, r: dict) -> dict:
    order_date = datetime.strptime(r["order_date"], "%Y-%m-%d").replace(tzinfo=UTC)
    return {
        "source": "fdic_enforcement",
        "external_id": r["docket_number"] or r["pdf_url"],
        "bank_id": bank_id,
        "published_at": order_date,
        "title": f"{r['institution_name']} — {r['order_type']}",
        "url": r["pdf_url"] or None,
        "domain": "orders.fdic.gov",
        "text_excerpt": None,   # PDF text extraction deferred (DATA_SOURCES.md)
        "title_hash": None,
        "n_duplicates": 0,
        "meta": {"order_type": r["order_type"], "docket": r["docket_number"],
                 "city": r["city"], "state": r["state"]},
    }


def main() -> None:
    started = time.monotonic()
    conn = db.connect()
    try:
        match = build_matcher(db.get_live_banks(conn))
        rows, skipped = [], 0
        for r in load_rows():
            if not (r["docket_number"] or r["pdf_url"]):
                sys.exit(f"row for {r['institution_name']!r} needs a docket_number or pdf_url")
            bank_id = match(r["institution_name"])
            if bank_id is None:
                skipped += 1   # small banks outside the tracked set — expected
                continue
            rows.append(to_row(bank_id, r))
        n = db.upsert_raw_items(conn, rows)
        print(f"fdic_enforcement: {len(rows)} matched, {skipped} outside tracked set, {n} inserted")
        db.write_heartbeat(conn, "poll_fdic_enforcement", len(rows), n,
                           time.monotonic() - started, True)
    except Exception:
        try:
            conn.rollback()
            db.write_heartbeat(conn, "poll_fdic_enforcement", 0, 0,
                               time.monotonic() - started, False)
        except Exception:
            pass  # never mask the original failure
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
