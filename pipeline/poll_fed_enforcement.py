"""Federal Reserve enforcement actions → raw_item.

The Fed publishes its complete enforcement-action history as one CSV
(no scraping, no API key). Every run re-fetches the whole file and
upserts the rows that match tracked banks — UNIQUE(source, external_id,
bank_id) makes re-runs no-ops, so no watermark is needed.

The "Banking Organization" field carries location suffixes and sometimes
several organizations joined by " and " ("TS Banking Group, Inc., Treynor,
Iowa and TS Contrarian Bancshares, Inc., Treynor, Iowa"); each part is
matched by trying progressively shorter comma-prefixes against the tracked
banks' normalized names. Actions against individuals only, or against
banks outside the tracked set, are skipped — expected, the file goes back
decades and covers every US banking organization.

Run: python -m pipeline.poll_fed_enforcement
"""

import csv
import io
import sys
import time
from datetime import UTC, datetime

from pipeline import db
from pipeline.http import throttled_get
from pipeline.poll_fdic_enforcement import build_matcher

CSV_URL = "https://www.federalreserve.gov/supervisionreg/files/enforcementactions.csv"
EXPECTED_COLUMNS = {"Effective Date", "Banking Organization", "Action", "URL"}


def match_organizations(field: str, match) -> set[str]:
    """Return bank_ids matched anywhere in a Banking Organization field."""
    hits = set()
    for part in field.split(" and "):
        segments = [s.strip() for s in part.split(",")]
        for k in range(len(segments), 0, -1):
            bank_id = match(", ".join(segments[:k]))
            if bank_id:
                hits.add(bank_id)
                break
    return hits


def to_row(bank_id: str, r: dict) -> dict:
    effective = datetime.strptime(r["Effective Date"], "%Y-%m-%d").replace(tzinfo=UTC)
    org = r["Banking Organization"].strip()
    external_id = r["URL"].strip() or f"{r['Effective Date']}|{org}"
    return {
        "source": "fed_enforcement",
        "external_id": external_id,
        "bank_id": bank_id,
        "published_at": effective,
        "title": f"{org} — {r['Action'].strip()}",
        "url": r["URL"].strip() or None,
        "domain": "federalreserve.gov",
        "text_excerpt": None,
        "title_hash": None,
        "n_duplicates": 0,
        "meta": {
            "action": r["Action"].strip(),
            "termination_date": r.get("Termination Date", "").strip(),
            "individual": r.get("Individual", "").strip(),
            "note": r.get("Note", "").strip(),
        },
    }


def main() -> None:
    started = time.monotonic()
    conn = db.connect()
    try:
        match = build_matcher(db.get_live_banks(conn))
        resp = throttled_get(CSV_URL, label="FederalReserve")
        reader = csv.DictReader(io.StringIO(resp.content.decode("utf-8-sig")))
        missing = EXPECTED_COLUMNS - set(reader.fieldnames or ())
        if missing:
            raise RuntimeError(f"Fed CSV is missing columns: {sorted(missing)} "
                               f"(got {reader.fieldnames})")
        rows, skipped = [], 0
        for r in reader:
            org = (r.get("Banking Organization") or "").strip()
            if not org or not (r.get("Effective Date") or "").strip():
                skipped += 1   # individual-only actions have no organization
                continue
            hits = match_organizations(org, match)
            if not hits:
                skipped += 1   # decades of small banks outside the tracked set
                continue
            rows.extend(to_row(bank_id, r) for bank_id in sorted(hits))
        n = db.upsert_raw_items(conn, rows)
        print(f"fed_enforcement: {len(rows)} matched, {skipped} skipped, {n} inserted")
        db.write_heartbeat(conn, "poll_fed_enforcement", len(rows), n,
                           time.monotonic() - started, True)
    except Exception:
        try:
            conn.rollback()
            db.write_heartbeat(conn, "poll_fed_enforcement", 0, 0,
                               time.monotonic() - started, False)
        except Exception:
            pass  # never mask the original failure
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
