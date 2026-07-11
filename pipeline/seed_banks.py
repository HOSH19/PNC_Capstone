"""Upsert db/seed/banks.csv into the bank table.

Idempotent: run as the first workflow step, so adding a bank is one CSV row
plus a re-run — no code changes. The aliases column is semicolon-separated.
"""

import csv
from pathlib import Path

from pipeline import db

SEED_CSV = Path(__file__).resolve().parent.parent / "db" / "seed" / "banks.csv"

COLUMNS = (
    "bank_id", "holding_name", "bank_legal_name", "cik", "ticker", "fdic_cert",
    "rssd_id", "gdelt_query", "aliases", "is_live", "is_backtest", "notes",
)


def load_rows() -> list[dict]:
    rows = []
    with open(SEED_CSV, newline="") as f:
        for r in csv.DictReader(f):
            r["aliases"] = [a.strip() for a in r["aliases"].split(";") if a.strip()]
            r["fdic_cert"] = int(r["fdic_cert"]) if r["fdic_cert"] else None
            r["rssd_id"] = int(r["rssd_id"]) if r["rssd_id"] else None
            r["is_live"] = r["is_live"].lower() == "true"
            r["is_backtest"] = r["is_backtest"].lower() == "true"
            rows.append(r)
    return rows


def main() -> None:
    rows = load_rows()
    cols = ", ".join(COLUMNS)
    params = ", ".join(f"%({c})s" for c in COLUMNS)
    updates = ", ".join(f"{c} = EXCLUDED.{c}" for c in COLUMNS if c != "bank_id")
    with db.connect() as conn, conn.cursor() as cur:
        cur.executemany(
            f"INSERT INTO bank ({cols}) VALUES ({params}) "
            f"ON CONFLICT (bank_id) DO UPDATE SET {updates}",
            rows,
        )
        conn.commit()
    print(f"seeded {len(rows)} banks from {SEED_CSV.name}")


if __name__ == "__main__":
    main()
