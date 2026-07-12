"""Upsert db/seed/banks.csv into the bank table.

Idempotent: run as the first workflow step, so adding a bank is one CSV row
plus a re-run — no code changes. The aliases column is semicolon-separated.
"""

import csv
import sys
from pathlib import Path

from pipeline import db

SEED_CSV = Path(__file__).resolve().parent.parent / "db" / "seed" / "banks.csv"

COLUMNS = (
    "bank_id", "holding_name", "bank_legal_name", "cik", "ticker", "fdic_cert",
    "rssd_id", "gdelt_query", "aliases", "is_live", "is_backtest", "notes",
)


def load_rows() -> list[dict]:
    """Parse and validate the seed CSV.

    The CSV is hand-edited (RUNBOOK §5), and a crash here skips BOTH pollers
    via the workflow guard — so instead of letting a ragged row or a stray
    space blow up as AttributeError/ValueError, collect every problem with
    its line number and exit with the full list.
    """
    rows, errors, seen_ids = [], [], set()
    with open(SEED_CSV, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != list(COLUMNS):
            sys.exit(f"banks.csv header mismatch:\n  expected: {list(COLUMNS)}\n"
                     f"  found:    {reader.fieldnames}")
        for lineno, r in enumerate(reader, start=2):  # header is line 1
            if None in r or any(v is None for v in r.values()):
                errors.append(f"line {lineno}: has more or fewer cells than "
                              f"the {len(COLUMNS)}-column header")
                continue
            r = {k: v.strip() for k, v in r.items()}
            if not r["bank_id"] or not r["holding_name"]:
                errors.append(f"line {lineno}: bank_id and holding_name are required")
                continue
            if r["bank_id"] in seen_ids:
                errors.append(f"line {lineno}: duplicate bank_id '{r['bank_id']}'")
                continue
            seen_ids.add(r["bank_id"])
            try:
                r["fdic_cert"] = int(r["fdic_cert"]) if r["fdic_cert"] else None
                r["rssd_id"] = int(r["rssd_id"]) if r["rssd_id"] else None
            except ValueError as exc:
                errors.append(f"line {lineno} ({r['bank_id']}): {exc}")
                continue
            r["aliases"] = [a.strip() for a in r["aliases"].split(";") if a.strip()]
            r["is_live"] = r["is_live"].lower() == "true"
            r["is_backtest"] = r["is_backtest"].lower() == "true"
            rows.append(r)
    if errors:
        sys.exit("banks.csv invalid:\n  " + "\n  ".join(errors))
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
