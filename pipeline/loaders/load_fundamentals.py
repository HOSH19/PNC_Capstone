"""Full loader: Shu Han's fundamentals CSVs → Postgres.

The stateless full-loader pattern (RUNBOOK §6): no watermark, each run
replaces every table with the current contents of
unified_ffiec_fdic_dataset/tables/*.csv inside one transaction per table
(TRUNCATE + COPY — atomic, so a failed run leaves the previous data intact).
The CSVs are the interface with Shu Han's module; his code is never imported.

Run: python -m pipeline.loaders.load_fundamentals
"""

import csv
import sys
import time
from pathlib import Path

from pipeline import db

TABLES_DIR = Path(__file__).resolve().parent.parent.parent / "unified_ffiec_fdic_dataset" / "tables"

# table -> (csv file, ordered columns as in CSV and DDL, primary-key columns)
TABLES = {
    "dim_bank": ("dim_bank.csv", (
        "fdic_cert_number", "rssd_id", "bank_name", "city", "state",
        "active_status", "total_assets", "charter_type", "established_date",
        "last_updated"), ("fdic_cert_number",)),
    "fact_call_report": ("fact_call_report.csv", (
        "fdic_cert_number", "rssd_id", "report_date", "total_assets",
        "total_deposits", "tier1_capital_ratio", "total_capital_ratio",
        "npl_ratio", "loan_loss_allowance_ratio", "liquidity_ratio",
        "securities_unrealized_loss", "cre_loans"),
        ("fdic_cert_number", "report_date")),
    "fact_distress_event": ("fact_distress_event.csv", (
        "fdic_cert_number", "failure_date", "bank_name", "city", "state",
        "acquiring_institution", "event_type", "distress_label"),
        ("fdic_cert_number", "failure_date")),
    "fact_bank_quarter": ("fact_bank_quarter.csv", (
        "fdic_cert_number", "quarter_end_date", "total_assets",
        "total_deposits", "tier1_capital_ratio", "total_capital_ratio",
        "npl_ratio", "loan_loss_allowance_ratio", "liquidity_ratio",
        "securities_unrealized_loss", "cre_loans", "distress_within_4q",
        "distress_within_8q", "days_to_distress"),
        ("fdic_cert_number", "quarter_end_date")),
}


def load_table(conn, table: str, filename: str, columns: tuple, pk: tuple) -> int:
    path = TABLES_DIR / filename
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        missing = set(columns) - set(reader.fieldnames or ())
        if missing:
            raise RuntimeError(f"{filename} is missing columns: {sorted(missing)}")
        col_list = ", ".join(columns)
        n = skipped_null_pk = skipped_dup_pk = 0
        seen_pk: set = set()
        with conn.cursor() as cur:
            cur.execute(f"TRUNCATE {table}")
            with cur.copy(f"COPY {table} ({col_list}) FROM STDIN") as copy:
                for r in reader:
                    key = tuple(r[c] for c in pk)
                    # Legacy rows (e.g. pre-1980s failures) can lack the FDIC
                    # cert; without a key they cannot join anything — skip.
                    if any(v == "" for v in key):
                        skipped_null_pk += 1
                        continue
                    if key in seen_pk:
                        skipped_dup_pk += 1
                        continue
                    seen_pk.add(key)
                    # '' -> NULL for every column; Postgres casts the rest.
                    copy.write_row(tuple(r[c] if r[c] != "" else None for c in columns))
                    n += 1
        conn.commit()  # TRUNCATE + COPY commit together: all-or-nothing
        if skipped_null_pk or skipped_dup_pk:
            print(f"{table}: skipped {skipped_null_pk} rows with empty key, "
                  f"{skipped_dup_pk} duplicate-key rows", file=sys.stderr)
        return n


def main() -> None:
    started = time.monotonic()
    total = 0
    conn = db.connect()
    try:
        for table, (filename, columns, pk) in TABLES.items():
            n = load_table(conn, table, filename, columns, pk)
            total += n
            print(f"{table}: {n} rows loaded from {filename}")
        db.write_heartbeat(conn, "load_fundamentals", total, total,
                           time.monotonic() - started, True)
    except Exception:
        try:
            conn.rollback()
            db.write_heartbeat(conn, "load_fundamentals", total, total,
                               time.monotonic() - started, False)
        except Exception:
            pass  # never mask the original failure
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
