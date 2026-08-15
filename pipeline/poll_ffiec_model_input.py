"""FFIEC Call Report → index_model_input, the 50 fields the frozen model reads.

The existing FFIEC collector (unified_ffiec_fdic_dataset) keeps nine derived
columns and writes fact_call_report. This one keeps the 50 raw MDRM items the
fundamentals model was frozen on and writes 014's table. They are separate on
purpose: fact_call_report is a general-purpose fact table several people read,
while this is a model input contract that must not drift when that table's
definitions change.

The field list is not hardcoded here. It comes from
index/fundamentals/frozen_params.json["raw_fields"], so the collector cannot
fall out of step with the model — re-freeze the model and the collector follows.

Prefix coalescing is extract.py's, imported rather than reimplemented. The
RCON/RCFN split is the whole difficulty of this data (§8.3 of the report) and
having two implementations of it is how the two would diverge.

By default this collects the latest quarter FFIEC publishes. --backfill walks
every quarter from FFIEC_START, for the initial load.

    python -m pipeline.poll_ffiec_model_input
    python -m pipeline.poll_ffiec_model_input --backfill --start 20170101
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "index" / "fundamentals"))

from pipeline import db

FROZEN = ROOT / "index" / "fundamentals" / "frozen_params.json"
CALL_FACT = ROOT / "unified_ffiec_fdic_dataset" / "tables" / "fact_call_report.csv"
JOB = "poll_ffiec_model_input"


def log(m):
    print(m, flush=True)


def raw_fields() -> list[str]:
    return json.load(open(FROZEN))["raw_fields"]


def cert_lookup() -> dict:
    """(rssd_id, report_date) -> fdic_cert_number.

    Keyed on both, not on rssd alone: 29 RSSDs carry more than one certificate
    across their history because of mergers, and only the pair resolves to one.
    Missing entries leave the column NULL rather than dropping the row — a filer
    with no certificate on record should still be collected and scored, it just
    will not join downstream.
    """
    if not CALL_FACT.exists():
        log(f"WARN {CALL_FACT} not found — fdic_cert_number will be NULL")
        return {}
    c = pd.read_csv(CALL_FACT, usecols=["rssd_id", "report_date", "fdic_cert_number"])
    c = c.dropna(subset=["rssd_id", "fdic_cert_number"])
    c["q"] = pd.to_datetime(c.report_date).dt.strftime("%Y%m%d")
    return {(int(r.rssd_id), r.q): int(r.fdic_cert_number) for r in c.itertuples()}


def quarter_rows(zip_path: str, quarter: str, fields: list[str], certs: dict):
    """One quarter's archive -> rows ready for index_model_input."""
    import extract

    df = extract.quarter_frame(zip_path, quarter)
    if df is None:
        return None
    present = [f for f in fields if f in df.columns]
    missing = [f for f in fields if f not in df.columns]
    if missing:
        # Not fatal: an item can be absent from one quarter's filing and the
        # model imputes it with the frozen training median. Loud, because a long
        # list means the field set has moved and the model needs re-freezing.
        log(f"    {quarter}: {len(missing)} of {len(fields)} fields absent: "
            f"{missing[:6]}{' ...' if len(missing) > 6 else ''}")

    out = df[["IDRSSD"] + present].copy()
    out = out.rename(columns={"IDRSSD": "rssd_id", **{f: f.lower() for f in present}})
    out["rssd_id"] = out.rssd_id.astype("int64")
    out["report_date"] = f"{quarter[:4]}-{quarter[4:6]}-{quarter[6:]}"
    out["fdic_cert_number"] = [certs.get((r, quarter)) for r in out.rssd_id]
    for f in missing:
        out[f.lower()] = None
    return out


def upsert(conn, rows: pd.DataFrame, fields: list[str]) -> int:
    """Idempotent by (rssd_id, report_date) — re-running a quarter overwrites it.

    FFIEC amends past filings, so a re-run must replace rather than skip:
    'already collected' is not the same as 'unchanged'.
    """
    cols = ["rssd_id", "report_date", "fdic_cert_number"] + [f.lower() for f in fields]
    # astype(object) first. On a float64 column .where(cond, None) coerces the
    # fill value back to NaN to preserve the dtype, and psycopg dumps NaN as the
    # literal 'nan' — which Postgres numeric accepts, so the write succeeds and
    # stores NaN where NULL was meant. Object dtype keeps None as None.
    rows = rows[cols].astype(object).where(pd.notna(rows[cols]), None)
    placeholders = ",".join(["%s"] * len(cols))
    updates = ",".join(f"{c}=EXCLUDED.{c}" for c in cols[2:])
    sql = (f"INSERT INTO index_model_input ({','.join(cols)}, collected_at) "
           f"VALUES ({placeholders}, now()) "
           f"ON CONFLICT (rssd_id, report_date) DO UPDATE SET "
           f"{updates}, collected_at=now()")
    with conn.cursor() as cur:
        cur.executemany(sql, [tuple(r) for r in rows.itertuples(index=False)])
    return len(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backfill", action="store_true",
                    help="every quarter from --start, not just the latest")
    ap.add_argument("--start", default=os.environ.get("FFIEC_START", "20170101"),
                    help="earliest quarter for --backfill (YYYYMMDD)")
    ap.add_argument("--dry-run", action="store_true",
                    help="parse and report, write nothing")
    args = ap.parse_args()

    from ffiec_data_collector import FFIECDownloader, FileFormat, Product

    fields = raw_fields()
    log(f"{len(fields)} raw fields from {FROZEN.name}")
    certs = cert_lookup()
    log(f"{len(certs):,} (rssd, quarter) -> certificate pairs")

    dl = FFIECDownloader()
    available = sorted(q for q in dl.get_bulk_data_sources_cdr()["available_quarters"]
                       if q >= args.start)
    quarters = available if args.backfill else available[-1:]
    log(f"collecting {len(quarters)} quarter(s): {quarters[0]} ~ {quarters[-1]}")

    conn = None if args.dry_run else db.connect()
    seen = written = 0
    failed: list[str] = []
    started, ok = time.monotonic(), False
    try:
        for i, q in enumerate(quarters, 1):
            # Per-quarter isolation. A backfill walks 37 archives and any one of
            # them can fail transiently; the run should log it and carry on, the
            # same way it already tolerates a failed download. Commits are per
            # quarter and the upsert is idempotent, so a partial run is safe to
            # repeat.
            try:
                r = dl.download(Product.CALL_SINGLE, q, FileFormat.TSV)
                if not r.success:
                    log(f"  [{i}/{len(quarters)}] {q}: DOWNLOAD FAILED")
                    failed.append(q)
                    continue
                try:
                    rows = quarter_rows(r.file_path, q, fields, certs)
                finally:
                    try:
                        os.remove(r.file_path)      # keep disk flat
                    except OSError as e:
                        log(f"    {q}: could not remove {r.file_path}: {e}")
                if rows is None or rows.empty:
                    log(f"  [{i}/{len(quarters)}] {q}: NO PARSABLE SCHEDULES")
                    failed.append(q)
                    continue
                seen += len(rows)
                n_cert = int(rows.fdic_cert_number.notna().sum())
                if args.dry_run:
                    log(f"  [{i}/{len(quarters)}] {q}: {len(rows):,} banks "
                        f"({n_cert:,} with a certificate) — dry run, not written")
                    continue
                written += upsert(conn, rows, fields)
                conn.commit()
                log(f"  [{i}/{len(quarters)}] {q}: {len(rows):,} banks "
                    f"({n_cert:,} with a certificate) upserted")
            except Exception as e:
                log(f"  [{i}/{len(quarters)}] {q}: ERROR {type(e).__name__}: {e}")
                failed.append(q)
                if conn:
                    conn.rollback()     # leave the session usable for the next
        ok = not failed
    finally:
        if conn:
            # Rollback first: db.connect() is not autocommit, so after any failed
            # statement the session is aborted and every later statement — the
            # heartbeat included — raises InFailedSqlTransaction. Without this
            # the failure path loses exactly the heartbeat that reports it.
            try:
                conn.rollback()
                db.write_heartbeat(conn, JOB, seen, written,
                                   time.monotonic() - started, ok)
            except Exception as e:
                log(f"WARN heartbeat not written: {type(e).__name__}: {e}")
            finally:
                conn.close()
    if failed:
        log(f"\nfailed quarters ({len(failed)}): {failed}")
    log(f"done: {seen:,} rows seen, {written:,} written")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
