"""Persist the attribution gate's verdict onto raw_item.attributed.

Batch job over rows where `attributed IS NULL` (migration 015). The verdict
is `pipeline.attribution`'s title check for GDELT rows; every other source
established the bank at ingest — EDGAR by the filer's CIK, the enforcement
feeds and agency RSS by alias match — so their rows are attributed by
construction and the title check would be wrong to apply (an 8-K's title
rarely names the bank).

A source this module has never heard of raises instead of defaulting either
way: defaulting True would let a future query-fetched source bypass the gate
silently, and False would zero a legitimate one out of the rollup. Same
loud-failure convention as pipeline.eligibility.

Scheduled in .github/workflows/score.yml before scoring; both walk the rows
the last ingest run added. Safe to re-run any time — rows with a verdict are
never revisited.

    python -m pipeline.attribute_items --limit 500000
"""

import argparse
import sys
import time

from pipeline import db
from pipeline.attribution import build_patterns, counts_for_bank

# Sources whose bank link is established at ingest, not by fetching under a
# query. Deliberately a closed list, not "everything except gdelt".
ATTRIBUTED_AT_INGEST = frozenset(
    {"edgar", "fdic_enforcement", "fed_enforcement", "occ_enforcement", "agency_rss"}
)

BATCH = 10_000


def verdict_for(row: dict, patterns: dict) -> bool:
    """The gate's verdict for one row. Pure."""
    if row["source"] in ATTRIBUTED_AT_INGEST:
        return True
    if row["source"] == "gdelt":
        return counts_for_bank(row.get("title"), row["bank_id"], patterns)
    raise RuntimeError(f"no attribution rule for source {row['source']!r}")


def fetch_unattributed(conn, limit: int) -> list[dict]:
    with conn.cursor() as cur:
        cur.execute(
            """SELECT id, source, title, bank_id FROM raw_item
               WHERE attributed IS NULL
               ORDER BY id
               LIMIT %(limit)s""",
            {"limit": limit},
        )
        return cur.fetchall()


def write_verdicts(conn, verdicts: list[tuple[int, bool]]) -> None:
    """One transaction per batch; every fetched row gets a verdict, so the
    NULL scan always advances."""
    yes = [i for i, v in verdicts if v]
    no = [i for i, v in verdicts if not v]
    with conn.cursor() as cur:
        for ids, value in ((yes, True), (no, False)):
            if ids:
                cur.execute(
                    "UPDATE raw_item SET attributed = %s WHERE id = ANY(%s)",
                    (value, ids),
                )
    conn.commit()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=1_000_000)
    args = ap.parse_args()

    started = time.monotonic()
    done = attributed = 0
    conn = db.connect()
    try:
        patterns = build_patterns(db.get_live_banks(conn))
        while done < args.limit:
            rows = fetch_unattributed(conn, min(BATCH, args.limit - done))
            if not rows:
                break
            verdicts = [(r["id"], verdict_for(r, patterns)) for r in rows]
            write_verdicts(conn, verdicts)
            done += len(verdicts)
            attributed += sum(v for _, v in verdicts)
            print(f"  {done} rows, {attributed} attributed", file=sys.stderr)
        db.write_heartbeat(
            conn, "attribute_items", done, attributed, time.monotonic() - started, True
        )
    except Exception:
        try:
            conn.rollback()
            db.write_heartbeat(
                conn,
                "attribute_items",
                done,
                attributed,
                time.monotonic() - started,
                False,
            )
        except Exception:
            pass  # never mask the original failure
        raise
    finally:
        conn.close()
    print(f"{done} rows, {attributed} attributed", file=sys.stderr)


if __name__ == "__main__":
    main()
