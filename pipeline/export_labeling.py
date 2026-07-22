"""Export eligible raw_item rows to a labeling batch CSV (Stage 1).

Pulls labeling-eligible sources, cross-source dedups by title_hash, applies
the shared eligibility filter, and writes labeling_batch_<date>.csv for the
Kaggle Llama round-trip. --dry-run prints the selection funnel and writes
nothing.

Text assembly and eligibility live in pipeline.eligibility (one definition,
shared with serving) — this script never re-implements a filter.

Funnel stages (DESIGN 6-stage funnel, mapped to what is computable in v1):
  total -> unique (title_hash dedup) -> eligible -> selected.
"bank-related" is implicit (raw_item is ingested per bank); "event-related"
is the syndication-noise predicate, currently a deferred pass-through hook
(see eligibility.is_syndication_noise). Its effect shows up as
skipped[syndication_noise] once the predicate lands.
"""

import argparse
import csv
import sys
from collections import Counter

from pipeline import db, eligibility

LABELING_SOURCES = tuple(eligibility.ADAPTERS)  # ('gdelt', 'edgar')
OUT_COLUMNS = (
    "raw_item_id",
    "source",
    "bank_id",
    "published_at",
    "title",
    "text_excerpt",
)
_SELECT_COLUMNS = (
    "id",
    "source",
    "bank_id",
    "published_at",
    "title",
    "text_excerpt",
    "title_hash",
    "meta",
)


def build_batch(rows: list[dict]) -> tuple[list[dict], dict]:
    """Cross-source title_hash dedup + eligibility filter. Pure, no DB.

    rows: raw_item dicts. Returns (selected_rows, funnel_counts).
    """
    funnel: dict = {"total": len(rows)}

    seen_hashes: set[str] = set()
    deduped: list[dict] = []
    for r in rows:
        h = r.get("title_hash")
        if h and h in seen_hashes:
            continue
        if h:
            seen_hashes.add(h)
        deduped.append(r)
    funnel["unique"] = len(deduped)

    selected: list[dict] = []
    skipped: Counter = Counter()
    for r in deduped:
        result = eligibility.check(r)
        if result.eligible:
            selected.append(r)
        else:
            skipped[result.reason] += 1
    funnel["eligible"] = len(selected)
    funnel["skipped"] = dict(skipped)
    return selected, funnel


def fetch_rows(conn, since: str | None = None) -> list[dict]:
    cols = ", ".join(_SELECT_COLUMNS)
    sql = f"SELECT {cols} FROM raw_item WHERE source = ANY(%(sources)s)"
    params: dict = {"sources": list(LABELING_SOURCES)}
    if since:
        sql += " AND published_at >= %(since)s"
        params["since"] = since
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchall()


def write_csv(path: str, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=OUT_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    "raw_item_id": r["id"],
                    "source": r["source"],
                    "bank_id": r["bank_id"],
                    "published_at": r["published_at"],
                    "title": r["title"],
                    "text_excerpt": r["text_excerpt"],
                }
            )


def print_funnel(funnel: dict) -> None:
    print("labeling export funnel:", file=sys.stderr)
    print(f"  total collected   {funnel['total']}", file=sys.stderr)
    print(f"  unique (dedup)    {funnel['unique']}", file=sys.stderr)
    print(f"  eligible          {funnel['eligible']}", file=sys.stderr)
    for reason, n in sorted(funnel["skipped"].items()):
        print(f"    skipped[{reason}]  {n}", file=sys.stderr)
    print(f"  -> selected       {funnel['eligible']}", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="labeling_batch.csv")
    ap.add_argument(
        "--since", default=None, help="only rows published on/after YYYY-MM-DD"
    )
    ap.add_argument(
        "--dry-run", action="store_true", help="print funnel, write nothing"
    )
    args = ap.parse_args()

    conn = db.connect()
    try:
        rows = fetch_rows(conn, since=args.since)
    finally:
        conn.close()

    selected, funnel = build_batch(rows)
    print_funnel(funnel)
    if args.dry_run:
        print("dry-run: no file written", file=sys.stderr)
        return
    write_csv(args.out, selected)
    print(f"wrote {len(selected)} rows to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
