"""GDELT 2020-2024 backfill via the GKG BigQuery tables.

The DOC API cannot serve this. It rate-limits by cumulative volume, not
spacing -- 8 s, 25 s and 60 s all 429'd identically -- and its own error body
says so: "All high-traffic users should switch to our ngram" datasets. The
backfill is ~3,000 requests, so it is exactly the traffic being turned away.
The live poller stays on the DOC API (decision 2026-08-14); only this
historical pull moves.

What changes, and it is not nothing:

  * Rows are matched on the TITLE naming a bank, because GKG has no clean
    company field -- V2Organizations is NLP extraction, mostly empty, and
    measured 0 hits for "Bank of America" on a day when JPMorgan had 177.
    The DOC API instead searched article bodies, so only 9.4% of the rows it
    returned named their own bank in the title. Backfilled rows are ~100%
    self-naming: cleaner for attribution, but a different distribution from
    the corpus the model was trained on. Say so when reporting backtest
    results.
  * Volume is ~313 rows/day against the DOC API's ~1,467, for the same
    reason: the 5x is mostly articles that were never about the bank.

Everything the pipeline consumes is unchanged: source='gdelt', title-only
text (GDELT rows have never had a body -- eligibility.py has always read
title alone), same title_hash dedup, same raw_item schema.

BigQuery is reached through the `bq` CLI rather than a client library, so
this adds no dependency to requirements.txt -- the same reasoning that keeps
vllm and torch out of it. Authenticate once with `gcloud auth login`.

Cost: ~356 GiB for all five years, inside BigQuery's 1 TB monthly free tier.

    python -m pipeline.backfill_gkg --year 2020
"""

import argparse
import csv
import html
import io
import json
import re
import subprocess
import sys
import time
from datetime import UTC, datetime

from pipeline import db
from pipeline.attribution import safe_banks, usable_aliases
from pipeline.poll_agency_rss import build_alias_index, match_banks
from pipeline.poll_gdelt import normalize_title_hash

TABLE = "gdelt-bq.gdeltv2.gkg_partitioned"

# Aliases shorter than this are dropped from the BigQuery pre-filter: "PNC"
# and "BNY" match too much inside a 76k-row/day English corpus. The precise
# per-bank attribution below still uses the full alias list, so a short-alias
# bank is only missed when its title contains no longer form -- rare, and the
# alternative is pulling thousands of false rows to filter locally.
MIN_ALIAS_LEN = 4

QUERY = """
SELECT
  DATE,
  SourceCommonName AS domain,
  DocumentIdentifier AS url,
  REGEXP_EXTRACT(Extras, r'<PAGE_TITLE>(.*?)</PAGE_TITLE>') AS title,
  V2Tone
FROM `{table}`
WHERE _PARTITIONTIME >= TIMESTAMP('{year}-01-01')
  AND _PARTITIONTIME < TIMESTAMP('{next_year}-01-01')
  AND (TranslationInfo IS NULL OR TranslationInfo = '')
  AND REGEXP_CONTAINS(
        REGEXP_EXTRACT(Extras, r'<PAGE_TITLE>(.*?)</PAGE_TITLE>'),
        r'(?i)\\b({aliases})\\b')
"""


def alias_pattern(banks: list[dict]) -> str:
    """Alternation of every alias long enough to pre-filter on."""
    names: set[str] = set()
    for b in banks:
        for value in (b.get("bank_legal_name"), b.get("holding_name")):
            if value and len(value) >= MIN_ALIAS_LEN:
                names.add(value)
        for a in usable_aliases(b):
            if a and len(a.strip()) >= MIN_ALIAS_LEN:
                names.add(a.strip())
    # BigQuery's regex is RE2: escape, and keep the alternation deterministic.
    return "|".join(re.escape(n) for n in sorted(names))


def run_query(sql: str, project: str) -> list[dict]:
    """Execute via the bq CLI and parse its CSV. No client library needed."""
    proc = subprocess.run(
        [
            "bq",
            "query",
            f"--project_id={project}",
            "--use_legacy_sql=false",
            "--format=csv",
            "--max_rows=1000000",
            sql,
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"bq failed: {proc.stderr[-500:]}")
    return list(csv.DictReader(io.StringIO(proc.stdout)))


def parse_date(value: str) -> datetime | None:
    """GKG DATE is YYYYMMDDHHMMSS."""
    try:
        return datetime.strptime(value[:14], "%Y%m%d%H%M%S").replace(tzinfo=UTC)
    except (ValueError, TypeError):
        return None


def to_rows(gkg_rows: list[dict], index: list) -> tuple[list[dict], dict]:
    """GKG rows -> raw_item rows, one per (article, matched bank). Pure.

    A title naming two banks yields a row for each, exactly as the DOC API
    path does when the same story arrives under two queries.
    """
    out: list[dict] = []
    funnel = {"fetched": len(gkg_rows)}
    skipped = {"no_title": 0, "no_date": 0, "no_bank": 0}
    seen: set[tuple[str, str]] = set()
    for r in gkg_rows:
        # Titles arrive HTML-escaped -- "&#xF3;" for ó. Store them readable,
        # because this string is the model's entire input for a GDELT row.
        title = html.unescape(r.get("title") or "").strip()
        published = parse_date(r.get("DATE") or "")
        url = r.get("url")
        if not title or not url:
            skipped["no_title"] += 1
            continue
        if published is None:
            skipped["no_date"] += 1
            continue
        banks = match_banks(title, index)
        if not banks:
            # The BigQuery pre-filter is broader than the per-bank matcher
            # (it ignores the "generic" exclusions), so some rows land here.
            skipped["no_bank"] += 1
            continue
        for bank_id in sorted(banks):
            if (url, bank_id) in seen:
                continue
            seen.add((url, bank_id))
            out.append(
                {
                    "source": "gdelt",
                    "external_id": url,
                    "bank_id": bank_id,
                    "published_at": published,
                    "title": title,
                    "url": url,
                    "domain": r.get("domain"),
                    "text_excerpt": None,
                    "title_hash": normalize_title_hash(title),
                    "n_duplicates": 0,
                    "meta": {
                        "language": "English",
                        "via": "gkg_backfill",
                        "v2tone": r.get("V2Tone"),
                    },
                }
            )
    funnel["skipped"] = skipped
    funnel["rows"] = len(out)
    return out, funnel


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--project", default="c247a-488706")
    ap.add_argument("--dry-run", action="store_true", help="query, insert nothing")
    args = ap.parse_args()

    started = time.monotonic()
    conn = db.connect()
    try:
        banks = db.get_live_banks(conn)
        sql = QUERY.format(
            table=TABLE,
            year=args.year,
            next_year=args.year + 1,
            aliases=alias_pattern(banks),
        )
        print(f"querying {args.year}…", file=sys.stderr)
        gkg = run_query(sql, args.project)
        rows, funnel = to_rows(gkg, build_alias_index(safe_banks(banks)))
        print(json.dumps(funnel, indent=2), file=sys.stderr)
        if args.dry_run:
            for r in rows[:5]:
                d = r["published_at"].date()
                print(f"  {r['bank_id']:6} {d} {r['title'][:70]}")
            print("dry-run: nothing written", file=sys.stderr)
            return
        inserted = 0
        for i in range(0, len(rows), 5000):
            inserted += db.upsert_raw_items(conn, rows[i : i + 5000])
        db.write_heartbeat(
            conn,
            f"backfill_gkg_{args.year}",
            funnel["fetched"],
            inserted,
            time.monotonic() - started,
            True,
        )
        print(f"{args.year}: {inserted} inserted", file=sys.stderr)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
