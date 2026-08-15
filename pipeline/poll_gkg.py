"""Incremental GDELT poller reading GKG in BigQuery, replacing poll_gdelt.

The DOC API stopped serving us around 2026-08-01: every scheduled run failed,
and 8 s, 25 s and 60 s spacings all 429'd identically, because the limit is on
cumulative volume rather than rate. Its own error body says as much — high
traffic is told to use another dataset. This is that dataset, and it is the
same source the 2020-2024 backfill already uses, so live and historical rows
now come from one place.

Two consequences, both measured rather than assumed:

  * Rows are matched on the title naming a bank. That is what GKG supports —
    V2Organizations is NLP extraction, mostly empty, and returned zero hits
    for "Bank of America" on a day JPMorgan had 177 — and it is also what the
    attribution gate keeps anyway. Raw volume drops (~313 rows/day against
    the DOC API's ~1,467), but *attributed* volume rises: 313 against 87,
    because the DOC API's extra 5x is articles that were never about the bank
    and never counted for one.
  * The labeler is better on this text, not worse. Split the 300 gold rows by
    whether the title names its own bank: kappa 0.678 / macro-F1 0.846 on the
    self-naming half against 0.486 / 0.633 on the rest. The analyst-attribution
    cases that prompts v3 and v4 struggled with live almost entirely in the
    half this corpus drops. Retraining was considered on that evidence and is
    not needed.

What is NOT resolved: the training corpus is still the DOC API one, and the
gold slices are drawn from it, so the quality gate no longer samples what we
collect. It still measures the labeler, which is what it is for; it no longer
measures the corpus. Fix by adding GKG-drawn gold rows.

Watermarks stay under ('gdelt', bank_id) — the same key poll_gdelt used, with
the same meaning ("newest article already seen"), so switching pollers does
not re-fetch history. Rows keep source='gdelt' for the same reason.

Needs the `bq` CLI and `gcloud auth login`; no client library, so nothing is
added to requirements. Roughly 0.18 GiB scanned per day, inside BigQuery's
1 TB monthly free tier.
"""

import sys
import time
from datetime import UTC, datetime, timedelta

from pipeline import db
from pipeline.attribution import safe_banks
from pipeline.backfill_gkg import (
    PROJECT,
    TABLE,
    alias_pattern,
    run_query,
    to_rows,
)
from pipeline.poll_agency_rss import build_alias_index

# GKG partitions land through the day, so a window ending "now" would miss
# rows still arriving for the current partition. Re-reading a whole day is
# cheap (0.18 GiB) and the title_hash dedup absorbs the overlap.
OVERLAP = timedelta(hours=6)
FIRST_RUN_LOOKBACK = timedelta(days=3)

QUERY = """
SELECT
  DATE,
  SourceCommonName AS domain,
  DocumentIdentifier AS url,
  REGEXP_EXTRACT(Extras, r'<PAGE_TITLE>(.*?)</PAGE_TITLE>') AS title,
  V2Tone
FROM `{table}`
WHERE _PARTITIONTIME >= TIMESTAMP('{start}')
  AND (TranslationInfo IS NULL OR TranslationInfo = '')
  AND REGEXP_CONTAINS(
        REGEXP_EXTRACT(Extras, r'<PAGE_TITLE>(.*?)</PAGE_TITLE>'),
        r'(?i)\\b({aliases})\\b')
"""


def window_start(watermarks: list[datetime | None], now: datetime) -> datetime:
    """One query covers every bank, so the window is the oldest watermark.

    Per-bank windows would mean 104 queries where one does; the cost is
    re-reading partitions some banks already have, which the title_hash dedup
    drops on insert.
    """
    seen = [w for w in watermarks if w]
    if not seen:
        return now - FIRST_RUN_LOOKBACK
    return min(seen) - OVERLAP


def main() -> None:
    started = time.monotonic()
    run_start = datetime.now(UTC)
    conn = db.connect()
    try:
        banks = [b for b in db.get_live_banks(conn) if b.get("gdelt_query")]
        marks = [db.get_watermark(conn, "gdelt", b["bank_id"]) for b in banks]
        start = window_start(marks, run_start)
        print(f"querying GKG from {start.date()}", file=sys.stderr)

        sql = QUERY.format(
            table=TABLE,
            start=start.date().isoformat(),
            aliases=alias_pattern(banks),
        )
        index = build_alias_index(safe_banks(banks))
        rows, funnel = to_rows(run_query(sql, PROJECT), index)

        # Group per bank so each one's watermark only advances if its own rows
        # committed -- a failure mid-way must not skip a bank's window.
        by_bank: dict[str, list[dict]] = {}
        for r in rows:
            by_bank.setdefault(r["bank_id"], []).append(r)

        seen = inserted = 0
        failed: list[str] = []
        for bank in banks:
            bank_rows = by_bank.get(bank["bank_id"], [])
            try:
                known = db.existing_title_hashes(
                    conn,
                    "gdelt",
                    bank["bank_id"],
                    [r["title_hash"] for r in bank_rows],
                )
                fresh = [r for r in bank_rows if r["title_hash"] not in known]
                n = db.upsert_raw_items(conn, fresh)
                db.set_watermark(conn, "gdelt", bank["bank_id"], run_start)
                seen += len(bank_rows)
                inserted += n
            except Exception as exc:
                # Same isolation as the pollers: one bank must not starve the
                # rest, and its watermark stays put so the window is retried.
                conn.rollback()
                failed.append(bank["bank_id"])
                print(f"{bank['bank_id']}: FAILED: {exc}", file=sys.stderr)
        print(
            f"{funnel['fetched']} fetched, {seen} matched, {inserted} inserted",
            file=sys.stderr,
        )
        db.write_heartbeat(
            conn, "poll_gkg", seen, inserted, time.monotonic() - started, not failed
        )
    except Exception:
        try:
            conn.rollback()
            db.write_heartbeat(
                conn, "poll_gkg", 0, 0, time.monotonic() - started, False
            )
        except Exception:
            pass  # never mask the original failure
        raise
    finally:
        conn.close()
    if failed:
        sys.exit(f"poll_gkg: failed banks: {', '.join(failed)}")


if __name__ == "__main__":
    main()
