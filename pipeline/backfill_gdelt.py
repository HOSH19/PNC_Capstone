"""One-off GDELT backfill for 2020-2024 (docs/roles/jiwon.md §5).

The live corpus starts 2026-04-13 but the distress events run 2017-2024,
so the backtest has zero scorable history. This walks the same 104 seed
banks forward through 2020-2024 in quarterly windows and writes the same
source='gdelt' rows into raw_item — no new tables, no bank-set expansion,
no labeling (the fine-tuned FinBERT scores these on CPU).

Everything GDELT-specific is reused from poll_gdelt: fetch_window (with
its 250-row bisect), to_rows, the normalized-title dedup. This module
only adds the calendar walk, the job chunking, and the resume bookkeeping.

Run it from the backfill_gdelt.yml workflow, not from ingest.yml — it is
manual and finite, not a 6-hourly poller.

    python -m pipeline.backfill_gdelt --start 2020-01-01 --end 2025-01-01 \
        --bank-slice 0/4
"""

import argparse
import sys
import time
from datetime import UTC, datetime, timedelta

from pipeline import db
from pipeline.http import RetriesExhausted
from pipeline.poll_gdelt import fetch_window, to_rows

# calibration: the sizing probe hit 429s at the poller's old 8 s spacing and
# succeeded at 25 s, but GDELT has since got stricter — see poll_gdelt's own
# note. 25 s has never been cleanly tested against the current API alone, so
# treat this as the starting guess and override it with --throttle-s rather
# than editing the file. Spacing multiplies the whole run here (thousands of
# requests, unlike the poller's ~104), so the minimum that works is worth
# finding, but only from a run that had GDELT to itself.
THROTTLE_S = 25.0

# One request covers a quarter unless it overflows 250 rows, in which case
# fetch_window bisects. ~20 windows x 104 banks ~= 2.1k requests ~= 15 h at
# 25 s, which is the run-length estimate the chunking below is sized against.
WINDOW = timedelta(days=90)

# Progress lives under its own watermark source. The live poller reads and
# writes ('gdelt', bank_id) as "newest article already seen"; writing a 2021
# timestamp there would rewind live polling five years and re-fetch the whole
# corpus on the next scheduled run. ('gdelt_backfill', bank_id) is a separate
# primary key in the same table — free text, no schema change — and nothing
# else reads it.
WATERMARK_SOURCE = "gdelt_backfill"

# Actions kills a job at 6 h. Stop early, on a window boundary, with the
# watermark committed, so the next run resumes instead of redoing work. The
# 60 min of headroom is for the window in flight: a heavily-bisected quarter
# can be dozens of 25 s requests, and the deadline is only checked between
# windows.
DEFAULT_BUDGET_MIN = 300


def parse_day(value: str) -> datetime:
    return datetime.fromisoformat(value).replace(tzinfo=UTC)


def iter_windows(
    start: datetime, end: datetime, step: timedelta = WINDOW
) -> list[tuple[datetime, datetime]]:
    """Chronological windows of at most `step` tiling [start, end)."""
    windows = []
    cursor = start
    while cursor < end:
        nxt = min(cursor + step, end)
        windows.append((cursor, nxt))
        cursor = nxt
    return windows


def bank_slice(banks: list, index: int, total: int) -> list:
    """Round-robin partition, so one job does not inherit every big bank.

    Cost per bank is driven by how often fetch_window bisects, which tracks
    news volume; the seed list is sorted by bank_id, so a contiguous split
    would clump the expensive ones together.
    """
    if not 0 <= index < total:
        raise ValueError(f"--bank-slice index {index} out of range for {total}")
    return banks[index::total]


def resume_start(
    watermark: datetime | None, start: datetime, end: datetime
) -> datetime | None:
    """Where a bank resumes, or None when its range is already backfilled.

    The watermark is the end of the last window fully committed for this
    bank, so a killed job re-runs at worst nothing.
    """
    if watermark is None or watermark <= start:
        return start
    return None if watermark >= end else watermark


def backfill_bank(
    conn,
    bank: dict,
    start: datetime,
    end: datetime,
    deadline: float,
    throttle_s: float = THROTTLE_S,
) -> tuple[int, int, bool]:
    """Walk one bank's windows. Returns (seen, inserted, finished)."""
    seen = inserted = 0
    resume = resume_start(
        db.get_watermark(conn, WATERMARK_SOURCE, bank["bank_id"]), start, end
    )
    if resume is None:
        return 0, 0, True
    for window_start, window_end in iter_windows(resume, end):
        if time.monotonic() > deadline:
            return seen, inserted, False
        articles = fetch_window(
            bank["gdelt_query"], window_start, window_end, throttle_s
        )
        rows = to_rows(bank["bank_id"], articles)
        # Windows are disjoint, but a syndicated story can straddle a window
        # boundary under a different representative URL, which the
        # UNIQUE(source, external_id, bank_id) upsert would not catch.
        known = db.existing_title_hashes(
            conn, "gdelt", bank["bank_id"], [r["title_hash"] for r in rows]
        )
        rows = [r for r in rows if r["title_hash"] not in known]
        n = db.upsert_raw_items(conn, rows)
        db.set_watermark(conn, WATERMARK_SOURCE, bank["bank_id"], window_end)
        seen += len(articles)
        inserted += n
    return seen, inserted, True


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", type=parse_day, default=parse_day("2020-01-01"))
    ap.add_argument(
        "--end", type=parse_day, default=parse_day("2025-01-01"), help="exclusive"
    )
    ap.add_argument(
        "--bank-slice",
        default="0/1",
        help="i/n — this job takes every n-th bank starting at i",
    )
    ap.add_argument(
        "--budget-min",
        type=float,
        default=DEFAULT_BUDGET_MIN,
        help="stop on a window boundary after this many minutes",
    )
    ap.add_argument(
        "--throttle-s",
        type=float,
        default=THROTTLE_S,
        help="seconds between GDELT requests; raise it if the run 429s",
    )
    args = ap.parse_args(argv)
    index, total = (int(x) for x in args.bank_slice.split("/"))
    if args.start >= args.end:
        ap.error("--start must precede --end")

    started = time.monotonic()
    deadline = started + args.budget_min * 60
    seen = inserted = 0
    failed: list[str] = []
    incomplete: list[str] = []
    conn = db.connect()
    try:
        banks = [b for b in db.get_live_banks(conn) if b["gdelt_query"]]
        for bank in bank_slice(banks, index, total):
            try:
                b_seen, b_inserted, finished = backfill_bank(
                    conn, bank, args.start, args.end, deadline, args.throttle_s
                )
                seen += b_seen
                inserted += b_inserted
                if not finished:
                    incomplete.append(bank["bank_id"])
                print(
                    f"{bank['bank_id']}: {b_seen} seen, {b_inserted} inserted"
                    f"{'' if finished else ' (out of time)'}"
                )
            except RetriesExhausted as exc:
                # Being rate-limited is "come back later", which is what
                # `incomplete` already means: the watermark stayed put, so the
                # next run redoes these windows. Failing the job for it would
                # paint every pass red on the one condition the resume design
                # exists to absorb.
                conn.rollback()
                incomplete.append(bank["bank_id"])
                print(f"{bank['bank_id']}: rate-limited, will resume: {exc}")
            except Exception as exc:
                # Same isolation as the poller: one bad query or API failure
                # must not starve the banks after it. Its backfill watermark
                # stays put, so the missed windows are retried next run.
                conn.rollback()
                failed.append(bank["bank_id"])
                print(f"{bank['bank_id']}: FAILED: {exc}", file=sys.stderr)
            if time.monotonic() > deadline:
                incomplete.append("...budget exhausted, re-run this slice")
                break
        db.write_heartbeat(
            conn,
            "backfill_gdelt",
            seen,
            inserted,
            time.monotonic() - started,
            not failed,
        )
    except Exception:
        try:
            conn.rollback()
            db.write_heartbeat(
                conn,
                "backfill_gdelt",
                seen,
                inserted,
                time.monotonic() - started,
                False,
            )
        except Exception:
            pass  # never mask the original failure
        raise
    finally:
        conn.close()
    if incomplete:
        # Not an error: re-running the same command picks up where this left
        # off. Exit 0 so the workflow step stays green.
        print(f"incomplete, re-run to resume: {', '.join(incomplete)}")
    if failed:
        sys.exit(f"backfill_gdelt: failed banks: {', '.join(failed)}")


if __name__ == "__main__":
    sys.exit(main())
