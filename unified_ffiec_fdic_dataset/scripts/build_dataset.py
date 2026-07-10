"""CLI entrypoint for refreshing the unified bank dataset."""

import argparse

from config import END_QUARTER, START_YEAR
from fdic_sources import (
    build_dim_bank,
    build_fact_distress_event,
    fetch_active_institutions,
    fetch_failure_rows,
)
from ffiec_call_reports import build_fact_call_report_from_ffiec
from modeling import build_fact_bank_quarter


def parse_args():
    """Return command-line options for selective table refreshes."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh-dim-bank", action="store_true")
    parser.add_argument("--refresh-distress", action="store_true")
    parser.add_argument("--start-year", type=int, default=START_YEAR)
    parser.add_argument("--end-quarter", default=END_QUARTER)
    parser.add_argument("--keep-ffiec-zips", action="store_true")
    return parser.parse_args()


def main():
    """Refresh the selected tables while preserving existing CSVs by default."""
    args = parse_args()

    if args.refresh_dim_bank:
        active_rows = fetch_active_institutions()
        failure_rows = fetch_failure_rows()
        build_dim_bank(active_rows, failure_rows)

    if args.refresh_distress:
        failure_rows = fetch_failure_rows()
        build_fact_distress_event(failure_rows)

    build_fact_call_report_from_ffiec(
        args.start_year,
        args.end_quarter,
        args.keep_ffiec_zips,
    )
    build_fact_bank_quarter()
    print("Done.")


if __name__ == "__main__":
    main()
