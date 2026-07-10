"""Derived modeling-table logic built from existing fact tables."""

from datetime import date

from config import BANK_QUARTER_HEADER, TABLES
from utils import integer, iso_from_mdy, read_csv_rows, write_csv


def earliest_failure_dates(distress_rows):
    """Return the first observed failure date for each certificate number."""
    first_failure = {}
    for row in distress_rows:
        cert = integer(row.get("fdic_cert_number"))
        failure_date = row.get("failure_date") or iso_from_mdy(row.get("FAILDATE"))
        if cert is None or not failure_date:
            continue
        current = date.fromisoformat(failure_date)
        if cert not in first_failure or current < first_failure[cert]:
            first_failure[cert] = current
    return first_failure


def label_row(call_report_row, first_failure):
    """Create one bank-quarter modeling row with forward-looking labels."""
    cert = integer(call_report_row["fdic_cert_number"])
    quarter_end_date = call_report_row["report_date"]
    distress_within_4q = 0
    distress_within_8q = 0
    days_to_distress = ""

    if cert in first_failure and quarter_end_date:
        quarter_date = date.fromisoformat(quarter_end_date)
        delta_days = (first_failure[cert] - quarter_date).days
        if delta_days >= 0:
            days_to_distress = delta_days
            distress_within_4q = int(delta_days <= 366)
            distress_within_8q = int(delta_days <= 731)

    return [
        call_report_row["fdic_cert_number"],
        quarter_end_date,
        call_report_row["total_assets"],
        call_report_row["total_deposits"],
        call_report_row["tier1_capital_ratio"],
        call_report_row["total_capital_ratio"],
        call_report_row["npl_ratio"],
        call_report_row["loan_loss_allowance_ratio"],
        call_report_row["liquidity_ratio"],
        call_report_row["securities_unrealized_loss"],
        call_report_row["cre_loans"],
        distress_within_4q,
        distress_within_8q,
        days_to_distress,
    ]


def build_fact_bank_quarter():
    """Rebuild the modeling table from the current call-report and distress CSVs."""
    print("Deriving fact_bank_quarter...")
    call_report_rows = read_csv_rows(TABLES / "fact_call_report.csv")
    distress_rows = read_csv_rows(TABLES / "fact_distress_event.csv")
    first_failure = earliest_failure_dates(distress_rows)
    rows = [label_row(row, first_failure) for row in call_report_rows]
    write_csv(TABLES / "fact_bank_quarter.csv", BANK_QUARTER_HEADER, rows)
    print(f"  fact_bank_quarter rows written: {len(rows)}")
