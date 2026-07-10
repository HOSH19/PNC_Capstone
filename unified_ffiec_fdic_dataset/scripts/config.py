"""Shared configuration and schema headers for the dataset builder."""

from pathlib import Path

API_BASE = "https://api.fdic.gov/banks"
HERE = Path(__file__).resolve().parent.parent
TABLES = HERE / "tables"

START_YEAR = 2008
END_QUARTER = "20260331"
ASSET_THRESHOLD = 1_000_000
API_LIMIT = 10000

DIM_BANK_HEADER = [
    "fdic_cert_number",
    "rssd_id",
    "bank_name",
    "city",
    "state",
    "active_status",
    "total_assets",
    "charter_type",
    "established_date",
    "last_updated",
]

CALL_REPORT_HEADER = [
    "fdic_cert_number",
    "rssd_id",
    "report_date",
    "total_assets",
    "total_deposits",
    "tier1_capital_ratio",
    "total_capital_ratio",
    "npl_ratio",
    "loan_loss_allowance_ratio",
    "liquidity_ratio",
    "securities_unrealized_loss",
    "cre_loans",
]

DISTRESS_HEADER = [
    "fdic_cert_number",
    "failure_date",
    "bank_name",
    "city",
    "state",
    "acquiring_institution",
    "event_type",
    "distress_label",
]

BANK_QUARTER_HEADER = [
    "fdic_cert_number",
    "quarter_end_date",
    "total_assets",
    "total_deposits",
    "tier1_capital_ratio",
    "total_capital_ratio",
    "npl_ratio",
    "loan_loss_allowance_ratio",
    "liquidity_ratio",
    "securities_unrealized_loss",
    "cre_loans",
    "distress_within_4q",
    "distress_within_8q",
    "days_to_distress",
]
