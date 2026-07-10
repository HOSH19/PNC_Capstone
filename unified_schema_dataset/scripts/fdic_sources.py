"""FDIC-backed dimension and distress-table refresh logic."""

import json
import time
import urllib.parse
import urllib.request
from datetime import date

from config import API_BASE, API_LIMIT, ASSET_THRESHOLD, DIM_BANK_HEADER, DISTRESS_HEADER, TABLES
from utils import integer, iso_from_mdy, write_csv


def api_get(endpoint, params):
    """Fetch one FDIC endpoint and return flattened records."""
    url = f"{API_BASE}/{endpoint}?" + urllib.parse.urlencode(params)
    request = urllib.request.Request(url, headers={"User-Agent": "ucla-capstone/1.0"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = json.loads(response.read().decode("utf-8"))
            return [row["data"] for row in payload.get("data", [])]
        except Exception as exc:
            if attempt == 3:
                raise
            print(f"  retry {attempt + 1} for {endpoint}: {exc}")
            time.sleep(2 * (attempt + 1))
    return []


def classify_event_type(restype):
    """Map FDIC resolution types into the simplified project label set."""
    text = (restype or "").strip().upper()
    return "assistance" if "ASSIST" in text else "failure"


def fetch_active_institutions():
    """Fetch active institutions for the current bank dimension."""
    print("Fetching FDIC institutions...")
    rows = api_get(
        "institutions",
        {
            "filters": f"ACTIVE:1 AND ASSET:>{ASSET_THRESHOLD}",
            "fields": "NAME,CERT,FED_RSSD,CITY,STALP,ACTIVE,ASSET,CHARTER,ESTYMD",
            "sort_by": "ASSET",
            "sort_order": "DESC",
            "limit": API_LIMIT,
        },
    )
    print(f"  institutions fetched: {len(rows)}")
    return rows


def fetch_failure_rows():
    """Fetch the FDIC failure history for distress labels."""
    print("Fetching FDIC failures...")
    rows = api_get(
        "failures",
        {
            "fields": "CERT,NAME,FAILDATE,CITY,STALP,RESTYPE,SAVR",
            "sort_by": "FAILDATE",
            "sort_order": "DESC",
            "limit": API_LIMIT,
        },
    )
    print(f"  failures fetched: {len(rows)}")
    return rows


def institution_record(row):
    """Normalize one institution row into the bank dimension shape."""
    return {
        "fdic_cert_number": integer(row.get("CERT")),
        "rssd_id": integer(row.get("FED_RSSD")),
        "bank_name": row.get("NAME", ""),
        "city": row.get("CITY", ""),
        "state": row.get("STALP", ""),
        "active_status": True if row.get("ACTIVE") == 1 else False,
        "total_assets": integer(row.get("ASSET")),
        "charter_type": row.get("CHARTER", ""),
        "established_date": iso_from_mdy(row.get("ESTYMD")),
    }


def failure_dimension_record(row):
    """Extract minimum bank metadata from failure records."""
    return {
        "fdic_cert_number": integer(row.get("CERT")),
        "rssd_id": None,
        "bank_name": row.get("NAME", ""),
        "city": row.get("CITY", ""),
        "state": row.get("STALP", ""),
        "active_status": False,
        "total_assets": None,
        "charter_type": "",
        "established_date": "",
    }


def merge_bank_record(existing, incoming):
    """Prefer richer values when the same bank appears more than once."""
    if existing is None:
        return incoming.copy()
    merged = existing.copy()
    for key, value in incoming.items():
        if merged.get(key) in ("", None) and value not in ("", None):
            merged[key] = value
    if incoming.get("active_status") is True:
        merged["active_status"] = True
    return merged


def build_dim_bank(active_rows, failure_rows):
    """Rebuild the bank dimension from FDIC institution and failure data."""
    banks = {}
    for row in active_rows:
        record = institution_record(row)
        if record["fdic_cert_number"] is None:
            continue
        banks[record["fdic_cert_number"]] = merge_bank_record(
            banks.get(record["fdic_cert_number"]),
            record,
        )
    for row in failure_rows:
        record = failure_dimension_record(row)
        if record["fdic_cert_number"] is None:
            continue
        banks[record["fdic_cert_number"]] = merge_bank_record(
            banks.get(record["fdic_cert_number"]),
            record,
        )

    today = date.today().isoformat()
    rows = []
    for cert in sorted(banks):
        bank = banks[cert]
        rows.append(
            [
                bank["fdic_cert_number"],
                bank["rssd_id"],
                bank["bank_name"] or f"Unknown Bank {bank['fdic_cert_number']}",
                bank["city"],
                bank["state"],
                bank["active_status"],
                bank["total_assets"],
                bank["charter_type"],
                bank["established_date"],
                today,
            ]
        )

    write_csv(TABLES / "dim_bank.csv", DIM_BANK_HEADER, rows)
    print(f"  dim_bank rows written: {len(rows)}")


def distress_event_record(row):
    """Map one FDIC failure event into the distress table shape."""
    return [
        integer(row.get("CERT")),
        iso_from_mdy(row.get("FAILDATE")),
        row.get("NAME", ""),
        row.get("CITY", ""),
        row.get("STALP", ""),
        "",
        classify_event_type(row.get("RESTYPE")),
        1,
    ]


def build_fact_distress_event(failure_rows):
    """Rebuild the distress event table from FDIC failures."""
    rows = [distress_event_record(row) for row in failure_rows]
    write_csv(TABLES / "fact_distress_event.csv", DISTRESS_HEADER, rows)
    print(f"  fact_distress_event rows written: {len(rows)}")
