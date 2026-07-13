"""FFIEC Call Report download and schedule parsing logic."""

import csv
import zipfile
from pathlib import Path

from ffiec_data_collector import FFIECDownloader, FileFormat, Product

from config import CALL_REPORT_HEADER, TABLES
from utils import (
    first_value,
    integer,
    iso_from_repdte,
    numeric,
    numeric_sum,
    percentage_ratio,
    quarter_sequence,
    write_csv,
)


def ffiec_downloader():
    """Create an FFIEC downloader client."""
    return FFIECDownloader()


def available_ffiec_quarters(downloader, start_year, end_quarter):
    """Return valid FFIEC quarters within the requested date window."""
    available = set(downloader.get_bulk_data_sources_cdr()["available_quarters"])
    requested = quarter_sequence(start_year, end_quarter)
    return [quarter for quarter in requested if quarter in available]


def ffiec_archive_path(downloader, quarter):
    """Download one FFIEC call-report TSV archive and return its path."""
    result = downloader.download(Product.CALL_SINGLE, quarter, FileFormat.TSV)
    if not result.success:
        raise RuntimeError(f"FFIEC download failed for {quarter}: {result.error_message}")
    return Path(result.file_path)


def schedule_member_name(zip_file, schedule_code):
    """Find the member name for a schedule code inside an FFIEC archive."""
    for name in zip_file.namelist():
        if f" Schedule {schedule_code} " in name or f" Bulk {schedule_code} " in name:
            return name
    return None


def read_tab_file(handle):
    """Read a tab-delimited FFIEC file into row lists."""
    text = handle.read().decode("latin1")
    return list(csv.reader(text.splitlines(), delimiter="\t"))


def parse_por_schedule(zip_file):
    """Parse the POR mapping file into IDRSSD->CERT metadata."""
    member = schedule_member_name(zip_file, "POR")
    rows = read_tab_file(zip_file.open(member))
    header = [cell.strip('"') for cell in rows[0]]
    id_idx = header.index("IDRSSD")
    cert_idx = header.index("FDIC Certificate Number")
    output = {}
    for row in rows[1:]:
        if len(row) <= max(id_idx, cert_idx):
            continue
        idrssd = integer(row[id_idx])
        cert = integer(row[cert_idx])
        if idrssd is not None:
            output[idrssd] = {"fdic_cert_number": cert}
    return output


def parse_schedule(zip_file, schedule_code):
    """Parse one FFIEC schedule into IDRSSD-keyed field dictionaries."""
    member = schedule_member_name(zip_file, schedule_code)
    if member is None:
        return {}
    rows = read_tab_file(zip_file.open(member))
    headers = [cell.strip('"') for cell in rows[0]]
    id_index = headers.index("IDRSSD")
    data = {}
    for row in rows[2:]:
        if not row or len(row) <= id_index:
            continue
        idrssd = integer(row[id_index])
        if idrssd is None:
            continue
        data[idrssd] = {
            header: row[idx].strip() if idx < len(row) else ""
            for idx, header in enumerate(headers)
        }
    return data


def clean_up_archive(path, keep_zip):
    """Delete a downloaded FFIEC zip unless retention was requested."""
    if not keep_zip and path.exists():
        path.unlink()


def total_assets_value(rc, rcri):
    """Return total assets using the main balance sheet with capital fallback."""
    return first_value(rc, ["RCON2170", "RCFD2170"]) or first_value(
        rcri,
        ["RCOA2170", "RCFA2170"],
    )


def total_deposits_value(rc):
    """Return total deposits from the balance sheet schedule."""
    return first_value(rc, ["RCON2200", "RCFN2200", "RCFD2200"])


def total_loans_value(rcci):
    """Return total loans and leases net of unearned income."""
    return first_value(rcci, ["RCON2122", "RCFD2122"])


def allowance_value(rc, rcri):
    """Return allowance for credit losses from core schedules."""
    return first_value(rc, ["RCON3123", "RCFD3123"]) or first_value(
        rcri,
        ["RCOAJJ30", "RCFAJJ30", "RCONJJ30", "RCFDJJ30"],
    )


def npl_balance_value(rcn):
    """Return a simple nonperforming-loan proxy from past-due and nonaccrual totals."""
    return numeric_sum(
        first_value(rcn, ["RCON1407", "RCFD1407"]),
        first_value(rcn, ["RCON1403", "RCFD1403"]),
    )


def liquidity_value(rc, assets):
    """Return a simple liquidity proxy from cash and AFS securities."""
    liquid_assets = numeric_sum(
        first_value(rc, ["RCON0071", "RCFD0071"]),
        first_value(rc, ["RCON0081", "RCFD0081"]),
        first_value(rc, ["RCON1773", "RCFD1773"]),
    )
    return percentage_ratio(liquid_assets, assets)


def cre_value(rcci):
    """Return a CRE exposure proxy from major construction and nonfarm categories."""
    return numeric_sum(
        first_value(rcci, ["RCON2746", "RCFD2746"]),
        first_value(rcci, ["RCONF158", "RCFDF158"]),
        first_value(rcci, ["RCONF160", "RCFDF160"]),
        first_value(rcci, ["RCONF161", "RCFDF161"]),
    )


def parse_capital_schedule(zip_file):
    """Parse RC-R Part I across its era-specific names (RCRI / RCRIA+RCRIB / RCR)."""
    data = parse_schedule(zip_file, "RCRI")
    if not data:
        data = parse_schedule(zip_file, "RCRIA")
        for idrssd, record in parse_schedule(zip_file, "RCRIB").items():
            data.setdefault(idrssd, {}).update(record)
    if not data:
        data = parse_schedule(zip_file, "RCR")
    return data


def ratio_value(record, keys):
    """Return a capital ratio in percent from percent- or fraction-style fields."""
    for key in keys:
        raw = (record.get(key) or "").strip()
        value = numeric(raw)
        if value is None:
            continue
        # 2015+ exports carry an explicit percent sign ("11.6340%"); older
        # exports store the same ratios as decimal fractions ("0.1181").
        return round(value, 4) if raw.endswith("%") else round(value * 100, 4)
    return None


def securities_unrealized_value(rcb):
    """Return net unrealized gain/loss on securities (fair value minus cost)."""
    afs_fair = first_value(rcb, ["RCON1773", "RCFD1773"])
    afs_cost = first_value(rcb, ["RCON1772", "RCFD1772"])
    htm_fair = first_value(rcb, ["RCON1771", "RCFD1771"])
    htm_cost = first_value(rcb, ["RCON1754", "RCFD1754"])
    afs = afs_fair - afs_cost if afs_fair is not None and afs_cost is not None else None
    htm = htm_fair - htm_cost if htm_fair is not None and htm_cost is not None else None
    return numeric_sum(afs, htm)


def call_report_row(idrssd, quarter, por, rc, rcci, rcn, rcri, rcb):
    """Build one unified call-report row from FFIEC schedule fragments."""
    por_record = por.get(idrssd, {})
    rc_record = rc.get(idrssd, {})
    rcci_record = rcci.get(idrssd, {})
    rcn_record = rcn.get(idrssd, {})
    rcri_record = rcri.get(idrssd, {})
    rcb_record = rcb.get(idrssd, {})

    total_assets = total_assets_value(rc_record, rcri_record)
    total_loans = total_loans_value(rcci_record)
    allowance = allowance_value(rc_record, rcri_record)
    npl_balance = npl_balance_value(rcn_record)

    return [
        por_record.get("fdic_cert_number"),
        idrssd,
        iso_from_repdte(quarter),
        integer(total_assets),
        integer(total_deposits_value(rc_record)),
        ratio_value(rcri_record, ["RCOA7206", "RCFA7206", "RCFW7206", "RCON7206", "RCFD7206"]),
        ratio_value(rcri_record, ["RCOA7205", "RCFA7205", "RCFW7205", "RCON7205", "RCFD7205"]),
        percentage_ratio(npl_balance, total_loans),
        percentage_ratio(allowance, total_loans),
        liquidity_value(rc_record, total_assets),
        integer(securities_unrealized_value(rcb_record)),
        integer(cre_value(rcci_record)),
    ]


def quarter_call_report_rows(downloader, quarter, keep_zip):
    """Download and parse one FFIEC quarter into unified rows."""
    archive_path = ffiec_archive_path(downloader, quarter)
    try:
        with zipfile.ZipFile(archive_path) as zip_file:
            por = parse_por_schedule(zip_file)
            rc = parse_schedule(zip_file, "RC")
            rcci = parse_schedule(zip_file, "RCCI")
            rcn = parse_schedule(zip_file, "RCN")
            rcri = parse_capital_schedule(zip_file)
            rcb = parse_schedule(zip_file, "RCB")

            idrssds = sorted(set(por) | set(rc) | set(rcci) | set(rcn) | set(rcri))
            rows = [call_report_row(idrssd, quarter, por, rc, rcci, rcn, rcri, rcb) for idrssd in idrssds]
            rows = [row for row in rows if row[0] is not None or row[1] is not None]
            print(f"  {quarter}: {len(rows)} FFIEC rows")
            return rows
    finally:
        clean_up_archive(archive_path, keep_zip)


def build_fact_call_report_from_ffiec(start_year, end_quarter, keep_zip):
    """Replace fact_call_report with real FFIEC call-report data."""
    print("Fetching FFIEC Call Reports...")
    downloader = ffiec_downloader()
    quarters = available_ffiec_quarters(downloader, start_year, end_quarter)
    print(f"  quarter window: {quarters[-1]} to {quarters[0]} ({len(quarters)} quarters)")

    rows = []
    for quarter in quarters:
        rows.extend(quarter_call_report_rows(downloader, quarter, keep_zip))

    write_csv(TABLES / "fact_call_report.csv", CALL_REPORT_HEADER, rows)
    print(f"  fact_call_report rows written: {len(rows)}")
