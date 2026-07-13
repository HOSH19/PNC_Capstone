"""The hand-edited seed CSVs must parse, and bad edits must fail loudly."""

from pathlib import Path

import pytest

from pipeline import seed_banks

HEADER = ",".join(seed_banks.COLUMNS)


def parse(tmp_path, content):
    p = tmp_path / "banks.csv"
    p.write_text(content)
    orig = seed_banks.SEED_CSV
    seed_banks.SEED_CSV = p
    try:
        return seed_banks.load_rows()
    finally:
        seed_banks.SEED_CSV = orig


def test_real_seed_file_is_valid():
    rows = seed_banks.load_rows()
    assert len(rows) >= 100
    ciks = [r["cik"] for r in rows]
    certs = [r["fdic_cert"] for r in rows if r["fdic_cert"]]
    assert len(set(ciks)) == len(ciks), "duplicate CIK in banks.csv"
    assert len(set(certs)) == len(certs), "duplicate fdic_cert in banks.csv"
    assert all(r["is_live"] in (True, False) for r in rows)


def test_ragged_row_is_reported_with_line_number(tmp_path):
    with pytest.raises(SystemExit, match="line 2"):
        parse(tmp_path, HEADER + "\nusb,US Bancorp,US Bank,0000036104,USB,6548\n")


def test_whitespace_numeric_cell_becomes_null(tmp_path):
    rows = parse(tmp_path, HEADER + "\nusb,US Bancorp,,0000036104,USB, ,,q,a,true,false,\n")
    assert rows[0]["fdic_cert"] is None


def test_duplicate_bank_id_and_missing_required_reported_together(tmp_path):
    with pytest.raises(SystemExit) as e:
        parse(tmp_path,
              HEADER + "\npnc,PNC,,,,,,q,a,true,false,\n"
                       "pnc,PNC2,,,,,,q,a,true,false,\n"
                       "usb,,,,,,,q,a,true,false,\n")
    assert "duplicate bank_id" in str(e.value) and "required" in str(e.value)


def test_header_drift_fails(tmp_path):
    with pytest.raises(SystemExit, match="header mismatch"):
        parse(tmp_path, "bank_id,holding_name\npnc,PNC\n")
