"""Contract tests for the FDIC enforcement CSV poller."""

from pathlib import Path

import pytest

from pipeline import poll_fdic_enforcement as pfe

BANKS = [
    {"bank_id": "pnc", "bank_legal_name": "PNC Bank, National Association",
     "holding_name": "PNC Financial Services Group, Inc.", "aliases": ["PNC", "PNC Bank"]},
    {"bank_id": "wfc", "bank_legal_name": "Wells Fargo Bank, National Association",
     "holding_name": "Wells Fargo & Company", "aliases": ["Wells Fargo"]},
]

HEADER = ",".join(pfe.COLUMNS)


def test_matcher_handles_legal_name_variants():
    m = pfe.build_matcher(BANKS)
    assert m("PNC BANK, NATIONAL ASSOCIATION") == "pnc"
    assert m("Wells Fargo Bank, N.A.") == "wfc"
    assert m("First Community Bank of Nowhere") is None


def test_end_to_end_matches_and_skips(tmp_path, monkeypatch, fake_db, capsys):
    fake_db.banks = BANKS
    csv_path = tmp_path / "fdic.csv"
    csv_path.write_text(
        HEADER + "\n"
        '2026-06-15,"PNC Bank, National Association",Pittsburgh,PA,'
        "Consent Order,FDIC-26-0012b,https://orders.fdic.gov/x.pdf\n"
        "2026-06-20,Tiny Community Bank,Smalltown,KS,Order of Prohibition,FDIC-26-0044,\n")
    monkeypatch.setattr(pfe, "SEED_CSV", csv_path)

    pfe.main()

    assert len(fake_db.rows) == 1
    row = fake_db.rows[0]
    assert (row["source"], row["bank_id"], row["external_id"]) == \
        ("fdic_enforcement", "pnc", "FDIC-26-0012b")
    assert fake_db.heartbeats == [("poll_fdic_enforcement", 1, 1, True)]
    assert "1 outside tracked set" in capsys.readouterr().out


def test_rerun_is_idempotent(tmp_path, monkeypatch, fake_db):
    fake_db.banks = BANKS
    csv_path = tmp_path / "fdic.csv"
    csv_path.write_text(
        HEADER + "\n2026-06-15,PNC Bank,Pittsburgh,PA,Consent Order,FDIC-26-0012b,\n")
    monkeypatch.setattr(pfe, "SEED_CSV", csv_path)
    pfe.main()
    pfe.main()
    assert len(fake_db.rows) == 1
    assert fake_db.heartbeats[1][2] == 0


def test_header_drift_fails(tmp_path, monkeypatch):
    csv_path = tmp_path / "fdic.csv"
    csv_path.write_text("wrong,header\n")
    monkeypatch.setattr(pfe, "SEED_CSV", csv_path)
    with pytest.raises(SystemExit, match="header mismatch"):
        pfe.load_rows()


def test_real_seed_csv_parses():
    assert isinstance(pfe.load_rows(), list)
