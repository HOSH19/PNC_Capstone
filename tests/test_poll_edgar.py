"""Contract tests for the EDGAR poller."""

from datetime import UTC, datetime

import pytest
import requests

from pipeline import poll_edgar

KEYS = ["accessionNumber", "filingDate", "form", "primaryDocument",
        "items", "acceptanceDateTime"]


def submissions(recent):
    return {"filings": {"recent": recent}}


def columns(n):
    return {k: [f"{k}{i}" for i in range(n)] for k in KEYS}


def test_equal_columns_yield_rows():
    assert len(list(poll_edgar.iter_recent_filings(submissions(columns(3))))) == 3


def test_missing_column_fails_loudly():
    bad = {k: v for k, v in columns(3).items() if k != "items"}
    with pytest.raises(RuntimeError, match="unexpected lengths"):
        list(poll_edgar.iter_recent_filings(submissions(bad)))


def test_all_columns_renamed_fails_loudly():
    renamed = {"accession_number": ["a", "b"], "filing_date": ["d1", "d2"]}
    with pytest.raises(RuntimeError):
        list(poll_edgar.iter_recent_filings(submissions(renamed)))


def test_empty_recent_is_fine():
    assert list(poll_edgar.iter_recent_filings(submissions({}))) == []
    assert list(poll_edgar.iter_recent_filings({})) == []


def filing(form="8-K", doc="doc.htm"):
    return {"accessionNumber": "0000713676-26-000001", "filingDate": "2026-07-01",
            "form": form, "primaryDocument": doc, "items": "7.01,9.01",
            "acceptanceDateTime": "2026-07-01T16:00:00.000Z"}


BANK = {"bank_id": "pnc", "cik": "0000713676", "holding_name": "PNC Financial"}


def test_excerpt_404_keeps_the_filing(monkeypatch, capsys):
    def boom(cik, acc, doc):
        raise requests.HTTPError("404 Client Error")
    monkeypatch.setattr(poll_edgar, "fetch_excerpt", boom)
    row = poll_edgar.to_row(BANK, filing())
    assert row["text_excerpt"] is None
    assert row["meta"]["excerpt_error"].startswith("404")


def test_excerpt_retry_exhaustion_still_fails_the_bank(monkeypatch):
    def exhausted(cik, acc, doc):
        raise RuntimeError("SEC still failing after 5 retries")
    monkeypatch.setattr(poll_edgar, "fetch_excerpt", exhausted)
    with pytest.raises(RuntimeError):
        poll_edgar.to_row(BANK, filing())


def test_unpadded_cik_is_padded_and_bad_bank_is_isolated(monkeypatch, fake_db):
    fake_db.banks = [
        {"bank_id": "jpm", "cik": "19617", "holding_name": "JPMorgan"},
        {"bank_id": "bad", "cik": "9999999999", "holding_name": "Ghost"},
        {"bank_id": "pnc", "cik": "0000713676", "holding_name": "PNC"},
    ]
    urls = []
    def fake_get(url):
        urls.append(url)
        if "9999999999" in url:
            raise requests.HTTPError("404")
        class R:
            def json(self):
                return submissions({
                    "accessionNumber": ["0000000000-26-000001"],
                    "filingDate": ["2026-07-10"], "form": ["10-Q"],
                    "primaryDocument": ["d.htm"], "items": [""],
                    "acceptanceDateTime": ["2026-07-10T12:00:00.000Z"]})
        return R()
    monkeypatch.setattr(poll_edgar, "_get", fake_get)

    with pytest.raises(SystemExit, match="bad"):
        poll_edgar.main()

    assert any("CIK0000019617.json" in u for u in urls)      # zfill applied
    assert {k[1] for k in fake_db.watermarks} == {"jpm", "pnc"}


def test_cutoff_overlap_covers_late_disseminated_filings():
    watermark = datetime(2026, 7, 11, 1, 0, tzinfo=UTC)   # 20:00 ET July 10
    cutoff = (watermark - poll_edgar.OVERLAP).date()
    late_filing_date = datetime(2026, 7, 10, tzinfo=UTC).date()
    assert late_filing_date >= cutoff
