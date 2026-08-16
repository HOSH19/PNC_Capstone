from datetime import date

from pipeline.backfill_edgar import (
    CSV_COLUMNS,
    csv_row,
    iter_columnar_filings,
    pages_in_range,
    resume_pairs,
    select_filings,
)

BANK = {
    "bank_id": "pnc",
    "cik": "0000713676",
    "holding_name": "PNC Financial Services Group, Inc.",
}
SINCE, UNTIL = date(2020, 1, 1), date(2025, 1, 1)


def _filing(acc="0000713676-22-000001", form="8-K", filed="2022-06-01", doc="ex99.htm"):
    return {
        "accessionNumber": acc,
        "filingDate": filed,
        "form": form,
        "primaryDocument": doc,
        "items": "2.02",
        "acceptanceDateTime": "2022-06-01T16:05:00.000Z",
    }


def test_select_filings_keeps_forms_and_window_only():
    filings = [
        _filing(),
        _filing(acc="a2", form="4"),  # insider form, never kept
        _filing(acc="a3", filed="2019-12-31"),  # before since
        _filing(acc="a4", filed="2025-01-01"),  # until is exclusive
        _filing(acc="a5", form="10-K/A", filed="2020-01-01"),  # since inclusive
    ]
    kept = list(select_filings(filings, SINCE, UNTIL))
    assert [f["accessionNumber"] for f in kept] == ["0000713676-22-000001", "a5"]


def test_select_filings_dedups_by_accession():
    """filings.recent and the first archive page can overlap at the seam."""
    kept = list(select_filings([_filing(), _filing()], SINCE, UNTIL))
    assert len(kept) == 1


def test_csv_row_shape_matches_the_live_poller():
    """Same title shaping, same index-page url, midnight-UTC published_at,
    and attributed is the literal 'true' — filings are the bank's own
    (attribute_items.ATTRIBUTED_AT_INGEST)."""
    row = csv_row(BANK, _filing(), "Results announced")
    assert list(row) == CSV_COLUMNS
    assert row["title"] == "PNC Financial Services Group, Inc. 8-K"
    assert row["published_at"] == "2022-06-01T00:00:00+00:00"
    assert row["url"] == (
        "https://www.sec.gov/Archives/edgar/data/713676/000071367622000001/"
        "0000713676-22-000001-index.htm"
    )
    assert row["text_excerpt"] == "Results announced"
    assert row["attributed"] == "true"


def test_csv_row_missing_excerpt_is_empty_string_not_none():
    """csv.writer would spell None as ''-adjacent junk on round-trip; write
    the empty string deliberately."""
    assert csv_row(BANK, _filing(form="10-K", doc=""), None)["text_excerpt"] == ""


def test_pages_in_range_skips_pages_outside_the_window():
    files = [
        {"name": "p1.json", "filingFrom": "2015-01-01", "filingTo": "2019-12-31"},
        {"name": "p2.json", "filingFrom": "2018-05-01", "filingTo": "2021-03-31"},
        {"name": "p3.json", "filingFrom": "2021-04-01", "filingTo": "2024-06-30"},
        {"name": "p4.json", "filingFrom": "2025-01-01", "filingTo": "2026-01-01"},
    ]
    assert pages_in_range(files, SINCE, UNTIL) == ["p2.json", "p3.json"]


def test_iter_columnar_filings_zips_an_archive_page():
    """Archive pages carry the same parallel arrays at top level that
    filings.recent nests; the wrapper feeds them through the live poller's
    zip + length check."""
    page = {
        "accessionNumber": ["a1", "a2"],
        "filingDate": ["2020-02-03", "2020-03-04"],
        "form": ["8-K", "10-Q"],
        "primaryDocument": ["d1.htm", "d2.htm"],
        "items": ["2.02", ""],
        "acceptanceDateTime": ["t1", "t2"],
    }
    filings = list(iter_columnar_filings(page))
    assert [f["accessionNumber"] for f in filings] == ["a1", "a2"]
    assert filings[1]["form"] == "10-Q"


def test_resume_pairs_builds_the_skip_set_from_existing_rows():
    existing = [
        {"bank_id": "pnc", "url": "https://sec.gov/1", "title": "x"},
        {"bank_id": "wfc", "url": "https://sec.gov/1", "title": "y"},
    ]
    done = resume_pairs(existing)
    assert ("pnc", "https://sec.gov/1") in done
    assert ("wfc", "https://sec.gov/1") in done
    assert ("pnc", "https://sec.gov/2") not in done
