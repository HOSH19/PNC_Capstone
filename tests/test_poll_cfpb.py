"""Contract tests for the CFPB poller: watermark resume, row shaping, upsert."""

from datetime import date, timedelta

from pipeline import poll_cfpb as cfpb
from tests.conftest import FakeConn


# --- start_day: the three resume branches -----------------------------------

def test_lookback_env_forces_a_wide_repoll(monkeypatch):
    monkeypatch.setenv("LOOKBACK_DAYS", "395")
    assert cfpb.start_day(FakeConn()) == date.today() - timedelta(days=395)


def test_empty_table_looks_back_the_default_window(monkeypatch):
    monkeypatch.delenv("LOOKBACK_DAYS", raising=False)
    conn = FakeConn(fetchone_result={"mx": None})
    assert cfpb.start_day(conn) == date.today() - timedelta(
        days=cfpb.DEFAULT_LOOKBACK_DAYS)


def test_resumes_from_stored_max_minus_overlap(monkeypatch):
    monkeypatch.delenv("LOOKBACK_DAYS", raising=False)
    conn = FakeConn(fetchone_result={"mx": date(2026, 7, 10)})
    assert cfpb.start_day(conn) == date(2026, 7, 10) - timedelta(
        days=cfpb.OVERLAP_DAYS)


# --- to_rows: field mapping and skip rules -----------------------------------

def bank(bank_id="pnc", legal="PNC Bank, National Association",
         holding="PNC Financial Services Group, Inc.",
         aliases=("PNC", "PNC Bank"), notes=None):
    return {"bank_id": bank_id, "bank_legal_name": legal, "holding_name": holding,
            "aliases": list(aliases), "notes": notes}


INDEX = cfpb.build_company_index([bank()])


def hit(**over):
    s = {"complaint_id": "123", "company": "PNC Bank, National Association",
         "date_received": "2026-07-01T12:00:00-05:00", "product": "Checking",
         "timely": "Yes", "complaint_what_happened": "It happened."}
    s.update(over)
    return {"_source": s}


def test_row_shape_and_field_mapping():
    [row] = cfpb.to_rows([hit()], INDEX)
    assert row["complaint_id"] == 123
    assert row["bank_id"] == "pnc"
    assert row["date_received"] == "2026-07-01"   # timestamp truncated to date
    assert row["timely_response"] is True
    assert row["narrative"] == "It happened."
    assert row["consumer_disputed"] is None       # field retired by CFPB in 2017


def test_timely_maps_no_to_false_and_missing_to_none():
    [row] = cfpb.to_rows([hit(timely="No")], INDEX)
    assert row["timely_response"] is False
    [row] = cfpb.to_rows([hit(timely=None)], INDEX)
    assert row["timely_response"] is None


def test_narrative_is_capped_and_empty_becomes_none():
    [row] = cfpb.to_rows([hit(complaint_what_happened="x" * 5000)], INDEX)
    assert len(row["narrative"]) == 4000
    [row] = cfpb.to_rows([hit(complaint_what_happened="")], INDEX)
    assert row["narrative"] is None


def test_incomplete_or_unmatched_hits_are_skipped():
    hits = [hit(company=None),                       # nothing to match on
            hit(complaint_id=None),                  # no primary key
            hit(date_received=None),                 # NOT NULL column
            hit(company="Equifax Inc.")]             # not a tracked bank
    assert cfpb.to_rows(hits, INDEX) == []


def test_company_field_matches_first_bank_in_index_order():
    # The CFPB company string maps to exactly one bank; on a double match the
    # first bank in index order (bank_id sort) wins. Documented behavior.
    index = cfpb.build_company_index(
        [bank("aaa", legal=None, holding=None, aliases=["PNC"]), bank()])
    assert cfpb.match_bank("PNC Bank", index) == "aaa"


# --- upsert_complaints: narrative COALESCE contract ---------------------------

def test_upsert_updates_fields_but_never_wipes_a_narrative():
    conn = FakeConn()
    assert cfpb.upsert_complaints(conn, cfpb.to_rows([hit()], INDEX)) == 1
    assert conn.commits == 1
    sql, rows = conn.cur.batches[0]
    assert "narrative = COALESCE(EXCLUDED.narrative, cfpb_complaint.narrative)" in sql
    assert "narrative = EXCLUDED.narrative" not in sql
    assert "complaint_id = EXCLUDED.complaint_id" not in sql
    assert "company_response = EXCLUDED.company_response" in sql
    assert len(rows) == 1


def test_upsert_of_nothing_is_a_noop():
    conn = FakeConn()
    assert cfpb.upsert_complaints(conn, []) == 0
    assert conn.commits == 0 and conn.cur.batches == []
