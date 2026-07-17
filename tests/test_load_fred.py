"""Contract tests for the FRED loader: value filtering and series isolation."""

import pytest

from pipeline.loaders import load_fred as lf
from tests.conftest import FakeConn


def obs(value, date="2026-07-01"):
    return {"date": date, "value": value}


def test_missing_value_markers_are_skipped():
    conn = FakeConn()
    observations = [obs("1.23"), obs("."), obs(None), obs("-0.5", "2026-07-02")]
    assert lf.upsert_observations(conn, "STLFSI4", observations) == 2
    assert conn.commits == 1
    sql, rows = conn.cur.batches[0]
    assert "ON CONFLICT (series_id, date)" in sql
    assert rows == [("STLFSI4", "2026-07-01", 1.23),
                    ("STLFSI4", "2026-07-02", -0.5)]


def test_all_missing_is_a_noop():
    conn = FakeConn()
    assert lf.upsert_observations(conn, "TOTLL", [obs("."), obs(None)]) == 0
    assert conn.commits == 0 and conn.cur.batches == []


def test_bad_series_is_isolated_and_flags_heartbeat(monkeypatch, fake_db):
    monkeypatch.setenv("FRED_API_KEY", "test-key")

    def fake_fetch(series_id, api_key):
        assert api_key == "test-key"
        if series_id == "BAMLH0A0HYM2":
            raise RuntimeError("boom")
        return [obs("1.0")]

    upserted = []
    monkeypatch.setattr(lf, "fetch_series", fake_fetch)
    monkeypatch.setattr(lf, "upsert_observations",
                        lambda conn, sid, observations: upserted.append(sid) or 1)

    with pytest.raises(SystemExit, match="BAMLH0A0HYM2"):
        lf.main()

    assert upserted == [s for s in lf.SERIES if s != "BAMLH0A0HYM2"]
    assert fake_db.rollbacks == 1
    job, seen, inserted, ok = fake_db.heartbeats[-1]
    assert (job, inserted, ok) == ("load_fred", len(upserted), False)
