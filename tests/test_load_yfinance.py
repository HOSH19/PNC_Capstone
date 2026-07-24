"""Contract tests for the yfinance loader: row shaping, retry, isolation."""

import time as real_time
import types
from datetime import date

import pandas as pd
import pytest

from pipeline.loaders import load_yfinance as ly
from tests.conftest import FakeConn


def frame(days: dict[str, tuple]) -> pd.DataFrame:
    """{'2026-07-15': (open, high, low, close, volume), ...} -> history frame."""
    cols = ["Open", "High", "Low", "Close", "Volume"]
    return pd.DataFrame([dict(zip(cols, v)) for v in days.values()],
                        index=pd.to_datetime(list(days)))


# --- upsert_market_daily ------------------------------------------------------

def test_price_rows_are_upserted_by_natural_key():
    conn = FakeConn()
    hist = frame({"2026-07-15": (1.0, 2.0, 0.5, 1.5, 100),
                  "2026-07-16": (1.5, 2.5, 1.0, 2.0, 200)})
    assert ly.upsert_market_daily(conn, "PNC", hist) == 2
    assert conn.commits == 1
    sql, rows = conn.cur.batches[0]
    assert "ON CONFLICT (ticker, date)" in sql
    assert rows[0] == ("PNC", date(2026, 7, 15), 1.0, 2.0, 0.5, 1.5, 100)


def test_nan_rows_are_dropped_instead_of_failing_the_bank():
    """Regression: yfinance returns NaN rows for halted/missing days.
    int(NaN) raised ValueError (failing the whole bank), and NaN prices
    would land as literal 'NaN' in double precision columns."""
    conn = FakeConn()
    nan = float("nan")
    hist = frame({"2026-07-15": (1.0, 2.0, 0.5, 1.5, 100),
                  "2026-07-16": (nan, nan, nan, nan, nan),   # whole row missing
                  "2026-07-17": (1.5, 2.5, 1.0, 2.0, nan)})  # volume missing
    assert ly.upsert_market_daily(conn, "PNC", hist) == 1
    [(_, rows)] = conn.cur.batches
    assert [r[1] for r in rows] == [date(2026, 7, 15)]


def test_empty_history_is_a_noop():
    conn = FakeConn()
    assert ly.upsert_market_daily(conn, "PNC", pd.DataFrame()) == 0
    assert conn.commits == 0


# --- upsert_analyst_target ----------------------------------------------------

def test_target_snapshot_row_and_missing_targets_noop():
    conn = FakeConn()
    targets = {"current": 10.0, "high": 15.0, "low": 8.0,
               "mean": 12.0, "median": 11.5}
    assert ly.upsert_analyst_target(conn, "PNC", targets, date(2026, 7, 17)) == 1
    _, params = conn.cur.executed[0]
    assert params == ("PNC", date(2026, 7, 17), 10.0, 15.0, 8.0, 12.0, 11.5)
    assert ly.upsert_analyst_target(conn, "PNC", None, date(2026, 7, 17)) == 0
    assert ly.upsert_analyst_target(conn, "PNC", {}, date(2026, 7, 17)) == 0


# --- _with_retry ----------------------------------------------------------------

def wired_time(monkeypatch):
    sleeps = []
    monkeypatch.setattr(ly, "time", types.SimpleNamespace(
        monotonic=real_time.monotonic, sleep=sleeps.append))
    return sleeps


def test_retry_backs_off_then_succeeds(monkeypatch):
    sleeps = wired_time(monkeypatch)
    effects = iter([RuntimeError("429"), RuntimeError("429"), "ok"])

    def flaky():
        e = next(effects)
        if isinstance(e, Exception):
            raise e
        return e

    assert ly._with_retry("PNC", flaky) == "ok"
    assert sleeps == [1.5, 3.0]   # THROTTLE_S * 2**attempt


def test_retry_exhaustion_raises(monkeypatch):
    wired_time(monkeypatch)

    def always_429():
        raise RuntimeError("429")

    with pytest.raises(RuntimeError, match="429"):
        ly._with_retry("PNC", always_429)


# --- main: per-bank isolation ---------------------------------------------------

class FakeTicker:
    def __init__(self, ticker):
        self._t = ticker

    def history(self, period, auto_adjust):
        if self._t == "BAD":
            raise RuntimeError("boom")
        return frame({"2026-07-16": (1.0, 2.0, 0.5, 1.5, 100)})

    @property
    def analyst_price_targets(self):
        return {"current": 10.0, "mean": 12.0}


def test_bad_ticker_is_isolated_and_tickerless_bank_is_skipped(monkeypatch, fake_db):
    fake_db.banks = [{"bank_id": "good", "ticker": "GOOD"},
                     {"bank_id": "none", "ticker": None},
                     {"bank_id": "bad", "ticker": "BAD"}]
    wired_time(monkeypatch)
    monkeypatch.setattr(ly.yf, "Ticker", FakeTicker)
    calls = []
    monkeypatch.setattr(ly, "upsert_market_daily",
                        lambda conn, t, h: calls.append(("prices", t)) or 1)
    monkeypatch.setattr(ly, "upsert_analyst_target",
                        lambda conn, t, tg, d: calls.append(("targets", t)) or 1)

    with pytest.raises(SystemExit, match="bad"):
        ly.main()

    assert calls == [("prices", "GOOD"), ("targets", "GOOD")]
    assert fake_db.rollbacks == 1
    job, seen, inserted, ok = fake_db.heartbeats[-1]
    assert (job, inserted, ok) == ("load_yfinance", 2, False)
