"""Full loader: yfinance daily prices + analyst target snapshots -> Postgres.

Stateless full-loader pattern (RUNBOOK §6): no watermark. Each run re-fetches
a short trailing window per ticker and upserts by natural key
(market_daily: ticker+date, analyst_target: ticker+snapshot_date). One
yfinance session per bank covers both tables, avoiding a second round of API
calls just for analyst targets.

analyst_price_targets has no history endpoint -- yfinance only ever returns
the current consensus. Its "history" in our table is entirely manufactured
by snapshotting once per run.

Historical backfill (full price history per ticker, not just the trailing
window) is a separate one-off script, not this scheduled run -- pulling
multi-decade history for every live bank on every scheduled run would be
slow and almost entirely redundant with the previous run's data.

Run: python -m pipeline.loaders.load_yfinance
"""

import sys
import time
from datetime import UTC, date, datetime

import yfinance as yf

from pipeline import db

# Rolling window: enough to cover a long weekend/holiday gap without
# re-pulling full history every run.
PRICE_PERIOD = "5d"
THROTTLE_S = 1.5  # yfinance rate-limits aggressively on rapid sequential calls
MAX_RETRIES = 3


def _with_retry(label: str, fn):
    """yfinance has no pipeline.http.throttled_get equivalent (it manages its
    own HTTP session internally) -- this is a minimal retry for the 429s it
    intermittently returns, not a general-purpose HTTP client."""
    for attempt in range(MAX_RETRIES):
        try:
            return fn()
        except Exception:
            if attempt + 1 >= MAX_RETRIES:
                raise
            time.sleep(THROTTLE_S * 2 ** attempt)
    return None  # unreachable


def upsert_market_daily(conn, ticker: str, hist) -> int:
    if hist.empty:
        return 0
    rows = [
        (ticker, idx.date(), float(r.Open), float(r.High), float(r.Low),
         float(r.Close), int(r.Volume))
        for idx, r in hist.iterrows()
    ]
    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO market_daily (ticker, date, open, high, low, close, volume) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s) "
            "ON CONFLICT (ticker, date) DO UPDATE SET "
            "open = EXCLUDED.open, high = EXCLUDED.high, low = EXCLUDED.low, "
            "close = EXCLUDED.close, volume = EXCLUDED.volume, collected_at = now()",
            rows,
        )
    conn.commit()
    return len(rows)


def upsert_analyst_target(conn, ticker: str, targets: dict | None, snapshot_date: date) -> int:
    if not targets:
        return 0
    row = (
        ticker, snapshot_date,
        targets.get("current"), targets.get("high"), targets.get("low"),
        targets.get("mean"), targets.get("median"),
    )
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO analyst_target "
            "(ticker, snapshot_date, current_price, target_high, target_low, target_mean, target_median) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s) "
            "ON CONFLICT (ticker, snapshot_date) DO UPDATE SET "
            "current_price = EXCLUDED.current_price, target_high = EXCLUDED.target_high, "
            "target_low = EXCLUDED.target_low, target_mean = EXCLUDED.target_mean, "
            "target_median = EXCLUDED.target_median, collected_at = now()",
            row,
        )
    conn.commit()
    return 1


def main() -> None:
    started = time.monotonic()
    seen = inserted = 0
    failed: list[str] = []
    today = datetime.now(UTC).date()
    conn = db.connect()
    try:
        for bank in db.get_live_banks(conn):
            ticker = bank["ticker"]
            if not ticker:
                continue
            try:
                t = yf.Ticker(ticker)

                hist = _with_retry(ticker, lambda: t.history(period=PRICE_PERIOD, auto_adjust=True))
                n_prices = upsert_market_daily(conn, ticker, hist)
                seen += len(hist)
                inserted += n_prices

                time.sleep(THROTTLE_S)

                targets = _with_retry(ticker, lambda: t.analyst_price_targets)
                n_targets = upsert_analyst_target(conn, ticker, targets, today)
                seen += 1
                inserted += n_targets

                print(f"{bank['bank_id']} ({ticker}): {n_prices} price rows, "
                      f"{n_targets} target row")
            except Exception as exc:
                # One bad ticker must not stop the others.
                conn.rollback()
                failed.append(bank["bank_id"])
                print(f"{bank['bank_id']} ({ticker}): FAILED: {exc}", file=sys.stderr)
            time.sleep(THROTTLE_S)
        db.write_heartbeat(conn, "load_yfinance", seen, inserted,
                           time.monotonic() - started, not failed)
    except Exception:
        try:
            conn.rollback()
            db.write_heartbeat(conn, "load_yfinance", seen, inserted,
                               time.monotonic() - started, False)
        except Exception:
            pass  # never mask the original failure
        raise
    finally:
        conn.close()
    if failed:
        sys.exit(f"load_yfinance: failed banks: {', '.join(failed)}")


if __name__ == "__main__":
    sys.exit(main())
