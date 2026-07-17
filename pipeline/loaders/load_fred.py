"""Full loader: FRED macro series -> Postgres.

Stateless full-loader pattern (RUNBOOK §6): no watermark, re-fetches each
series' complete history every run and upserts by natural key
(series_id, date). No bank matching -- these are industry-wide macro
control variables, not per-bank signals (see DATA_SOURCES.md).

Run: python -m pipeline.loaders.load_fred
"""

import os
import sys
import time

from pipeline import db
from pipeline.http import throttled_get

API_URL = "https://api.stlouisfed.org/fred/series/observations"

# series_id -> human label (label is documentation only, not stored)
SERIES = {
    "STLFSI4": "St. Louis Fed Financial Stress Index",
    "BAMLH0A0HYM2": "ICE BofA US High Yield Index Option-Adjusted Spread",
    "T10YFF": "10-Year Treasury Constant Maturity Minus Federal Funds Rate",
    "DPSACBW027SBOG": "Deposits, All Commercial Banks",
    "TOTLL": "Total Loans and Leases, All Commercial Banks",
}


def fetch_series(series_id: str, api_key: str) -> list[dict]:
    resp = throttled_get(API_URL, params={
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
    }, throttle_s=1.0, label=f"FRED:{series_id}")
    return resp.json().get("observations", [])


def upsert_observations(conn, series_id: str, observations: list[dict]) -> int:
    rows = []
    for obs in observations:
        value = obs.get("value")
        if value in (None, "."):  # "." is FRED's missing-value marker
            continue
        rows.append((series_id, obs["date"], float(value)))
    if not rows:
        return 0
    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO fred_observation (series_id, date, value) "
            "VALUES (%s, %s, %s) "
            "ON CONFLICT (series_id, date) DO UPDATE SET "
            "value = EXCLUDED.value, collected_at = now()",
            rows,
        )
    conn.commit()
    return len(rows)


def main() -> None:
    started = time.monotonic()
    seen = inserted = 0
    failed: list[str] = []
    api_key = os.environ["FRED_API_KEY"]
    conn = db.connect()
    try:
        for series_id in SERIES:
            try:
                observations = fetch_series(series_id, api_key)
                n = upsert_observations(conn, series_id, observations)
                seen += len(observations)
                inserted += n
                print(f"{series_id}: {len(observations)} observations, {n} upserted")
            except Exception as exc:
                # One bad series must not stop the others.
                conn.rollback()
                failed.append(series_id)
                print(f"{series_id}: FAILED: {exc}", file=sys.stderr)
        db.write_heartbeat(conn, "load_fred", seen, inserted,
                           time.monotonic() - started, not failed)
    except Exception:
        try:
            conn.rollback()
            db.write_heartbeat(conn, "load_fred", seen, inserted,
                               time.monotonic() - started, False)
        except Exception:
            pass  # never mask the original failure
        raise
    finally:
        conn.close()
    if failed:
        sys.exit(f"load_fred: failed series: {', '.join(failed)}")


if __name__ == "__main__":
    sys.exit(main())
