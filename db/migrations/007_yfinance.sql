-- Full loader tables for yfinance (Ming): daily prices + analyst target snapshots.
-- Both stateless full loaders (RUNBOOK §6): re-fetch and upsert by natural PK,
-- no watermark. pipeline/loaders/load_yfinance.py writes both tables in one
-- run (one yfinance session per bank -- avoids a second API round trip just
-- for analyst targets).

CREATE TABLE market_daily (
    ticker       text NOT NULL,
    date         date NOT NULL,
    open         double precision,
    high         double precision,
    low          double precision,
    close        double precision,
    volume       bigint,
    collected_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (ticker, date)
);

-- yfinance's analyst_price_targets has no history endpoint -- it only ever
-- returns the current consensus. This table's "history" is entirely
-- manufactured by snapshotting once per loader run; snapshot_date is the run
-- date, not a source-provided date (contrast with market_daily.date, which
-- is a real trading day from the source).
CREATE TABLE analyst_target (
    ticker        text NOT NULL,
    snapshot_date date NOT NULL,
    current_price double precision,
    target_high   double precision,
    target_low    double precision,
    target_mean   double precision,
    target_median double precision,
    collected_at  timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (ticker, snapshot_date)
);
