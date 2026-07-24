-- Full loader table for FRED macro series (Ming). No bank_id: these are
-- industry-wide control variables, not per-bank signals (see
-- DATA_SOURCES.md). pipeline/loaders/load_fred.py.

CREATE TABLE fred_observation (
    series_id    text NOT NULL,
    date         date NOT NULL,
    value        numeric,
    collected_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (series_id, date)
);
