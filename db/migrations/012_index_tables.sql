-- 012_index_tables.sql — fundamentals axis (Ming). Output of the
-- GaussianProcessClassifier over Call Report features; see index/README.md
-- and docs/roles/ming.md.
--
-- Two tables, both written ONLY by the index recompute job:
--   bank_index_score   — one row per bank-quarter: the score, its band, and
--                        the GP's uncertainty.
--   bank_index_feature — one row per bank-quarter-feature: the values,
--                        thresholds and statuses the dashboard shows.
--
-- Consumers (Shu Han's backtest, Rita's dashboard) read these tables and never
-- import index/ code — the schema is the only shared contract. Everything
-- GP-related comes from here, including feature values: some features the
-- dashboard needs are derived (fee income ratio, CRE/total capital) or do not
-- exist in fact_call_report at all, and thresholds/statuses are model logic.
--
-- Join key is (fdic_cert_number, quarter_end_date), matching
-- fact_call_report.report_date and evals/items/distress_bank_quarter.csv.
-- Value join, no FK — same convention as bank.fdic_cert ↔ fdic_cert_number.

CREATE TABLE bank_index_score (
    fdic_cert_number   integer NOT NULL,
    quarter_end_date   date    NOT NULL,
    bank_id            text,              -- convenience, from db/seed/banks.csv

    -- Point estimates
    distress_prob      numeric NOT NULL,  -- GP P(distress within 4q); backtest metric input
    score              numeric NOT NULL,  -- 0-100, monotone decreasing in distress_prob
    band               text    NOT NULL   -- display band, cutoffs in index/config
                       CHECK (band IN ('sound', 'neutral', 'distress')),

    -- Uncertainty. Score is a monotone transform of the GP latent, so latent
    -- quantiles map exactly onto score quantiles (endpoints flip: a high latent
    -- means high distress risk, i.e. a LOW score).
    score_lo_80        numeric,
    score_hi_80        numeric,
    score_lo_95        numeric,
    score_hi_95        numeric,
    latent_mean        numeric,           -- raw GP output, kept so any other
    latent_var         numeric,           -- interval can be recomputed later

    -- Provenance / data quality
    n_missing_features smallint,          -- inputs imputed for this row. npl_ratio is
                                          -- ~42% NULL DB-wide: a score built on imputed
                                          -- inputs must be presentable as low-confidence,
                                          -- not silently shown as fact.
    model_version      text    NOT NULL,
    config_version     text,              -- which index/config/*.yaml produced this
    computed_at        timestamptz NOT NULL DEFAULT now(),

    PRIMARY KEY (fdic_cert_number, quarter_end_date)
);

CREATE INDEX idx_bank_index_score_bank_quarter
    ON bank_index_score (bank_id, quarter_end_date DESC);

-- Per-feature detail behind a score: what the dashboard's quarter-over-quarter
-- table renders. Values are the ones the MODEL saw (post winsorize/impute), so
-- they can differ slightly from raw fact_call_report — that is intentional.
CREATE TABLE bank_index_feature (
    fdic_cert_number   integer NOT NULL,
    quarter_end_date   date    NOT NULL,
    feature_name       text    NOT NULL,  -- 'tier1_capital_ratio', 'fee_income_ratio', ...
    display_name       text,              -- 'Tier 1 capital ratio %'
    value              numeric,           -- NULL when the input was missing entirely
    is_imputed         boolean NOT NULL DEFAULT false,
    threshold_text     text,              -- ready-to-render, e.g. '>= 9.0', '2.0 - 7.0'
    status             text               -- CHECK, not enum: adding a status later is
                       CHECK (status IN ('within_range', 'near_threshold', 'breach')),

    PRIMARY KEY (fdic_cert_number, quarter_end_date, feature_name)
);
