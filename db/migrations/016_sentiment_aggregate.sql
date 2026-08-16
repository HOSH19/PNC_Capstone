-- 016_sentiment_aggregate.sql — sentiment axis rollup, the counterpart to
-- 012's bank_index_score. Written ONLY by pipeline/aggregate_sentiment.py;
-- consumers (Rita's dashboard, Shu Han's backtest) read the table and never
-- import scoring code — same contract convention as the fundamentals axis.
--
-- Keyed (bank_id, quarter_end_date) rather than 012's certificate-first key
-- because bank_id is the sentiment side's native identity (raw_item.bank_id);
-- fdic_cert_number is carried resolved for the join to bank_index_score and
-- distress_bank_quarter.csv, mirroring how 012 carries bank_id.
--
-- The directional counts are threshold counts, NOT argmax (DESIGN, "Training
-- on a champion that missed the criteria", obligation 2): the model
-- under-calls direction, so serving lowers the bar instead of retraining.
-- The threshold used is stored on every row; tuning it is a cheap full
-- recompute, and the backtest tunes against item-level rows anyway.

CREATE TABLE bank_sentiment_quarter (
    bank_id          text    NOT NULL REFERENCES bank(bank_id),
    quarter_end_date date    NOT NULL,
    fdic_cert_number integer,

    -- The funnel, so a low signal count is legible: collected -> passed the
    -- attribution gate -> gated AND scored. n_items counts everything filed
    -- under the bank with a publish date; the gap to n_attributed is mostly
    -- live-GDELT rows whose titles never name the bank (~93% of them).
    n_items          integer NOT NULL,
    n_attributed     integer NOT NULL,
    n_scored         integer NOT NULL,

    -- Directional signal among gated+scored rows: count with
    -- p(class) >= threshold, and the mean probabilities for
    -- threshold-free consumers.
    n_negative       integer NOT NULL,
    n_positive       integer NOT NULL,
    mean_p_negative  numeric,
    mean_p_positive  numeric,

    threshold        numeric NOT NULL,
    model_version    text,              -- distinct versions seen, comma-joined
    computed_at      timestamptz NOT NULL DEFAULT now(),

    PRIMARY KEY (bank_id, quarter_end_date)
);

-- The join path to the fundamentals axis and the distress labels.
CREATE INDEX idx_bank_sentiment_quarter_cert
    ON bank_sentiment_quarter (fdic_cert_number, quarter_end_date DESC);
