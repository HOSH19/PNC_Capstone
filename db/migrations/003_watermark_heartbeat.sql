-- 003_watermark_heartbeat.sql — incremental-poller state and run observability.

CREATE TABLE watermark (
    source         text NOT NULL,
    bank_id        text NOT NULL REFERENCES bank(bank_id),
    last_polled_at timestamptz NOT NULL,
    PRIMARY KEY (source, bank_id)
);

CREATE TABLE pipeline_heartbeat (
    id             bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    run_at         timestamptz NOT NULL DEFAULT now(),
    job            text NOT NULL,              -- 'poll_gdelt' | 'poll_edgar'
    items_seen     integer NOT NULL DEFAULT 0,
    items_inserted integer NOT NULL DEFAULT 0,
    duration_s     real,
    ok             boolean NOT NULL
);
