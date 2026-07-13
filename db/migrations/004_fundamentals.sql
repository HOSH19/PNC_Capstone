-- Fundamentals tables: Postgres translation of Shu Han's
-- unified_ffiec_fdic_dataset/sql/schema.sql. Column names match his CSVs
-- 1:1 so the loader (pipeline/loaders/load_fundamentals.py) copies them
-- straight in. Units: dollar amounts are in THOUSANDS of USD, ratios are
-- percentages (see unified_ffiec_fdic_dataset/README.md).
--
-- Deliberate deviation from his schema.sql: no FOREIGN KEY constraints.
-- SQLite never enforced them, so the CSVs may legitimately contain fact
-- rows for banks absent from dim_bank (e.g. pre-2015 call reports for
-- since-dissolved banks). Joins still work through fdic_cert_number;
-- the crosswalk join key from our side is bank.fdic_cert.

CREATE TABLE dim_bank (
    fdic_cert_number integer PRIMARY KEY,
    rssd_id          integer,
    bank_name        text NOT NULL,
    city             text,
    state            text,
    active_status    boolean,
    total_assets     double precision,
    charter_type     text,
    established_date date,
    last_updated     date
);

CREATE TABLE fact_call_report (
    fdic_cert_number          integer NOT NULL,
    rssd_id                   integer,
    report_date               date NOT NULL,
    total_assets              double precision,
    total_deposits            double precision,
    tier1_capital_ratio       double precision,
    total_capital_ratio       double precision,
    npl_ratio                 double precision,
    loan_loss_allowance_ratio double precision,
    liquidity_ratio           double precision,
    securities_unrealized_loss double precision,
    cre_loans                 double precision,
    PRIMARY KEY (fdic_cert_number, report_date)
);

CREATE TABLE fact_distress_event (
    fdic_cert_number      integer NOT NULL,
    failure_date          date NOT NULL,
    bank_name             text,
    city                  text,
    state                 text,
    acquiring_institution text,
    event_type            text,
    distress_label        integer NOT NULL DEFAULT 1,
    PRIMARY KEY (fdic_cert_number, failure_date)
);

CREATE TABLE fact_bank_quarter (
    fdic_cert_number          integer NOT NULL,
    quarter_end_date          date NOT NULL,
    total_assets              double precision,
    total_deposits            double precision,
    tier1_capital_ratio       double precision,
    total_capital_ratio       double precision,
    npl_ratio                 double precision,
    loan_loss_allowance_ratio double precision,
    liquidity_ratio           double precision,
    securities_unrealized_loss double precision,
    cre_loans                 double precision,
    distress_within_4q        integer NOT NULL DEFAULT 0,
    distress_within_8q        integer NOT NULL DEFAULT 0,
    days_to_distress          integer,
    PRIMARY KEY (fdic_cert_number, quarter_end_date)
);
