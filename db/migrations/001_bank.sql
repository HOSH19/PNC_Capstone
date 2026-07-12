-- 001_bank.sql — manual bank crosswalk, seeded from db/seed/banks.csv.
-- Adding a bank must require one seed row and zero code changes.
-- Per-source identifiers live here (cik → edgar, gdelt_query → gdelt); if a new
-- source needs its own identifier, add a column via migration and mirror it as
-- a column in db/seed/banks.csv + pipeline/seed_banks.py COLUMNS.

CREATE TABLE bank (
    bank_id         text PRIMARY KEY,          -- slug, e.g. 'pnc'
    holding_name    text NOT NULL,             -- holding company legal name
    bank_legal_name text,                      -- lead bank legal name
    cik             text,                      -- zero-padded 10-digit string
    ticker          text,
    fdic_cert       integer,
    rssd_id         integer,
    gdelt_query     text,                      -- GDELT DOC 2.0 query string
    aliases         text[] NOT NULL DEFAULT '{}',
    is_live         boolean NOT NULL DEFAULT false,
    is_backtest     boolean NOT NULL DEFAULT false,
    notes           text
);
