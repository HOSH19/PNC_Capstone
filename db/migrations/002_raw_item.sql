-- 002_raw_item.sql — one row per ingested text item, all sources.
-- source is a CHECK (not an enum) so adding a source later is a one-line migration:
--   ALTER TABLE raw_item DROP CONSTRAINT raw_item_source_check;
--   ALTER TABLE raw_item ADD CONSTRAINT raw_item_source_check
--       CHECK (source IN ('gdelt', 'edgar', '<new-source>'));
-- Full "adding a new source" checklist: RUNBOOK.md §6.

CREATE TABLE raw_item (
    id             bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    source         text NOT NULL CHECK (source IN ('gdelt', 'edgar')),
    external_id    text NOT NULL,              -- gdelt: article url, edgar: accession number
    bank_id        text NOT NULL REFERENCES bank(bank_id),
    published_at   timestamptz,
    title          text,
    url            text,
    domain         text,
    text_excerpt   text,                       -- EDGAR 8-K only, capped app-side at ~4000 chars
    title_hash     text,                       -- normalized-title hash (syndication dedup)
    n_duplicates   integer NOT NULL DEFAULT 0, -- duplicates folded into this row pre-insert
    meta           jsonb NOT NULL DEFAULT '{}',
    finbert_status text NOT NULL DEFAULT 'pending',
    llm_status     text NOT NULL DEFAULT 'not_required',
    llm_attempts   integer NOT NULL DEFAULT 0,
    last_error     text,
    collected_at   timestamptz NOT NULL DEFAULT now(),
    UNIQUE (source, external_id, bank_id)
);

CREATE INDEX idx_raw_item_bank_published ON raw_item (bank_id, published_at DESC);
