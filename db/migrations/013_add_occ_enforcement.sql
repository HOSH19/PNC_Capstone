-- Allow OCC enforcement actions into raw_item (Shu Han task 1b).
-- Extends migration 010's CHECK with occ_enforcement only.
-- OCC ownership: DATA_SOURCES.md (Yusheng). Coordinate with Ming on 012
-- (012_index_tables.sql) before applying — this file is numbered 013.
-- Pattern: db/migrations/002_raw_item.sql header.

ALTER TABLE raw_item DROP CONSTRAINT IF EXISTS raw_item_source_check;
ALTER TABLE raw_item ADD CONSTRAINT raw_item_source_check
    CHECK (source IN (
        'gdelt',
        'edgar',
        'fdic_enforcement',
        'fed_enforcement',
        'agency_rss',
        'occ_enforcement'
    ));
