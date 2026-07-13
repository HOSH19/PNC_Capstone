-- Allow FDIC enforcement orders into raw_item.
-- Pattern documented in db/migrations/002_raw_item.sql header.

ALTER TABLE raw_item DROP CONSTRAINT raw_item_source_check;
ALTER TABLE raw_item ADD CONSTRAINT raw_item_source_check
    CHECK (source IN ('gdelt', 'edgar', 'fdic_enforcement'));
