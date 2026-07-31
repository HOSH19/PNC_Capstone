-- Allow OCC enforcement actions into raw_item (Shu Han task 1b).
-- Completes the enforcement trio with FDIC (005) and Fed (006). ~35 of the
-- 104 seed banks are national associations supervised by the OCC — without
-- this source, enforcement coverage is systematically blind for those banks.
-- Pattern documented in db/migrations/002_raw_item.sql header.
--
-- NOTE: Do NOT re-add alpha_vantage / newsapi here (tiered out in
-- scoring/DESIGN.md). Extend only from the current live CHECK (010).

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
