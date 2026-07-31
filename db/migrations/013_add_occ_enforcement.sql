-- Allow OCC enforcement actions into raw_item (Shu Han task 1b).
-- Completes the enforcement trio with FDIC (005) and Fed (006). ~35 of the
-- 104 seed banks are national associations supervised by the OCC — without
-- this source, enforcement coverage is systematically blind for those banks.
-- Pattern documented in db/migrations/002_raw_item.sql header.
--
-- IMPORTANT: this CHECK is additive. Historical rows may still use
-- alpha_vantage / newsapi (tiered out of pollers in scoring/DESIGN.md but
-- present in the live DB). Do not drop those values or ALTER will fail with
-- CheckViolation. fdic_enforcement is listed for parity with migration 005
-- even if a given environment has not ingested FDIC rows yet.

ALTER TABLE raw_item DROP CONSTRAINT IF EXISTS raw_item_source_check;
ALTER TABLE raw_item ADD CONSTRAINT raw_item_source_check
    CHECK (source IN (
        'gdelt',
        'edgar',
        'fdic_enforcement',
        'fed_enforcement',
        'agency_rss',
        'alpha_vantage',
        'newsapi',
        'occ_enforcement'
    ));
