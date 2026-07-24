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