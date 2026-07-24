-- Allow OCC enforcement actions into raw_item.

ALTER TABLE raw_item DROP CONSTRAINT raw_item_source_check;

ALTER TABLE raw_item ADD CONSTRAINT raw_item_source_check
CHECK (
    source IN (
        'gdelt',
        'edgar',
        'fed_enforcement',
        'agency_rss',
        'alpha_vantage',
        'newsapi',
        'occ_enforcement'
    )
);