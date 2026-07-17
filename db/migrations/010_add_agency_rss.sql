-- Allow Fed general press-release/policy RSS into raw_item (Ming).
-- Distinct from fed_enforcement (006): that covers enforcement actions from
-- a dedicated CSV; this covers general press releases and policy notices
-- (Fed_AllReleases, Fed_BankRegPolicy, H.4.1) from RSS feeds with no
-- historical query capability -- see pipeline/poll_agency_rss.py.
-- Pattern documented in db/migrations/002_raw_item.sql header.

ALTER TABLE raw_item DROP CONSTRAINT raw_item_source_check;
ALTER TABLE raw_item ADD CONSTRAINT raw_item_source_check
    CHECK (source IN ('gdelt', 'edgar', 'fdic_enforcement', 'fed_enforcement', 'agency_rss'));
