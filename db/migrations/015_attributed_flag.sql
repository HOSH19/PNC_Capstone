-- 015_attributed_flag.sql — persist the attribution gate's verdict per row.
--
-- `attributed` answers: does this row count toward the bank it was filed
-- under? (scoring/DESIGN.md "Bank attribution".) The check itself is Python
-- regex (pipeline/attribution.py), so it cannot run inside a SQL rollup;
-- persisting the verdict once per row is what lets the aggregation layer —
-- and Shu Han's backtest, which needs to tune thresholds over item-level
-- rows — be plain SQL.
--
-- NULL means "not yet computed", written by pipeline/attribute_items.py.
-- The verdict is a pure function of (source, title, bank_id), all immutable
-- after insert, so a computed value never goes stale. Re-run a changed gate
-- by setting the column back to NULL for the affected rows.

ALTER TABLE raw_item ADD COLUMN attributed boolean;

-- The batch job's scan, same shape as idx_raw_item_finbert_pending.
CREATE INDEX idx_raw_item_attribution_pending
    ON raw_item (id) WHERE attributed IS NULL;
