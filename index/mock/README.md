# Mock data for the fundamentals axis

Sample rows matching `db/migrations/012_index_tables.sql` column-for-column, so
the dashboard can be built before real scores exist. Numbers are illustrative,
not model output.

| File | Feeds |
|---|---|
| `bank_index_score_sample.csv` | the score panel — score, band, confidence intervals |
| `bank_index_feature_sample.csv` | the quarter-over-quarter feature table |

States covered (so every UI variant can be styled):

| bank_id | State |
|---|---|
| `jpm`, `pnc` | healthy, tight confidence band |
| `wfc` | drifting down across quarters (band changes) |
| `wal` | deteriorating into distress, features in `breach` |
| `zion` | borderline, `near_threshold` |
| `cbu` | 2 missing inputs, 80% interval ≈ [11, 99] — the "too thin to trust" case |

## Feature availability

`fact_call_report` has 12 columns: assets, deposits, tier1/total capital ratios,
npl, loan-loss allowance, liquidity, unrealized securities loss, cre_loans.

The mentor's reference feature list also mentions **fee income ratio** and
**loans vs capital**; the dataset contains neither (no fee/noninterest income,
no total loans, no capital level). `fee_income_ratio` appears in these mock rows
for layout purposes, but **may never be populated with real values**. Render
whatever feature rows are present rather than assuming a fixed list.

The real model uses what the data actually supports: the ratios above plus
derived ones (cre/assets, unrealized loss/assets, deposits/assets, log assets)
and quarter-over-quarter changes.
