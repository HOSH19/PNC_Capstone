# Backtest report — harness smoke (run A)

Labels: `evals/items/distress_bank_quarter.csv`  
Split: train ≤ `2021-12-31`, test `2021-12-31` < date ≤ `2024-12-31`  
Target: `distress_within_4q`. Score: higher = riskier.

## Results

| model_version | n_test | n_pos | PR-AUC | precision@50 | recall@budget=10 (micro) | recall detail |
|---|---:|---:|---:|---:|---:|---|
| `naive_neg_tier1` | 1246 | 32 | 0.1342 | 0.1400 | 0.3438 | 11/32 pos over 10 qtrs |
| `random_seed0` | 1246 | 32 | 0.0364 | 0.0200 | 0.1562 | 5/32 pos over 10 qtrs |

## Checklist

- [x] model_version and score definition stated
- [x] train/test cuts and positive counts stated
- [x] PR-AUC, precision@k, recall@budget reported
- [x] random + naive −tier1 both present
- [ ] combined run — blocked until sentiment scores exist

## Notes

Naive score is `risk = −tier1_capital_ratio` from `fact_call_report`. If naive is not clearly above random on PR-AUC / recall@budget, treat the harness as suspect before grading Ming's GP.
