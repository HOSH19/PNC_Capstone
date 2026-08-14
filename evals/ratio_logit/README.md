# Fundamentals rivals to Ming's `gp50_fixed_v1`

Hand-built / engineered Call Report features, trained on the **full filer
panel** with a **366-day embargo** relative to the protocol split
(`train ≤ 2020-12-30`, test after `2021-12-31`).

```bash
# 1. Full-sample labels (drops unclosed tails; verifies vs seed CSV)
python3 evals/build_distress_labels_full.py --verify-seed

# 2a. Linear ratio model (v2)
python3 evals/ratio_logit/train_score.py

# 2b. Engineered HGB / XGB (v2)
pip install -r evals/ratio_logit/requirements.txt
python3 evals/ratio_logit/train_score_boost.py

# 3. Fair intersected backtest (same test keys for every model)
python3 evals/backtest.py --intersect --budget auto \
  --labels evals/items/distress_bank_quarter_full.csv \
  --scores index/data/scores_hgb_eng_v2.csv \
  --scores index/data/scores_xgb_eng_v2.csv \
  --scores index/data/scores_gp50_fixed_v1.csv \
  --scores index/data/scores_ratios_logit_v2.csv
```

Fair comparison writeup: [`evals/reports/2026-08-07_fair_vs_gp50.md`](../reports/2026-08-07_fair_vs_gp50.md).
