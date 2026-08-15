# Backtest — combined ladder vs each axis alone (first combined run)

The run `evals/backtest.py`'s checklist called "blocked until sentiment
scores exist". Sentiment history comes from the file-based corpus (BigQuery
GKG news + EDGAR filings, scored on Kaggle GPU with
`finbert-ft-2026-08-09`); fundamentals from `bank_index_score`; the
combination rule is the dashboard's four-tier ladder
(`pipeline/combine_axes.py`, cutoffs at their starting values: Imminent 30,
Elevated 80, Watch 90, negative when a quarter's share of p≥0.5 negative
items ≥ the cut below).

Labels `evals/items/distress_bank_quarter_full.csv`, test window
2022-01-01..2024-12-31, intersected keys: **1,245 bank-quarters, 32
positives**.

| model | PR-AUC | precision@50 | recall@budget=10 |
|---|---:|---:|---:|
| combined ladder, neg cut 0.10 | **0.0810** | 0.0400 | 9/32 |
| combined ladder, neg cut 0.05 | 0.0794 | 0.0400 | **10/32** |
| combined ladder, neg cut 0.20 | 0.0808 | 0.0400 | 8/32 |
| fundamentals only (gp50_fixed_v1) | 0.0590 | **0.0800** | 8/32 |
| sentiment only (neg_share) | 0.0426 | 0.0800 | 6/32 |

## Reading

1. **The combination beats both axes alone** on PR-AUC and
   recall@budget — the project's central claim, holding on its first
   measurement. Sentiment alone is the weakest signal, yet adding it to
   fundamentals improves both metrics: the axes catch different events.
2. **Robust to the negative cut** across 0.05–0.20 (PR-AUC 0.079–0.081),
   which supports shipping the starting cutoffs. 0.05 maximizes
   recall@budget (10/32).
3. **32 positives is small.** Model differences are 1–2 events; this says
   "the combination is promising", not "proven better".
4. **precision@50 trades away.** The ladder pins all 37 Elevated-Risk
   quarters to the top of the ranking; a pure probability ranking places 4
   true positives in its top-50 where the ladder places 2. Tier
   interpretability costs raw top-k precision — state this in the
   write-up rather than hiding it.

## Reproduce

```sh
pip install pyarrow  # local-only, like torch
python3 -m pipeline.aggregate_sentiment_files \
    --parquet scores_gkg_2020_2024.parquet --out corpus/sentiment_quarter.csv
python3 -m pipeline.combine_axes \
    --sentiment corpus/sentiment_quarter.csv --out corpus/combined.csv
python3 evals/backtest.py --intersect \
    --scores corpus/combined.csv \
    --scores index/data/scores_gp50_fixed_v1.csv \
    --labels evals/items/distress_bank_quarter_full.csv --budget auto
```

The parquet is not in the repo (data stays out); it is reproducible
end-to-end: `backfill_gkg --to-csv` + `backfill_edgar --to-csv` →
`kaggle_score_backfill.ipynb` → this. Known corpus gaps, both recorded in
scoring/DESIGN.md 2026-08-14: ozk has no EDGAR history (FDIC filer), and
pnfp's history was supplemented under its pre-2025 CIK.

## Not yet swept

The item-level directional threshold is fixed at p≥0.5 here. The model
under-calls direction and `probs` is stored precisely so serving can lower
that bar (DESIGN, obligation 2) — the full grid (item threshold × negative
cut × ladder cutoffs) is the natural next run before declaring final
cutoffs.
