# Backtest evaluation protocol (v1)

Owner: Shu Han. Implements role task 2 before `evals/backtest.py`.

Ground truth: [`evals/items/distress_bank_quarter.csv`](items/distress_bank_quarter.csv)
([`evals/distress_definition.md`](distress_definition.md)).

**Status:** protocol locked. Harness: `python3 evals/backtest.py --smoke`.

---

## 1. What we are measuring

> Do risk scores rank bank-quarters that are about to be distressed
> (`distress_within_4q = 1`) above those that are not?

This is **not** the sentiment gold-set quality gate. Sentiment labels and
distress labels must never be mixed ([`evals/README.md`](README.md)).

**Target column:** `distress_within_4q`  
**Not the target:** `is_event_quarter` (contemporaneous event; early-warning
eval uses the lookahead label).

**Score convention:** higher score = higher predicted risk.  
Any “soundness” feature (e.g. tier-1 capital) must be **negated or inverted**
before ranking.

---

## 2. Metrics (accuracy is banned)

Positives are rare (~120 / 3,848 ≈ 3.1% under v1 labels; still skewed).
Predicting “no distress” every time looks excellent under accuracy and is
useless. Report only:

| Metric | Definition |
|---|---|
| **PR-AUC** | Area under the precision–recall curve over test bank-quarters, ranking by score. Prefer sklearn `average_precision_score`. |
| **Precision@k (pooled)** | Among the top `k` scored bank-quarters in the **entire test set**, fraction with `distress_within_4q = 1`. Default **k = 50**. |
| **Recall@budget** | Each test quarter, alert the top **B** banks by score. Recall = (# alerted positives) / (# positives in that quarter), then **micro-average** over quarters that have ≥1 positive. Quarters with zero positives are skipped for this metric (still included in PR-AUC / precision@k). Default **B = 10** on the 104-bank seed panel. On the full filer panel use **`--budget auto`**, which scales 10/104 to the median test-quarter size (~441). |
| **Baseline** | Always report the same metrics for a **random score** (fixed seed) and for the **naive tier-1 inverted** control, so a model number is interpretable. |

Do **not** report accuracy, ROC-AUC alone (can look strong under imbalance), or
F1 at an arbitrary 0.5 threshold unless the threshold is chosen only on train.

---

## 3. Time split (no random split)

Leakage rule: never put a later quarter in train and an earlier one in test.

| Split | Quarters | Positives (v1 CSV) |
|---|---|---|
| **Train** | `quarter_end_date <= 2021-12-31` | 88 |
| **Test** | `quarter_end_date >= 2022-01-01` and `<= 2024-12-31` | 32 |

- Test stops at **2024-12-31** so the window matches the planned GDELT backfill
  era (2020–2024) and does not lean on thin 2025–2026 tails.
- Ming may train the GP only on train rows; the harness scores **test** rows.
  If a method needs no fitting (naive tier-1), still restrict metrics to test.
- Do **not** use a random bank holdout as the primary split; bank identity
  correlates with repeated stress. Optional sensitivity: leave-one-bank-out
  later, not required for v1.

---

## 4. Evaluation windows (fair control)

A fundamentals-only number computed on a longer window than the text model is
**not** a control — it is a different experiment.

| Run | Score source | Rows included |
|---|---|---|
| **A. Harness smoke** | Naive: `risk = −tier1_capital_ratio` from Call Reports | Seed banks; test split above; join on `fdic_cert_number` + `quarter_end_date` |
| **B. Fundamentals control** | Ming’s GP score table | Same test split; drop rows missing a GP score |
| **C. Combined** | Agreed blend of GP + bank×quarter sentiment rollup | Intersection: row must have **both** scores; window further restricted to quarters where the sentiment corpus actually exists (after Jiwon’s 2020–24 backfill + scoring). Until that lands, **do not claim a combined result**. |

Join keys (value join, no FK):

- Labels: `fdic_cert_number`, `quarter_end_date`
- Fundamentals: `fact_call_report.fdic_cert_number`, `report_date`
- Live bank id (if needed): `bank.fdic_cert` ↔ `fdic_cert_number`

---

## 5. Input contract for `evals/backtest.py` (next unit)

Minimal score CSV / frame:

| Column | Meaning |
|---|---|
| `fdic_cert_number` | int |
| `quarter_end_date` | `YYYY-MM-DD` |
| `risk_score` | float, higher = riskier |
| `model_version` | string (e.g. `naive_neg_tier1`, `gp_v1`, `combined_v1`) |

CLI sketch (to implement next):

```bash
python3 evals/backtest.py \
  --scores path/to/scores.csv \
  --labels evals/items/distress_bank_quarter.csv \
  --split-date 2021-12-31 \
  --test-end 2024-12-31 \
  --budget 10 \
  --precision-at 50

# Fair multi-model table (shared test keys):
python3 evals/backtest.py --intersect --budget auto \
  --labels evals/items/distress_bank_quarter_full.csv \
  --scores index/data/scores_hgb_eng_v2.csv \
  --scores index/data/scores_gp50_fixed_v1.csv
```

Print a small markdown table to stdout (and optionally
`evals/reports/…_backtest.md`).

**First implementation milestone:** run A only (naive −tier1). If PR-AUC /
recall@10 are nonsense vs random, fix the harness before trusting Ming’s
numbers.

---

## 6. Reporting checklist

Every backtest write-up must include:

1. `model_version` and score definition  
2. Train/test cut dates and row / positive counts  
3. PR-AUC, precision@50, recall@budget=10 (micro)  
4. Same three metrics for random + naive −tier1  
5. Confirmation that combined runs used the **intersection** window (`--intersect` when comparing multiple score files)

---

## 7. Version

| Version | Date | Change |
|---|---|---|
| v1 | 2026-07-31 | Lock metrics, 2021/2022 split, test through 2024, naive-first harness plan |
