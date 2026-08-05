# index/fundamentals — fundamentals axis

Predicts the probability that a bank triggers a distress event within the next four
quarters, from FFIEC Call Report data. Emits a 0–100 score with confidence intervals
and a risk band.

The distress event is defined in `evals/distress_definition.md` (branch `SH`). This
axis does not change the rule — it only widens the bank scope from the 104 seed banks
to every filer, because 104 banks yield 120 positives, too few to train on.

**Current model** `gp50@fixed` — a Gaussian Process classifier over 50 fixed features.

**Performance and boundaries are in [REPORT.md](REPORT.md)**: §5 results · §6 why this
model · §7.3 why the 104-bank scope cannot measure it.

---

## What this adds

### Pipeline — four scripts, run in order

| Script | What it does | Output | Runtime |
|---|---|---|---|
| `extract.py` | Downloads quarterly FFIEC archives, parses 17 schedules, coalesces prefixes by MDRM item | per-quarter parquet, 1,576 raw fields | ~60 min |
| `features.py` | Builds the labelled panel, five screens | `panel.parquet`, 529 candidate features | ~15 min |
| `models.py` | 6-fold walk-forward, 5 XGBoost tiers × 6 GP dimensions + an ensemble | `summary.csv` · `per_fold.csv` · `oos_predictions.parquet` | ~25 min |
| `final_model.py` | Fits on label-closed rows, calibrates, scores | `scores.parquet` · `final_model.json` | ~5 min |

```bash
python3 index/fundamentals/extract.py            # FFIEC_START=20170101 fetches only the 37 quarters needed
python3 index/fundamentals/features.py
python3 index/fundamentals/models.py             # DROP_SAMESOURCE=1 runs the label-adjacent-fields control
MODEL_DIM=50 python3 index/fundamentals/final_model.py
```

`score.py` maps a calibrated probability onto the published score and band;
`pipeline/db.py` supplies the database connection.

### Deliverables — two

| # | File | For | Purpose |
|---|---|---|---|
| **1** | `scores_gp50_fixed_v1.csv` | **backtest (Shu Han)** | feeds `evals/backtest.py` directly |
| **2** | `final_model.json` | this axis | reproduction and version tracking |

**The dashboard has no data yet.** `final_model.py` produces `scores.parquet`
(167,872 rows of scores / intervals / bands), but **nothing writes it into
`bank_index_score`** — that belongs to the deployment path, which is out of scope
here. The dashboard reads the table, not a file; until the loader exists, that table
is empty.

---

#### 1 · `scores_gp50_fixed_v1.csv` — backtest (Shu Han)

65,686 rows / 4,925 banks / 2021-12-31 to 2025-03-31. Four columns matching the input
contract in `evals/backtest_protocol.md`: `fdic_cert_number` · `quarter_end_date` ·
`risk_score` (higher = riskier) · `model_version`.

The bank coverage is far wider than the labels CSV, but the harness joins on
`(fdic_cert_number, quarter_end_date)`, so the extra rows fall away on their own —
**no pre-filtering needed**. The default window (`--split-date 2021-12-31`,
`--test-end 2024-12-31`) is fully covered: all 12 quarters, all 104 banks.

```bash
python3 evals/backtest.py \
  --scores <path>/scores_gp50_fixed_v1.csv \
  --labels evals/items/distress_bank_quarter.csv \
  --split-date 2021-12-31 --test-end 2024-12-31 --budget 10 --precision-at 50
```

**Read REPORT §7.3 before running** — that window holds only 29–32 positives, so the
resulting number is uninformative in either direction. Moving `--split-date` earlier
requires refitting a model and regenerating the scores (~3 min), because the current
model has seen everything before 2021Q4.

#### 2 · `final_model.json` — reproduction and version tracking

The 50 feature names, Platt calibration parameters, score anchors, and training window.
Compare against it when re-freezing to decide whether `model_version` should increment.

---

### Two models — do not mix them

`scores.parquet` and `scores_gp50_fixed_v1.csv` come from **models with different
training cutoffs**. Both GPs fit on only 5,000 rows (2,000 positive + 3,000 negative);
what differs is the pool they are drawn from:

| | backtest | dashboard |
|---|---|---|
| Sampling pool | report quarters 2017Q1–**2021Q3**, 102,186 rows | report quarters 2017Q1–**2024Q1**, 149,644 rows |
| Actually fitted on | 5,000 rows | 5,000 rows |
| Scores | everything after 2021-12-31 | all 167,872 rows |

The two purposes pull in opposite directions: the dashboard wants **as much training
data as possible**; the backtest requires a model that **has not seen the test period**.
Running the backtest against the production model measures memorisation; scoring the
dashboard with the backtest model throws away three years of data.

**Scores in `scores.parquet` for 2024Q1 and earlier are in-sample** — the model saw
those rows during training. A dashboard time series must not use them; use the
out-of-fold predictions in `oos_predictions.parquet` instead.

---

## Four decisions that carry the design

| | |
|---|---|
| **Widen the bank axis, not the time axis** | The NPL leg depends on `RCON1403` / `RCON1407`, which first appear in 2017Q1. Reaching back further leaves that leg permanently unable to fire, turning the label into a deposit-leg-only variant that means something different from the post-2017 one |
| **Deposits are domestic + foreign, summed** | `RCON2200` (domestic offices) and `RCFN2200` (foreign offices) are complements, not alternatives. Taking one by priority understated deposits for 44 of the 200 largest banks on 2021Q1, five of them to zero |
| **The feature list is fixed, not re-ranked per fold** | Ranked once on the first fold's training window (2017Q1–2018Q3); every fold and production use that same list. It is derived only from data preceding any test period, so the walk-forward numbers describe the model that ships — and the collector can fix 51 fields permanently |
| **The last four quarters are dropped** | A label needs four forward quarters to exist. The final quarters of any extract do not have them, so their positives are silently recorded as 0 (positive rate falls 10.9% → 8.4% → 6.0% → 3.3% → 0%). Those 17,642 rows are dropped rather than left in the panel labelled 0 |

---

## Reading these scores

| | |
|---|---|
| **Intervals do not indicate band membership** | The 80% interval averages 16.3 points while a band spans 10, so 67.2% of banks have an interval crossing two bands or more (21.0 points among the 104 tracked banks). Width does carry meaning, and its direction is positive — the widest quartile has roughly twice the event rate of the narrowest — but it expresses distance from the training sample, not confidence in the band |
| **Do not display raw `distress_prob`** | Raw probabilities overstate by 2–4×; they must pass through `score.py`'s Platt calibration and quantile anchors. Ranking is unaffected |
| **The 104 seed banks cannot validate this model** | That window holds 29–32 positives; the resulting number is uninformative in either direction. Judge this axis on the full sample or on size tiers |
| **The final model has no test set** | Its performance claim is inherited from walk-forward. The only clean check is to write every scoring run with a timestamp and revisit a year later |
| **All 50 features are raw MDRM codes** | `RCL_3814`, `RCRII_S442`, and so on — no readable names, not suitable for per-feature display to end users |

---

## To do: deployment (next batch)

This round ships the model and the report only; the scoring path is not built.

**When to start**: once the team has read REPORT §5–6 and agreed the performance is
acceptable. **Do not gate on the §7.3 backtest number** — that scope holds 29–32
positives, so it cannot come back "good" or "bad" in any meaningful sense.

Items 1, 3 and 6 depend only on the 51-field list, which is already fixed; they do not
wait on the model being final.

### Six things to write

| # | Item | Notes |
|---|---|---|
| 1 | **Input-table migration** `014_index_model_input.sql` | 51 fields (50 features + `RC_2170` total assets as denominator) plus `rssd_id` and `report_date`. **Watch the number**: 012 is this axis's output table (already on the branch), 013 is taken by the OCC table on branch `SH` |
| 2 | **Frozen training set** | Generate `train_sample.parquet` (5,000 rows × 50 features, ~2 MB) and `frozen_params.json` (medians / scaler / Platt / anchors), then commit both |
| 3 | **Collector** `pipeline/poll_ffiec_model_input.py` | Pull the latest FFIEC quarter → reuse `extract.py`'s prefix-coalescing → keep those 51 fields → upsert into the input table |
| 4 | **Scoring script** `index/fundamentals/score_quarter.py` | Read the latest quarter plus the frozen training set → fit the GP (~60 s) → map through `score.py` → write `bank_index_score` |
| 5 | **Historical backfill** | A dashboard time series needs history |
| 6 | **Workflow** `.github/workflows/index_score.yml` | Quarterly schedule, same shape as the existing `fundamentals.yml` |

### Design notes

**The input table needs 51 fields, not all 529.** The full pool is only needed when
re-freezing the model, and `extract.py` can regenerate it from source then — no
long-term storage required.

**Why freeze a training set instead of saving the model.** A GP's predictions depend
on its entire training set plus an n×n kernel matrix — roughly 194 MB serialised, and
fragile across scikit-learn versions. Committing 5,000 training rows (~2 MB) plus a
parameter file lets each quarter refit in about 60 seconds. **Same rows, same seed →
identical fit every time**, which is what keeps scores comparable across quarters;
resampling each quarter would let scores drift for reasons unrelated to the banks.

**Quarterly flow** pull the latest quarter → keep 51 fields → read the frozen sample
and fit → score → write `bank_index_score`. To upgrade, move the training cutoff,
re-freeze, and increment `model_version` — that column marks which scores came from
the same yardstick.

### Two decisions to make first

**1. Which scores to backfill (item 5).** The production model is in-sample for 2024Q1
and earlier, so backfilling directly makes the dashboard's history look better than it
is. Options: backfill only the out-of-sample quarters, use the out-of-fold predictions
in `oos_predictions.parquet`, or backfill and record the provenance in the table.

**2. Refit frequency.** The current model trains through 2024Q1 and goes stale with
age. The test noted in REPORT §8.1 — how much annual folding wastes by holding the
`cut` constant across a fold — has not been run, and its answer decides whether to
refit quarterly or annually.
