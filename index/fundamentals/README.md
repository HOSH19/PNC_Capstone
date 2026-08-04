# index/fundamentals — fundamentals axis

Predicts the probability that a bank is closed by the FDIC within 366 days, from FFIEC Call
Report data. Outputs a 0-100 score with confidence intervals.

**Model** — `gp3`: a Gaussian Process classifier over 3 features (pre-tax income, total equity,
retained earnings, each divided by total assets)
**Result** — on the 2012 fold, watching 3.1% of US banks (227 of them) captures 93% of that
year's failures
**Boundary** — effectively blind to rate/duration-driven failures like 2023. The reason is
accounting, not tuning.

Full findings in **[REPORT.md](REPORT.md)**.

> This is the **fundamentals axis** of the stability index. The combined index described in
> `index/README.md` also needs a sentiment axis (Jiwon's), and the combination step is not built
> — sentiment data is close to zero inside the modelling window.

---

## What's here

### Pipeline — four scripts, run in order

| Script | What it does | Output | Runtime |
|---|---|---|---|
| **`extract.py`** | Downloads quarterly FFIEC CDR archives and parses 17 schedules, **coalescing RCFD/RCON prefixes by MDRM item** (consolidated wins) | per-quarter parquet, 1,966 raw fields | ~60 min |
| **`features.py`** | Builds the labelled panel, applies four screens (stability / missingness-vs-label / **missingness-vs-size** / type) and correlation pruning | `panel.parquet`, 329 candidate features | ~15 min |
| **`models.py`** | Walk-forward over 13 folds: 5 XGBoost tiers × 4 GP dimensions × an ensemble, scored on the full sample and on the large-bank subset | `summary.csv`, `per_fold.csv`, `stability.csv` | ~90 min |
| **`final_model.py`** | Fits the chosen model on every row whose label window has closed, calibrates, and scores | `scores.parquet`, `final_model.json` | ~5 min |

```bash
python3 index/fundamentals/extract.py            # FFIEC_START=20080101 cuts ~40% of the download
python3 index/fundamentals/features.py
python3 index/fundamentals/models.py
MODEL_DIM=3 python3 index/fundamentals/final_model.py
```

`score.py` maps a calibrated probability onto the published score. `pipeline/db.py` supplies the
database connection.

### Three decisions that carry the design

| | |
|---|---|
| **Consolidated basis wins** | Banks with foreign offices file FFIEC 031 (`RCFD` prefix); domestic-only banks file 041 (`RCON`). Both columns of the same item are populated in some quarters, so extraction has to choose by priority — otherwise large banks lose 30-60% of their balance sheet. See REPORT §1.1 |
| **Missingness is screened against size, not only against the label** | A field can show 98%+ coverage with no label gap while every large bank is missing it: the affected banks are 1.8% of the panel and nearly all survivors. This screen compares the **top 1%** by assets against the rest, because FFIEC-031 filers are ~56% of the top 1% but only ~9.7% of the top decile |
| **Selection happens inside each fold** | Feature ranking, tier choice and calibration all use that fold's training rows only. Ranking once on the full panel lets the candidate list see every test year — everything downstream inflates, and nothing in the results reveals it |

---

## Running in production

Scoring has three parts. The scripts currently read and write local parquet; wiring them to the
database and to Actions is not yet implemented.

### 1. Input table

The model needs four FFIEC fields. They live separately from `fact_call_report`, which has its
own consumers and reads values on a different basis (REPORT §10).

```sql
create table <name tbd> (
  rssd_id            integer not null,
  report_date        date    not null,
  total_assets       double precision,  -- RC_2170, consolidated preferred
  pretax_income      double precision,  -- RI_4300
  total_equity       double precision,  -- RC_3210
  retained_earnings  double precision,  -- RC_3632
  primary key (rssd_id, report_date)
);
```

About 17 MB. The collector reuses `extract.py`'s prefix-coalescing logic and keeps these four
fields.

### 2. Frozen training set

**The training cutoff is fixed at 2024Q1**, which makes the 4,919 training rows (1,919 positives
plus a 3,000-row negative sample at `seed=0`) a fixed set. They are committed alongside the four
parameter groups derived from them:

```
train_sample.parquet    4,919 rows × 3 features + label       ~200 KB
frozen_params.json      imputation medians · StandardScaler ·   <1 KB
                        Platt calibrator · score anchors
```

Scoring refits from that 200 KB in about 60 seconds. Fixed data plus a fixed seed makes the
result exactly reproducible, which is what keeps scores comparable across quarters.

A GP's predictions depend on its entire training set plus an n×n matrix — roughly 194 MB once
serialised. Storing the seed data instead is three orders of magnitude smaller, survives
scikit-learn upgrades, and can be rebuilt from source.

**Upgrading the model**: move the training cutoff, regenerate the frozen files, and bump
`model_version` from `gp3-2024Q1`. That column marks which scores came from the same yardstick.

### 3. Quarterly scoring

```
1. pull the latest FFIEC quarter → 4 fields → upsert input table     ~3 min
2. read the frozen training set → fit the GP                         ~1 min
3. score the latest quarter → write bank_index_score                 ~1 min
```

An Actions runner has 7 GB, comfortably above the 194 MB kernel matrix. Same shape as the
existing `fundamentals.yml`.

Output tables `bank_index_score` and `bank_index_feature` are defined in
`db/migrations/012_index_tables.sql`, on this branch. Their columns match the model's outputs;
note that they key on `(fdic_cert_number, quarter_end_date)` while the model works in
`rssd_id` plus prediction date, so the writer needs one mapping step.

---

## Notes for consumers

| | |
|---|---|
| **Scores belong to the bank, not the group** | Goldman Sachs Bank USA is \$502B; The Goldman Sachs Group is ~\$1.6T. 103 of 104 tracked banks have a different `holding_name` and `bank_legal_name` — display the latter |
| **Show the score and band, not the raw probability** | When the calibration period's base rate differs from the scoring period's, the probability runs high. The score is a relative ranking and is unaffected |
| **A wide interval does not mean unreliable, and a narrow one does not mean reliable** | `latent_var` measures how far a bank sits from the training data, not how likely the prediction is to be right |
| **The final model has no test set** | Its performance claim is inherited from walk-forward. Write every scoring run with a timestamp so it can be checked a year later — that is the only clean validation path |
