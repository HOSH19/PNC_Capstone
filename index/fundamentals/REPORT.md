# Bank distress early-warning model

A Gaussian Process model that predicts, from FFIEC Call Report data, the probability a bank is
closed by the FDIC within the next four quarters. Outputs a 0-100 score with confidence
intervals.

**In one line**: it works in credit-driven banking crises of the 2008-2016 kind — watching
**3.1%** of US banks captures **93%** of failures — and is **effectively blind** to the
rate/duration-driven failures of 2023. That blindness is explicable, and not something tuning
can fix.

---

## 1. Data sources

**Supabase is not the source.** The `fact_call_report` table there keeps 9 financial fields — a
trimmed view of the same official archive. This model re-downloads the full field set from that
archive. Same banks, same date range, far wider.

| Source | Used for | URL | Coverage | Fields |
|---|---|---|---|---|
| **FFIEC CDR Bulk Data** | **every model input** | https://cdr.ffiec.gov/public/PWS/DownloadBulkData.aspx<br>(via `ffiec_data_collector`, the same library the team's `unified_ffiec_fdic_dataset` uses) | **101 quarters, 2001Q1–2026Q1**<br>4,336–8,857 banks per quarter | **1,966 raw fields**<br>(9 in Supabase) |
| **FDIC Failures** | labels | https://banks.data.fdic.gov/api/failures | 1970–2026, 3,583 banks | `failure_date` |
| Supabase `fact_call_report` | its `rssd_id ↔ fdic_cert` mapping only | — | — | — |

**Excluded sources** (reasons in §8): FRED macro series, yfinance prices, CFPB complaints,
GDELT/EDGAR text, `dim_bank` static attributes.

**Modelling actually spans the 73 quarters from 2008Q1.** The extractor pulls 101 quarters back
to 2001Q1, but **not one row of 2001-2007 is currently used** — those quarters were lookback
padding for 8-quarter derived features, a direction that produced no improvement and was dropped
(§8). Setting `FFIEC_START=20080101` in `extract.py` saves roughly 40% of the download.

### 1.1 Unit of observation and reporting basis

**One row is one chartered bank in one quarter. Not a holding company.**

Call Reports are filed by the **bank**. Holding companies file a different form (FR Y-9C), which
is not collected here.

```
JPMorgan Chase & Co.              holding company  → FR Y-9C (not collected)
  └─ JPMorgan Chase Bank, N.A.    bank             → Call Report ← what we use
       ├─ US domestic offices                        RCON  domestic
       └─ foreign branches + bank subsidiaries        RCFN  foreign
                          RCFD = both combined        consolidated
```

**"Consolidated" in RCFD is consolidation at the bank level** — the bank's foreign branches and
its own subsidiaries — **not roll-up to the holding company**. The gap is large: Goldman Sachs
Bank USA reported **\$502B** in 2022Q2 while The Goldman Sachs Group was around \$1.6T.

| | |
|---|---|
| **The consolidated figure is reported by the bank**, not synthesised by us | The raw file carries `RCFD2170` (consolidated) and `RCON2170` (domestic) as **two columns**.<br>**The form changes over the years**: in 2010Q4 an 031 filer populated **both** (JPMorgan: \$1,631.6B and \$1,009.1B); by 2022Q2 only RCFD is filled and RCON is blank.<br>Extraction must therefore **take the consolidated column by priority** (`RCFD` ahead of `RCON` in `PREFIX_RANK`), not simply "whichever column has a value" |
| **Sibling banks under one holding company each get their own row** | Morgan Stanley Bank N.A. (RSSD 1456501, \$191B) and Morgan Stanley **Private** Bank N.A. (RSSD 2489805, **\$200B**) are separately chartered, file separately, and occupy separate rows |
| **Branches and subsidiaries never get their own row** | A foreign branch is not a separate legal entity and has no RSSD; a bank's own non-bank subsidiaries are consolidated into its RCFD.<br>**No row in the panel is "a branch" or "a subsidiary" — every row is a chartered bank** |
| **The magnitudes line up** | Total assets across the 4,821 banks in 2022Q2 sum to **\$23.73T**, consistent with the FDIC industry figure (\$23-24T).<br>⚠️ This confirms the order of magnitude but **does not prove there is no double counting** — the FDIC total is also built by summing institutions, so a bank-owns-bank structure would double-count on both sides. That structure is uncommon in the US but **has not been verified** (no ownership data on hand) |

**This model always takes the consolidated basis** (`RCFD` ahead of `RCON` in `PREFIX_RANK`).

That is load-bearing rather than incidental. Both columns hold values in some quarters (JPMorgan
2010Q4: \$1,631.6B and \$1,009.1B), and "whichever column has a value" resolves by column order —
for an 031 filer the two figures can differ by 60%.

> ⚠️ **Note**: Supabase's `fact_call_report` reads with **domestic first**
> (`first_value(rc, ["RCON2170", "RCFD2170"])`), a different basis from this model. This model
> does not depend on that table beyond the `rssd_id ↔ fdic_cert` mapping; do not compare the two
> sets of numbers directly.

> ⚠️ **For the dashboard** (`dashboard/` currently holds only a README and concept images):
> `holding_name` and `bank_legal_name` differ for 103 of the 104 tracked banks, and **the score
> belongs to the bank**. Displaying "Goldman Sachs" suggests the \$1.6T group; the entity scored
> is the \$502B Goldman Sachs Bank USA. Use `bank_legal_name`, or show both.
>
> ⚠️ The `rssd_id` column in the `bank` table sits next to `holding_name` but **holds the bank's
> RSSD**, not the holding company's.

### Modelling panel

| | |
|---|---|
| Rows | **435,264** (bank × quarter) |
| Banks | **8,655** (nationwide, including unlisted) |
| Feature quarters | 2008Q1 – 2026Q1 (73 quarters) |
| Positives | **1,934** (0.444%) |
| Prediction time | quarter end **+ 45 days** (FFIEC allows 30-35 days to file; 45 leaves a buffer) |
| Label window | **366 days** from the prediction time |

---

## 2. Feature screening

```
1,966  raw FFIEC fields
  ↓  ① stability: present in ≥90% of the MODELLED quarters
 628
  ↓  ② missingness vs label: pos/neg coverage within 10pt, both above 70%    → 446 pass
  ↓  ③ missingness vs size: top 1% vs the rest, within 10pt                  → 430 pass
  ↓  ④ type: drop identifiers, dates, near-constants (172 fields)
 369
  ↓  ⑤ correlation pruning at |Spearman| ≥0.95 (merged, not discarded)
 329  candidate features
```

**Normalisation**: everything is divided by total assets. Raw FFIEC items (**MDRM** is the
regulator's code for each report line — `RC_2170` is balance-sheet total assets) are dollar
amounts in thousands. The reported ratios live in schedule RC-R, and RC-R loses every field to
the stability screen because Basel III renumbered it in 2015. Size is preserved separately as
`log(total assets)`.

**Screens ② and ③ run on the first fold's training window only** (prediction time ≤ 2011-05-15),
never on a test year.

### ③ Why missingness needs a size axis of its own

Checking missingness against the label is not enough — **missingness can correlate with any
covariate, and size is the dangerous one**.

Banks with foreign offices file FFIEC 031 (`RCFD` prefix); domestic-only banks file 041
(`RCON`). If extraction fails to coalesce the two prefixes, large banks lose whole swathes of
fields — and that is **invisible on the label axis**: the affected banks are 1.8% of the panel,
overall coverage still reads 98%+, and they are almost all survivors, so positive and negative
coverage look identical.

The asymmetry in consequences is severe: **all 104 tracked banks are large**, so their scores
would come entirely from median imputation.

This screen compares the **top 1% against the rest** rather than deciles. FFIEC-031 filers are
only ~9.7% of the top decile, so a field missing for all of them shifts a decile gap by about
0.09 — under any sensible threshold. They are ~56% of the top 1%.

---

## 3. Model comparison

Walk-forward, 13 folds, test years 2012-2025 (2021 skipped — no positives), **405** test
positives in total.

**Fold construction** (test year 2022 as the example):

```
train    prediction time ≤ 2021-02-14   (= 2022-02-14, the first test prediction, minus 366 days)
         └ left edge fixed at 2008Q1, right edge advances each fold (expanding window)
embargo  2021-02-14 – 2022-02-14
test     every row whose prediction time falls in 2022
```

**Why the embargo is not optional**: the label asks whether a bank fails within the next 366
days. A row at 2021Q2 has a label window reaching into 2022 — training on it tells the model
which banks fail in the test year. Training must therefore stop where label windows have fully
closed, and the intervening year is discarded (it becomes legal in the next fold).

**Feature ranking, tier choice and calibration all happen inside the fold's training set.**

### Model naming

| Name | Meaning |
|---|---|
| **`gp`** | **Gaussian Process classifier**. A kernel method; emits uncertainty alongside the prediction |
| **`xgb`** | **XGBoost**, gradient-boosted trees. Handles all 430k rows and NaNs natively, but **emits no uncertainty** |
| **the number** | **how many features**. `gp3` is a GP over 3 features; `xgb50` an XGBoost over 50 |
| **`gp5+xgb10`** | **ensemble**: each model's probabilities converted to percentile ranks, then averaged |

How features are chosen: **each fold runs one XGBoost over all 329 candidates, sorts by gain,
and takes the top N**. `gp3` and `xgb50` therefore use the top 3 and top 50 of the same ordering.

### What happens inside each fold

```
for each test year Y:

 1. split      train = prediction time ≤ (Y's first prediction time − 366 days)
               test  = prediction time within Y
               everything between is discarded (embargo)

 2. rank       one XGBoost over all 329 candidates on the TRAINING SET, sorted by gain
               ⚠️ training rows only; the test year takes no part

 3. take N     xgb5/10/22/50/100 and gp3/5/8/12 all draw from the SAME ordering, so they compare

 4. fit        XGBoost: all ~400k training rows, NaNs handled natively
               GP:      every positive + 3,000 sampled negatives ≈ 4,900 rows (O(n³) limit)
                        missing values filled with training medians (GP rejects NaN) + StandardScaler
                        kernel = ConstantKernel(10) × Matern(ν=1.5, ℓ=√N×1.5)
                        optimizer=None — unbounded optimisation drives ℓ to ≈1e4 and the kernel collapses

 5. score      predict on the test set in batches of 5,000 (the full kernel matrix OOMs)

 6. metrics    computed twice: full sample, and the D9 large-bank subset
```

**The constraint that matters is step 2: ranking must happen inside the fold.** Ranking once on
the full panel and reusing it means the candidate list has seen every test year — every metric
downstream inflates, and the results give no sign of it.

### Which features each model actually received

The 13 folds rank independently, and the ordering shifts. The table counts **how many folds a
field placed inside that model's top N** — that is, how many folds actually used it. Since every
model takes the top N of one ordering, the models are nested.

Example: `RCCI_F159` sits around rank 4 — it misses `gp3`'s top 3 in all but 1 of 13 folds, yet
makes `gp5`'s top 5 in every fold.

| Rank | Field | Meaning (each ÷ total assets) | folds in<br>gp3's top 3 | folds in<br>gp5's top 5 | gp8's<br>top 8 | xgb10's<br>top 10 |
|---|---|---|---|---|---|---|
| 1 | `RI_4300` | **pre-tax income** | **13/13** | 13/13 | 13/13 | 13/13 |
| 2 | `RC_3210` | **total equity capital** | **13/13** | 13/13 | 13/13 | 13/13 |
| 3 | `RC_3632` | **undivided profits and capital reserves** | **12/13** | 13/13 | 13/13 | 13/13 |
| 4 | `RCCI_F159` | construction and land development loans | 1/13 | **13/13** | 13/13 | 13/13 |
| 5 | `RC_3123` | allowance for loan and lease losses | — | 4/13 | 11/13 | 12/13 |
| 6 | `RCE_2365` | total brokered deposits | — | 1/13 | 9/13 | 11/13 |
| 7 | `RCE_2236` | non-transaction deposits of foreign banks | — | 4/13 | 6/13 | 6/13 |
| 8 | `RCN_B576` | credit card loans past due 90 days+ | — | — | 5/13 | 8/13 |
| 9 | `RI_4073` | total interest expense | — | — | 5/13 | 6/13 |
| 10 | `RCL_8765` | spot foreign exchange contracts | — | 1/13 | 4/13 | 7/13 |

**The top 3 are fixed** (12-13 of 13 folds), rank 4 is stable from gp5 upward, and **rank 5 is
where it starts drifting** — gp5's fifth slot rotates between `RCE_2236` (4 folds), `RC_3123`
(4 folds) and five other fields.

That also explains why gp3 has the highest full-sample PR-AUC: **it uses only the fields that are
the same in every fold**, while anything below that is fold-dependent.

---

### What the metrics mean

| Metric | Definition | Why it is used |
|---|---|---|
| **PR-AUC** | area under the precision-recall curve | positives are 0.44% of rows, so ROC is diluted by the negatives and reads high; **this is the only metric comparable across folds** |
| **base rate** | share of the fold's test rows that are positive = **the hit rate of guessing** | 0.459% in 2012, 0.022% in 2018 — a factor of 20.7 |
| ROC | area under the ROC curve | overall ranking quality; optimistic under heavy imbalance |
| **top100 sum** | in each fold, take the 100 rows the model ranks riskiest and count the real failures; summed over 13 folds | closest to practice: "given a list of 100, how many do we catch" |
| **recall@500** | share of all positives sitting inside the top 500 rows | recall for a longer list |

| Model | Dim | **PR-AUC** | ROC | top100 sum | recall@500 |
|---|---|---|---|---|---|
| **gp3** | 3 | **0.293** | 0.950 | 179 | 80.5% |
| xgb50 | 50 | 0.281 | **0.973** | 169 | 80.0% |
| xgb100 | 100 | 0.276 | 0.973 | 161 | **83.2%** |
| gp8 | 8 | 0.254 | 0.958 | **182** | 75.8% |
| gp5+xgb10 ensemble | — | 0.252 | 0.959 | 179 | 77.3% |
| gp5 | 5 | 0.251 | 0.960 | 169 | 76.0% |
| xgb5 | 5 | 0.233 | 0.948 | 155 | 77.8% |

### 🔴 Every difference sits inside the noise

Per-fold PR-AUC; the last two rows are the weighted aggregate and each model's own fold-to-fold
standard deviation.

| Year | Pos | gp3 | gp5 | gp8 | xgb10 | xgb50 | xgb100 | ensemble | best |
|---|---|---|---|---|---|---|---|---|---|
| 2012 | 132 | **0.370** | 0.305 | 0.294 | 0.274 | 0.319 | 0.313 | 0.275 | gp3 |
| 2013 | 81 | **0.353** | 0.281 | 0.246 | 0.196 | 0.268 | 0.290 | 0.261 | gp3 |
| 2014 | 54 | 0.333 | 0.314 | 0.315 | 0.296 | **0.380** | 0.350 | 0.325 | xgb50 |
| 2015 | 21 | 0.090 | 0.099 | 0.086 | 0.106 | **0.159** | 0.148 | 0.108 | xgb50 |
| 2016 | 29 | 0.302 | 0.279 | 0.269 | **0.443** | 0.242 | 0.206 | 0.394 | xgb10 |
| 2017 | 12 | **0.207** | 0.083 | 0.130 | 0.118 | 0.150 | 0.132 | 0.102 | gp3 |
| 2018 | 5 | 0.034 | 0.048 | 0.126 | 0.133 | 0.031 | **0.444** | 0.070 | xgb100 |
| 2019 | 20 | 0.182 | 0.190 | 0.234 | 0.275 | **0.292** | 0.240 | 0.218 | xgb50 |
| 2020 | 8 | 0.444 | 0.487 | 0.557 | 0.505 | **0.581** | 0.503 | 0.430 | xgb50 |
| 2022 | 12 | 0.003 | 0.002 | 0.002 | **0.004** | 0.003 | 0.002 | 0.003 | xgb10 |
| 2023 | 12 | 0.026 | 0.013 | 0.014 | **0.149** | 0.005 | 0.009 | 0.039 | xgb10 |
| 2024 | 10 | 0.009 | 0.006 | 0.003 | 0.003 | 0.003 | **0.010** | 0.006 | xgb100 |
| 2025 | 9 | 0.273 | 0.237 | 0.655 | 0.554 | **0.722** | 0.690 | 0.398 | xgb50 |
| **weighted** | **405** | **0.293** | 0.251 | 0.254 | 0.251 | 0.281 | 0.276 | 0.252 | |
| **fold σ** | | 0.156 | 0.151 | 0.203 | 0.178 | 0.224 | 0.207 | 0.156 | |

| | |
|---|---|
| Mean of each model's own fold-to-fold PR-AUC σ | **0.183** |
| Spread of weighted PR-AUC across models (all 10 configurations, incl. xgb5) | **0.060** |

**Fold-to-fold variation is 3.0× the spread between models.** Fold winners: xgb50 five times,
xgb10 three, gp3 three, xgb100 twice — **no model wins even half the folds**.

→ **The noise floor for this report: models differing by less than roughly 20% cannot be told
apart.** Selection therefore cannot rest on accuracy; the criteria are in §4.

### The subset matching the tracked banks' size (D9, top asset decile)

The 104 tracked banks carry 7 positives in their entire history (all in 2008), so they cannot be
validated directly. The closest proxy is banks in the same size bracket — but that subset yields
only **6 evaluable folds and 30 positives** (3-9 per fold).

| Model | D9 PR-AUC | full-sample PR-AUC |
|---|---|---|
| gp5+xgb10 | 0.222 | 0.252 |
| gp8 | 0.212 | 0.254 |
| xgb10 | 0.204 | 0.251 |
| gp5 | 0.187 | 0.251 |
| xgb50 | 0.156 | 0.281 |
| gp3 | 0.144 | 0.293 |

**What this supports: the magnitudes match the full sample; no size-driven degradation is
visible.**

**What it does not support: this ordering.** D9's fold-to-fold σ is **0.226**, larger than the
full sample's 0.183, against a between-model spread of only 0.078 — variation is 2.9× the
difference. The same `gp3` scores 0.004 in the 2016 fold and 0.325 in 2023, a factor of 80.

> ⚠️ D9's `recall@500` is not comparable to the full sample's: a D9 fold holds 1,886-2,874 rows,
> so the top 500 is **17-27%** of it, against **1.7-2.7%** for the full sample. Different units;
> the column is omitted above.

---

## 4. Final model

**`gp3` is selected** — a Gaussian Process classifier over 3 features.

### Why this one

**The first step is not choosing on accuracy but accepting that accuracy cannot choose.** §3's
per-fold table shows fold-to-fold variation at 3.0× the spread between models, with none of the
10 configurations winning half the folds. Selection has to rest on something else:

| Criterion | Consequence |
|---|---|
| **1. Must emit uncertainty** | Eliminates every XGBoost configuration. `bank_index_score` carries `latent_mean`, `latent_var` and four interval columns; only a GP can fill them |
| **2. Score and interval must share a source** | Eliminates the `gp5+xgb10` ensemble — its score is an average of two models' ranks while the interval could only come from the GP side. The two would not be consistent |
| **3. Choose among the GPs** | `gp3` has the **highest full-sample PR-AUC (0.293** against gp5 0.251, gp8 0.254, gp12 0.251) and the narrowest intervals (median `latent_var` **0.088** against gp5 0.108, gp8 0.146, gp12 0.173) |

**Why fewer dimensions do better here**: the top 3 fields are identical in 12-13 of 13 folds and
rank 5 onward rotates (§3). gp3 uses only the stable ones; everything added below them is
fold-dependent.

**Note — the ordering differs on the D9 subset, but does not constitute evidence.** On the
subset matching the tracked banks' size, `gp3` scores 0.144 against gp5's 0.187. That ordering
cannot be relied on:

| | fold σ | weighted between-model spread | ratio |
|---|---|---|---|
| Full sample (405 positives) | 0.183 | 0.060 | 3.0× |
| **D9 (30 positives)** | **0.226** | 0.078 | **2.9×** |

D9's fold-to-fold variation exceeds the full sample's while each fold holds only 3-9 positives —
the same `gp3` scores 0.004 in 2016 and 0.325 in 2023, a factor of 80. A 30% gap between 0.144
and 0.187 is well inside that.

The cause is that **large banks rarely fail**: six folds yield 30 positives between them. That is
a limit on evidence, not evidence that the model is worse on large banks — but by the same token,
**its performance on large banks cannot be confirmed either**.

### Why these three features

The three are not independent indicators — they are the same quantity in three tenses:

```
equity / assets              how much cushion is left now          stock
pre-tax income / assets      whether the cushion is growing        flow
retained earnings / assets   how much cushion was ever built       accumulated flow
```

They separate cleanly, and they carry distinct information (pairwise Spearman 0.15-0.36, far
below the 0.95 pruning threshold):

| | Pre-failure median | Surviving median | Negative before failure | Negative while surviving |
|---|---|---|---|---|
| pre-tax income / assets | **−1.82%** | +0.45% | **87.6%** | 9.3% |
| equity / assets | **4.21%** | 10.43% | 2.5% | 0.1% |
| retained earnings / assets | **−4.07%** | +6.00% | **74.8%** | 9.6% |

Retained earnings is a component of equity, yet the two correlate at only 0.360 — the remainder
is paid-in capital and AOCI. A bank with healthy equity but negative retained earnings is being
kept alive by shareholder injections rather than by its own earnings, and that distinction
carries signal.

**But there is a more mechanical reason capital dominates, and it cuts both ways.**

FDIC closure is itself defined in capital terms. Under Prompt Corrective Action, a bank whose
tangible equity falls to 2% of assets is "critically undercapitalized", and regulators are
required to place it in receivership within 90 days. The prediction target — closure within 366
days — is therefore tied by statute to the very quantity these features measure.

That explains two things at once:

| | |
|---|---|
| **Why NPL, liquidity and concentration never reach the top 3** | They are *causes* of capital depletion. The model reads the outcome directly, one step closer to the target |
| **Why SVB is invisible** | SVB met its capital requirements when it was closed. It died of a deposit run, not of a PCA trigger — the model learned the statutory closure path, and SVB took a different one |

The second row is the sharper statement of §5.4: the limitation is not merely that unrealised
HTM losses bypass the income statement, but that **the model learned one legal pathway to
closure and 2023 ran through another**.

### Specification

| | |
|---|---|
| **Model** | Gaussian Process classifier, `ConstantKernel(10) × Matern(ν=1.5, length_scale=√3×1.5)`, `optimizer=None` |
| **Inputs (3, each ÷ total assets)** | `RI_4300` pre-tax income · `RC_3210` total equity capital · `RC_3632` undivided profits and capital reserves |
| **Training** | 399,394 rows / **1,919 positives**, feature quarters 2008Q1–2024Q1 |
| **Rows the GP actually sees** | **4,919** (every positive + 3,000 sampled negatives, an O(n³) limit) |
| **Calibration** | prediction times 2024-08 – 2025-05 (see below) |
| **Cutoff** | prediction time ≤ 2025-08-03, so label windows have closed; 17,642 later rows excluded |
| **Test set** | **none** — the performance claim is inherited from walk-forward |
| **Latest scoring run** | 2026Q1, 4,336 banks: 344 distress / 889 neutral / 3,103 sound |

### Two steps between the GP's output and the published score

```
raw GP probability
   ↓ ① Platt scaling — a logistic regression fitted on held-out data corrects the LEVEL of the
   ↓    probability to match observed frequency. It changes the level, not the ranking
calibrated probability
   ↓ ② quantile anchoring — the calibration set's 70th and 90th percentiles are mapped onto the
   ↓    90 and 80 band cutoffs.
   ↓    (1 − probability) × 100 cannot be used: at a 0.44% base rate the calibrated probability
   ↓    almost never exceeds 20%, so every bank would land between 92 and 96 and be rated sound
final score, 0-100
```

**Output columns**: `prob` (calibrated probability) · `score` 0-100 ·
`score_lo_80/hi_80/lo_95/hi_95` (interval bounds) · `band` · `latent_mean` / `latent_var` (the
GP posterior's mean and variance, from which the intervals derive) — matching the
`bank_index_score` schema in `db/migrations/012_index_tables.sql`.

> ⚠️ **Do not display the raw probability.** It runs systematically high whenever the calibration
> period's base rate differs from the scoring period's. Display the score and the band, which are
> relative rankings and unaffected by base-rate drift.
>
> ⚠️ The calibration set contains only **10 positives**, so calibration is not robust.

---

## 5. Results (gp3)

### 5.1 Bands against realised failure rates (417,622 rows with closed labels)

**Band rule** (the mentor's cutoffs): score **≥90 = sound**, **80-90 = neutral**, **≤80 =
distress**. "Closed label" means the row's 366-day window has elapsed and the outcome is known —
the three most recent quarters do not qualify and are excluded here.

| Band | Rows | **Realised failure rate** |
|---|---|---|
| **distress** | 51,518 | **3.508%** |
| neutral | 84,688 | 0.080% |
| sound | 281,416 | 0.019% |

The distress band's failure rate is **44×** the neutral band's and **185×** the sound band's,
monotone with no inversion. (Full-sample base rate is 0.444% — the share of rows that are
positive, i.e. the hit rate of guessing.)

### 5.2 What that means in practice (2012 fold — 132 positives, the best-evidenced fold)

The fold holds **7,421 banks, 61 of which actually failed**. Ranked by gp3's score:

| List | Distinct banks | Share of US banks | Failing banks caught |
|---|---|---|---|
| top 100 rows | 65 | 0.9% | 30 / 61 = **49.2%** |
| top 300 rows | 151 | 2.0% | 51 / 61 = **83.6%** |
| **top 500 rows** | **227** | **3.1%** | **57 / 61 = 93.4%** |
| top 1000 rows | 425 | 5.7% | 61 / 61 = 100% |

**Watching 227 banks — 3.1% of the industry — catches 93% of failures**, roughly 30× random.
Returns fall away sharply beyond that: nearly doubling the list from 227 to 425 banks adds only
4 more.

> Every externally-facing statement should be phrased in **banks**, not rows. A bank occupies 4
> rows a year, so "the top 500 rows" is really 227 banks — the former reads as though 500 need
> watching.

### 5.3 The confidence intervals are usable

Median `latent_var` is **0.088** at 3 dimensions. The 80% interval averages **4.5 points** wide,
the 95% interval **6.6**.

⚠️ **The interval measures how far a bank sits from the training data, not how likely the
prediction is to be right.** A wide interval means the bank resembles nothing the model has seen
and the score deserves little weight; **a narrow interval does not make the score trustworthy**.

### 5.4 🔴 SVB: the model never saw it

SVB, Signature and First Republic all carry complete data in the panel (176 of 187 non-null
features in each 2022 quarter) and are genuine positives in both the 2022 and 2023 folds.

| Bank | 2022Q1 | 2022Q2 | 2022Q3 | 2022Q4 (final filing) |
|---|---|---|---|---|
| **Silicon Valley Bank** | 981 | 2,508 | 3,742 | **5,034 / 18,846 (27th percentile)** |
| Signature Bank | 1,488 | 3,152 | 4,301 | 6,763 |
| First Republic | 3,817 | 4,430 | 5,707 | 6,294 |

**The closer the failure, the worse the model ranks them.**

The reason is accounting. The three inputs are pre-tax income, total equity and retained
earnings, and SVB was **profitable and well capitalised to the end**. It died of unrealised
losses on held-to-maturity securities — carried at amortised cost, so they **touch neither the
income statement nor equity** — and of an uninsured-deposit run. **Read through income and
capital, SVB is invisible by construction.**

The same model ranked the credit-driven REPUBLIC BANK **14 / 18,846** in the 2023 fold.

**Targeted test** (forcing in "HTM unrealised loss / assets" and "uninsured deposits / assets"):

| | gp3 | gp3 + the two fields |
|---|---|---|
| SVB 2022Q4 rank | 5,034 | **1,399** |
| SVB across 2022 | 981 → 2,508 → 3,742 | 1,101 → 1,042 → **1,044** |
| Median rank of the fold's positives | 2,830 | **1,468** |
| Aggregate PR-AUC | no improvement (0.003 → 0.004) | |

The signal is present and was being missed — but the bank **still does not reach the top 100**,
and aggregate metrics do not move. ⚠️ These two fields were chosen **with hindsight**, so their
2022-2024 performance is not an honest out-of-sample estimate.

---

## 6. Recommended backtesting practice

**1. Walk-forward with an embargo** (implemented in `models.py`)
Each fold's training set ends 366 days before the test period's first prediction time. The label
looks 366 days forward; without the embargo the model is simply told which banks fail in the test
year.

**2. Select on PR-AUC, not lift**

`lift = PR-AUC ÷ the fold's base rate`, and **base rates differ by a factor of 20.7 across the 13
folds** (0.022% to 0.459%) — failures grow rarer over time, from 132 positives among 28,729 rows
in 2012 to 5 among 22,498 in 2018.

Within a single fold it already misleads:

```
2012:  PR-AUC 0.370  ÷  0.459%  =  lift  80.6
2018:  PR-AUC 0.034  ÷  0.022%  =  lift 153.4
       ↑ ranking quality is 1/11 of 2012's       ↑ yet lift reads 1.9× higher
```

**Aggregation makes it worse** — the `1/base rate` factor shifts weight away from the folds with
the most evidence:

| Fold | Positives | Weight in the PR-AUC aggregate | Weight in the lift aggregate |
|---|---|---|---|
| **2012** | **132** (32.6% of all) | **32.6%** | **9.7%** |
| 2018 + 2020 + 2025 | 22 (5.4%) | 5.0% | **21%** |

Whether a fold holds 5 positives or 132, its contribution to lift flattens out to 6-10%. That is
how `xgb100` ends up first on lift (190.0) while placing third on PR-AUC (0.276).

**→ This report uses PR-AUC throughout.** Lift is reserved for explaining "how much better than
guessing" to a non-technical reader, and only within a single fold.

**3. Respect the noise floor**
405 test positives spread across 13 folds means **models differing by less than roughly 20%
cannot be distinguished** (derivation in §3). Improvements below that threshold do not belong in
a conclusion.

**4. Report in banks, not rows**
A bank occupies 4 rows a year; "the top 500 rows" is really 227 banks (§5.2).

**5. Prospective validation — the only clean check**
Write every scoring run into `bank_index_score` with a timestamp and revisit it a year later. It
costs nothing beyond getting the logging right now. **The final model has no test set of its
own**, and this is the only path that can validate it.

**6. Report a separate D9 figure for the tracked banks**
The 104 banks hold 7 positives in their entire history (all in 2008) and none after 2017, so they
cannot be validated directly. Same-size banks (D9, 30 positives) are the only available proxy —
state the evidence count alongside.

---

## 7. Where the model works and where it does not

Per-fold results vary enormously (PR-AUC 0.003 to 0.444), but **two entirely different effects
are mixed together** and have to be separated, or the conclusion comes out wrong.

| Year | Test rows | Positives | Base rate | PR-AUC |
|---|---|---|---|---|
| 2012 | 28,729 | **132** | 0.459% | **0.370** |
| 2013 | 28,195 | **81** | 0.287% | **0.353** |
| 2014 | 27,044 | **54** | 0.200% | **0.333** |
| 2015 | 25,786 | 21 | 0.081% | 0.090 |
| 2016 | 24,560 | 29 | 0.118% | 0.302 |
| 2017 | 23,502 | 12 | 0.051% | 0.207 |
| 2018 | 22,498 | 5 | 0.022% | 0.034 |
| 2019 | 21,527 | 20 | 0.093% | 0.182 |
| 2020 | 20,588 | 8 | 0.039% | **0.444** |
| 2022 | 19,348 | 12 | 0.062% | **0.003** |
| 2023 | 18,846 | 12 | 0.064% | 0.026 |
| 2024 | 18,430 | 10 | 0.054% | 0.009 |
| 2025 | 17,974 | 9 | 0.050% | 0.273 |

### 7.1 ✅ Years with enough events: consistently effective

| Positives | Folds | PR-AUC range | Spread |
|---|---|---|---|
| **≥50** | 3 (267 positives together) | **0.333 – 0.370** | **1.1×** |
| <20 | 7 | 0.003 – 0.444 | **171×** |

**Where there are enough events both to learn from and to measure against, performance is highly
consistent.** Those three folds hold 66% of all 405 test positives and are the firmest part of
the conclusion.

### 7.2 ⚠️ Years with few events: the metric is unreadable, which is not the same as failure

With 8-12 positives, one or two banks moving in the ranking can multiply PR-AUC. The evidence:

- **the 2020 fold has 8 positives and PR-AUC 0.444 — the best of all 13**
- 2025 has 9 positives and 0.273, also strong
- the Spearman correlation between positive count and PR-AUC is only **0.398**

**A poor result in one low-positive fold therefore proves nothing about the model.**

### 7.3 🔴 But 2022-2024's failure has evidence independent of sample size

**With 12 positives each: the 2017 fold scores 0.207 and the 2022 fold 0.003 — a factor of 70.**
Sample size does not explain that.

The decisive evidence is **within a single fold**. In 2023, same model, same 12 positives:

| Bank | Cause of failure | Rank |
|---|---|---|
| **REPUBLIC BANK** | credit losses eroding capital (**the familiar mode**) | **14 / 18,846** |
| Silicon Valley Bank | duration / run | 5,034 |
| First Republic | duration / run | 6,294 |
| Signature Bank | duration / run | 6,763 |

**Sample size cannot explain a difference inside one fold.** Same day, same data: the familiar
failure lands in the 0.07th percentile, the duration-driven ones in the 27th to 36th. The
discriminator is the **failure mode**, not the sample size.

### 7.4 Conclusion

> **The model has learned one real, reproducible failure mode** — credit losses eroding capital
> until it is gone. In years dominated by that mode (2008-2016) it is consistently effective:
> in the 2012 fold, watching **3.1%** of US banks (227) captures **93%** of failures.
>
> **It is effectively blind to rate/duration-driven failure**, for reasons rooted in accounting
> (§5.4).
>
> **Recent events are too few to establish or rule out contemporary validity** — 90%+ of
> positives fall in 2008-2013, the ten years after 2017 hold only 82, and 2021 holds none.

---

## 8. Other known limitations

1. **XGBoost sets the feature ranking and the GP would rank differently — but it does not change
   the outcome.** Both families share one gain ordering. Trees favour features that cut sharp
   thresholds; a GP favours smooth gradients, so the ordering might not transfer.

   Checked with **ARD** (2012 fold, 20 dimensions, `length_scale` bounded to 0.3-50):

   | | |
   |---|---|
   | Spearman correlation between the two rankings | **0.388** — genuinely different. `RI_4300` is XGBoost's 1st and ARD's 8th |
   | Top-3 overlap | 2/3 (both take `RC_3210` and `RCCI_F159`; XGBoost's third is `RI_4300`, ARD's is `RCN_B576`) |
   | **PR-AUC** | XGBoost's top 3 **0.370** vs ARD's top 3 **0.362** — a 2% gap, **far inside the noise floor** |
   | ARD across all 20 dimensions | 0.296, clearly worse |

   **The rankings differ; the results do not. Low dimension is what matters.** ARD also switched
   off `RCE_2365` (brokered deposits), `RCE_2236`, and every derivatives/FX/trading field —
   supporting the reading that those act as business-model proxies rather than risk signals.

   ⚠️ One fold only. The production GP does not enable ARD (`optimizer=None`); unbounded
   optimisation drives `length_scale` to ≈1e4.

2. **The 3,000-row negative sample was never tuned.** It gives the GP a 1,919:3,000 class balance
   against a true base rate of 0.44%. That ratio may matter more than which features are chosen,
   and it has not been tested.

3. **The 104 tracked banks cannot calibrate the model** → the score is a **relative ranking**,
   not a failure probability. Their lack of positives is also partly constructed: the seed list
   selects currently-listed banks, so large banks that did fail are absent by definition.
4. **Roughly 20% of failing banks are missed by every model** (three models cross-checked; the
   missed sets overlap heavily). Plausibly fraud, single-counterparty concentration, or parent
   distress — none of which leaves a trace in the Call Report.
5. **All of RC-R fails the stability screen**, so CET1, leverage and total risk-based capital
   ratios are absent from the candidate pool. `RC_3210` (equity ÷ assets) is the only proxy.
6. **`RCO_5597` (uninsured deposits) has 12.5% coverage**, which limits its usefulness.
7. **Excluded sources**: FRED (no bank dimension — it can only encode which year it is, and with
   90% of positives in 2008-2013 that trains a crisis detector); prices, news and complaints
   (only the 104 tracked banks = 1.7% of rows); `dim_bank` static fields (0.5% coverage for
   failed banks against 48% for survivors — the missingness itself leaks the label).
8. **Eight-quarter derived forms (QoQ, slope, self z-score, peer percentile) showed no
   improvement** and were dropped — which is why scoring needs only a bank's most recent quarter.
9. **The calibration set holds only 10 positives**, so calibration is not robust (§4).

---

## 9. Code

| File | Purpose |
|---|---|
| `extract.py` | Full FFIEC CDR extraction (101 quarters), coalescing RCFD/RCON across schedule parts |
| `features.py` | Panel construction, four screens, correlation pruning → `panel.parquet` |
| `models.py` | Walk-forward over 13 folds: 5 XGBoost tiers × 4 GP dimensions × ensemble × D9 subset |
| `final_model.py` | Full fit, calibration, scoring → `scores.parquet` |

```bash
python3 index/fundamentals/extract.py       # ~60 min
python3 index/fundamentals/features.py      # ~15 min
python3 index/fundamentals/models.py        # ~90 min
MODEL_DIM=3 python3 index/fundamentals/final_model.py
```

---

## 10. A reporting-basis issue in the existing tables (FYI, not a change request)

This model does not depend on Supabase's `fact_call_report` beyond its `rssd_id ↔ fdic_cert`
mapping, but the issue below affects every downstream consumer of that table, so it is recorded
here.

> **Why this matters**: a foreign branch is not a separate legal entity and is not ring-fenced —
> its losses are the parent bank's losses, and the FDIC resolves the whole charter. **"Will this
> bank fail" is only meaningful at the level of the whole legal entity**, so risk measures need
> the consolidated basis. The regulatory forms concede the point themselves: they require
> domestic **assets** to be reported but never domestic **capital** (across the 113 FFIEC-031
> filers in 2010Q4, not one reports `RCON3210`) — because loss-absorbing capacity is a property
> of the whole and cannot be split. See §10.0.1.

### 10.0 Terminology

| Prefix | Expansion | Meaning |
|---|---|---|
| **RCON** | Report of Condition, d**O**mestic | **domestic basis** — US offices only |
| **RCFN** | …**F**oreig**N** | **foreign basis** — foreign offices only |
| **RCFD** | …**F**ully consoli**D**ated | **consolidated** — domestic + foreign, the bank's complete figure |
| RCOA / RCFA / RCFW | the RC-R capital schedule's equivalents | RCOA is domestic; RCFA and RCFW are consolidated |

Banks filing **FFIEC 041** (no foreign operations, ~98%) report only RCON, where domestic equals
consolidated and **there is no distinction**. Banks filing **FFIEC 031** (foreign operations,
~1.8%, but all large) report two different figures, and **which one is taken matters**.

### 10.0.1 What the domestic basis leaves out

The domestic basis (RCON) **counts US offices only**. It excludes:

- **foreign branches** (Citibank's London and Singapore offices, for instance)
- **subsidiaries inside the bank's consolidated statements**, including foreign ones

For a bank with no foreign business the two are identical. For a large bank with foreign
operations the gap is the entire foreign business.

**Why risk measures need the consolidated basis**

A foreign branch is **not a separate legal entity** and is not ring-fenced — its losses are the
parent bank's losses, and the FDIC resolves the entire charter; there is no "close only the US
part". So for the question "will this bank fail", the denominator has to be the whole entity.

The regulatory forms already reflect this. Across the 113 FFIEC-031 filers in 2010Q4:

| Field | Consolidated | Domestic |
|---|---|---|
| Total assets `2170` | 113 / 113 reported | 113 / 113 reported |
| **Total equity `3210`** | **113 / 113 reported** | **0 / 113 — never reported** |

**There is no such thing as domestic equity.** A foreign branch has no capital of its own;
capital exists only at the level of the whole bank. The forms require domestic assets but never
domestic capital, precisely because loss-absorbing capacity is a property of the whole.

The consequence: reading on a domestic basis makes banks **look systematically safer** — capital
(consolidated only) divided by an understated domestic denominator inflates the capital ratio.

**Example 1 — total assets (JPMorgan, 2010Q4, a quarter where both columns are populated)**

```
RCFD2170  consolidated   $1,631.6B   ← the bank's full size
RCON2170  domestic       $1,009.1B   ← what the table stores
                         ─────────
difference                 $622.5B   = foreign branches + consolidated subsidiaries, 38% of the bank
```

**Example 2 — total loans (2022Q2, the five largest domestic-vs-consolidated gaps)**

| Bank | RCON domestic | RCFD consolidated | Understated by | Share |
|---|---|---|---|---|
| Citibank, N.A. | \$412.3B | **\$650.4B** | \$238.1B | **36.6%** |
| Banco Popular de Puerto Rico | \$15.6B | \$21.3B | \$5.7B | 26.9% |
| The Bank of New York Mellon | \$27.0B | \$32.4B | \$5.4B | 16.6% |
| State Street Bank and Trust | \$28.5B | \$33.8B | \$5.2B | 15.5% |
| JPMorgan Chase Bank, N.A. | \$986.2B | \$1,110.6B | \$124.4B | 11.2% |

**Example 3 — total deposits (2022Q2, where `RCFD2200` is unpopulated and domestic and foreign
are filed as separate columns)**

| Bank | RCON domestic | RCFN foreign | Actual total | Omitted |
|---|---|---|---|---|
| The Northern Trust Company | \$54.6B | **\$80.8B** | \$135.4B | **59.7%** |
| Citibank, N.A. | \$753.7B | \$600.1B | \$1,353.8B | **44.3%** |
| The Bank of New York Mellon | \$213.7B | \$110.2B | \$323.9B | 34.0% |
| State Street Bank and Trust | \$166.6B | \$80.6B | \$247.2B | 32.6% |
| JPMorgan Chase Bank, N.A. | \$2,128.5B | \$420.3B | \$2,548.7B | 16.5% |

Northern Trust holds **more deposits abroad than at home** — taking the domestic column alone
discards 60% of the bank's deposits. Custodians (BNY Mellon, State Street, Northern Trust) and
Citibank are hit hardest, since cross-border custody and clearing is what they do.

---

### 10.1 Root cause: the lookup order puts domestic first

Every `first_value` call in `unified_ffiec_fdic_dataset/scripts/ffiec_call_reports.py` lists
RCON (domestic) ahead of RCFD (consolidated), and `first_value` returns **the first non-empty
value** — so the consolidated figure is only reached when the domestic column is blank. It is a
fallback rather than the preference.

### 10.2 Affected tables

| Table | Affected | Note |
|---|---|---|
| **`fact_call_report`** | 🔴 all 9 numeric fields | produced by `ffiec_call_reports.py` |
| **`fact_bank_quarter`** | 🔴 inherits all of it | `modeling.py` reads `fact_call_report.csv` directly |
| `dim_bank` / `fact_distress_event` | ✅ unaffected | sourced from FDIC, not FFIEC |

### 10.3 Field by field: current logic → what it should be

Script: `unified_ffiec_fdic_dataset/scripts/ffiec_call_reports.py`. `first_value` takes the first
non-empty value, so **order is priority**. Every call currently leads with RCON.

| Field | Current order | **Should be** | Affected, 2022Q2 |
|---|---|---|---|
| `total_assets` | `RCON2170` → `RCFD2170`<br>fallback `RCOA2170` → `RCFA2170` | `RCFD2170` → `RCON2170`<br>fallback `RCFA2170` → `RCOA2170` | 0 of 104 (RCON blank that quarter)<br>all wrong across 2008Q2-2010Q4 |
| `total_deposits` | `RCON2200` → `RCFN2200` → `RCFD2200` | `RCFD2200`; **when blank, `RCON2200 + RCFN2200`** (a sum, not a choice) | **22 / 104** |
| `tier1_capital_ratio` | `RCOA7206` → `RCFA7206` → `RCFW7206` → `RCON7206` → `RCFD7206` | `RCFA7206` → `RCFW7206` → `RCFD7206` → `RCOA7206` → `RCON7206` | not measured per quarter |
| `total_capital_ratio` | same, with 7205 | same, with 7205 | not measured per quarter |
| `npl_ratio` | (`RCON1407`→`RCFD1407`) + (`RCON1403`→`RCFD1403`)<br>÷ (`RCON2122`→`RCFD2122`) | **`RCFD` first** in all three | **21 / 104**<br>denominator understated → ratio **too high** |
| `loan_loss_allowance_ratio` | (`RCON3123`→`RCFD3123`) ÷ (`RCON2122`→`RCFD2122`) | same | **21 / 104**<br>**too high** |
| `liquidity_ratio` | (`RCON0071`→`RCFD0071` + `RCON0081`→`RCFD0081` + `RCON1773`→`RCFD1773`) ÷ total assets | **`RCFD` first** in all four | **10 / 104**<br>numerator understated → **too low** |
| `securities_unrealized_loss` | (`RCON1773`→`RCFD1773` − `RCON1772`→`RCFD1772`)<br>+ (`RCON1771`→`RCFD1771` − `RCON1754`→`RCFD1754`) | **`RCFD` first, and each subtraction must use one basis** | **30 / 104**<br>🔴 **sign can invert** |
| `cre_loans` | `RCON2746`→`RCFD2746` + `RCONF158`→`RCFDF158`<br>+ `RCONF160`→`RCFDF160` + `RCONF161`→`RCFDF161` | **`RCFD` first** in all four | 3-15 / 104<br>total **understated** |

### 10.4 Two that reordering alone will not fix

**① `total_deposits`** — in 2022Q2 `RCFD2200` (consolidated) is **unpopulated across the board**,
with domestic and foreign filed as separate columns. Reordering still lands on `RCON2200`. The
fix: when the consolidated column is blank, take `RCON + RCFN`.

```
Northern Trust 2022Q2   domestic $54.6B + foreign $80.8B = $135.4B
                        the table holds $54.6B — 59.7% missing
```

**② `securities_unrealized_loss`** — four independent `first_value` calls, so fair value may come
from the consolidated column while cost comes from the domestic one. The subtraction is then
neither a gain nor a loss:

```
RSSD 210434   table shows +12,135M (gain)   correct −1,187M (loss)   ← sign inverted
RSSD  35301   table shows  +3,294M          correct −4,159M          ← sign inverted
```

The fix: establish which basis the row uses, then take all four items on that basis.

### 10.5 The historical window

Between 2008Q2 and 2010Q4, `RCON2170` was also populated for 031 filers, so `total_assets` is
wrong for those 11 quarters as well: JPMorgan 2010Q4 reads \$1,009.1B where the correct figure is
\$1,631.6B, and the following quarter jumps to \$1,723.5B — **+71% in a single quarter**, a change
of basis rather than of business.

**The forms change over the years** — 031 filers populated both columns in the early period and,
for some items, only RCFD more recently. "Whichever column has a value" therefore happens to be
right in some quarters and wrong in others, which is exactly why the consolidated basis has to be
selected explicitly by priority.
