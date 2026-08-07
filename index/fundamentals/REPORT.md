# Fundamentals axis: distress-state warning model

Predicts the probability that a bank triggers a distress event within the next four
quarters, from FFIEC Call Report data, and emits a rankable risk score with
uncertainty intervals.

**The distress event is not defined by this report** — it comes from
`evals/distress_definition.md` (branch `SH`) and is used unchanged; only the bank
scope is widened. **Backtesting is also out of scope**, owned by `evals/backtest.py`;
what this axis delivers is scores (§7).

> An earlier version targeted official FDIC failure labels instead; it is kept on the
> branch `ming/v3-fundamentals`. The two have different targets and base rates differing
> by 20×, so their metrics are not comparable; no number from it appears here.

**What to read, by role:**

| Role | Must read |
|---|---|
| dashboard (Rita) | **§7.1 "Two columns this model cannot fill"** — leave `threshold_text` and `status` NULL, no schema change; everything else renders as usual |
| backtest (Shu Han) | **§7.3** — this scope cannot measure the model; the number is uninformative in either direction |
| maintainer of `fact_call_report` | **§8.3** — three reporting-basis issues, one of them a data error. No impact on this axis; noted for reference |

---

## 1. Target

### 1.1 The rule

An event fires in quarter Q when either leg holds:

| Leg | Condition | Fired |
|---|---|---|
| Deposit outflow | `(dep_Q − dep_{Q−1}) / dep_{Q−1} ≤ −10%` | 1,883 |
| NPL spike | `npl_Q / npl_{Q−1} ≥ 1.5` **and** `npl_Q > 2%` | 2,614 |

When an event fires at Q, the label is 1 on the **four preceding quarters** Q−1 … Q−4,
not on Q itself; but Q can still be 1 if another event falls within the four quarters
after it. Equivalently: a row's label asks whether an event fires in the next four
quarters.

**4,446** events in total, 51 of which fire both legs.

### 1.2 The only change: widening the bank scope

| | Original definition | This report |
|---|---|---|
| Banks | 104 seed banks | **all 5,994 filers** |
| Time | 2017Q1 onward | 2017Q1 onward (same) |
| Source | `fact_call_report` | FFIEC source archives |

**Widening the bank axis**: 104 banks yield **120 positives**, too few to train on;
widening gives **16,130**. The 104-bank restriction exists so the *combined* evaluation
has sentiment text coverage — a constraint on evaluation, not on training.

**Not widening the time axis**: the NPL leg depends on `RCON1403` / `RCON1407`, which
**first appear in 2017Q1**. Reaching back further leaves that leg permanently unable to
fire, turning the label into a deposit-leg-only variant.

### 1.3 Implementation check

Feeding this implementation's rule code the original author's own inputs
(`fact_call_report` fields, 104 banks, 2017 onward) reproduces
`evals/items/distress_bank_quarter.csv` **row for row**: 33 events, 120 positives,
`event_reason` identical throughout.

Recomputed on the FFIEC source archives, agreement with that CSV:

| | Banks | Agreement |
|---|---|---|
| Domestic-only filers | 90 | 99.88% |
| With foreign offices | 15 | 98.20% |
| **All** | **105** | **99.64%** |

The residual 0.36% traces entirely to input basis differences (§8.3), not to the rule.

---

## 2. Data

| | |
|---|---|
| Source | FFIEC CDR bulk archives, 17 schedules |
| Panel | **167,872 rows / 5,994 banks / 2017Q1–2025Q1** |
| Positives | **16,130 rows (9.61%) / 2,326 banks** |
| Prediction time | report date + 45 days (FFIEC allows 30–35 days to file) |

**The last four quarters are dropped.** A label needs four forward quarters to exist;
the final quarters of any extract do not have them, so their positives are silently
recorded as 0 (the last five quarters run 10.9% → 8.4% → 6.0% → 3.3% → 0%). Those
17,642 rows are dropped rather than left in the panel labelled 0.

**Deposits are domestic plus foreign, summed.** `RCON2200` (domestic offices) and
`RCFN2200` (foreign offices) are complements, not alternatives; taking one by priority
loses half the deposit base for banks with foreign operations. Extraction decides per
row: use the consolidated column where filed, otherwise sum the two. The accounting
identity `noninterest-bearing + interest-bearing = total deposits` then holds for every
bank with foreign deposits.

---

## 3. Features

| Step | Remaining |
|---|---|
| FFIEC raw fields | 1,576 |
| [1] Stability: present in ≥90% of modelled quarters | 1,198 |
| [2] Missingness vs label: coverage gap ≤10%, both sides >70% | 906 |
| [3] Missingness vs size: top 1% vs the rest, gap ≤10% | 866 (**40 caught only here**) |
| [4] Drop identifiers / dates / near-constants (388 across the 1,198; 226 not already dropped above) | 640 |
| [5] Correlation pruning \|ρ\| ≥ 0.95 (merged, not discarded), plus the derived column `log_assets` | **529** |

Screen [3] is not optional: a field can show 98% overall coverage with no gap between
positives and negatives while being missing for every large bank — the label axis
cannot see that class.

Every **label-touching** screen (step [2], and the AUC used to pick cluster
representatives) is restricted to the first fold's training window (prediction time
≤ 2019-02-13, i.e. report quarters 2017Q1–2018Q3, 40,034 rows / 4,197 positives),
exactly matching that fold's training set and touching no test year.

### 3.1 Fields sharing a source with the label

The label is built from deposits and NPL, so fields reporting those same quantities are
partly **definitional** rather than predictive — the event test at Q+1 uses the feature
quarter's own value as its denominator. This is not leakage (no future information is
used), but it changes what a high score means.

| Schedule / item | Count |
|---|---|
| RC-N past due and nonaccrual | 73 |
| RC-E deposit liabilities | 36 |
| RC-K quarterly averages (incl. deposits) | 15 |
| RC-O deposit insurance | 13 |
| RI-B II allowance roll-forward | 4 |
| Balance-sheet deposit items `RC_2200` / `RC_6631` / `RC_6636` | 3 |
| **Total** | **144 / 529 = 27%** |

That last row cannot be identified by prefix alone: `RC_2200` is total deposits — the
literal input to the deposit-outflow leg — but it sits on Schedule RC rather than RC-E.
Treatment: keep them and model normally; a separate control run drops all 144, and the
gap between the two arms appears in §8.1.

---

## 4. Validation design

**A fold is numbered by the year of the prediction time, not the report date.** A
filing becomes public 45 days after quarter end, so a 2019Q4 report belongs to the
"2020 fold". One row is: that bank-quarter's 529 fields (each divided by total assets),
labelled by whether an event fires in the following four quarters.

```
train    prediction time ≤ (fold's earliest prediction time − 366 days)
gap      the intervening year is discarded ← embargo
test     every row whose prediction time falls in the fold's year
```

| Fold | Training quarters | Train | Test quarters | Test |
|---|---|---|---|---|
| 2020 | 2017Q1–2018Q3 | 40,034 / 4,197 pos | 2019Q4–2020Q3 | 20,588 / 1,707 pos |
| 2021 | 2017Q1–2019Q4 | 66,788 / 7,009 pos | 2020Q4–2021Q3 | 20,037 / 1,411 pos |
| 2022 | 2017Q1–2020Q3 | 82,149 / 8,206 pos | 2021Q4–2022Q3 | 19,348 / 1,897 pos |
| 2023 | 2017Q1–2021Q3 | 102,186 / 9,617 pos | 2022Q4–2023Q3 | 18,846 / 1,800 pos |
| 2024 | 2017Q1–2022Q3 | 121,534 / 11,514 pos | 2023Q4–2024Q3 | 18,430 / 1,832 pos |
| 2025 | 2017Q1–2023Q4 | 145,021 / 13,769 pos | 2024Q4–2025Q1 | 9,062 / 984 pos |

**9,631** test positives in total. The training set is anchored at 2017Q1, so later
folds train on more. The 2025 fold has only two test quarters — the rest have
incomplete label windows and were dropped in §2.

The label actually closes **320 days** after the prediction time (four quarters run
from quarter end, and the prediction time is 45 days later), so a 366-day embargo is
conservative.

**The feature list is fixed.** Ranked once by XGBoost gain on the first fold's training
set, take the top N, and every fold — plus production — uses that same list. It is
derived only from data preceding any test period, so the six folds evaluate the model
that ships rather than a "re-select features every time" procedure; the collector can
therefore fix 50 raw fields permanently: the 49 MDRM items among the 50 features, plus
`RC_2170` total assets, which serves both as the denominator and as the source of the
derived `log_assets`.

---

## 5. Results

### 5.1 All candidate models

| Model | PR-AUC | ROC-AUC |
|---|---|---|
| xgb100 | **0.2940** | 0.7581 |
| xgb50 | 0.2918 | 0.7560 |
| xgb22 | 0.2707 | 0.7459 |
| gp50 | 0.2690 | **0.7678** |
| gp22 | 0.2538 | 0.7536 |
| xgb10 | 0.2236 | 0.7152 |
| gp12 | 0.2219 | 0.7253 |
| gp5+xgb10 | 0.2154 | 0.7167 |
| gp8 | 0.2097 | 0.7088 |
| xgb5 | 0.1923 | 0.6870 |
| gp5 | 0.1900 | 0.6894 |
| gp3 | 0.1548 | 0.6551 |

**Metric definitions**

| | PR-AUC | ROC-AUC |
|---|---|---|
| Full name | Area under the precision-recall curve | Area under the receiver operating characteristic curve |
| Plain reading | Take a list from the top down — what is the average precision | Probability that a random positive scores above a random negative |
| Random baseline | **equals the base rate** (0.0906 across the test folds) | **0.5** |
| Below the baseline means | the top of the list is worse than chance | **the ranking is inverted** (see the naive baseline at 0.42, §5.3) |
| Weighted toward | **the top of the list** — rank 100 counts far more than rank 5,000 | **the whole ranking** — all thresholds equally |
| Affected by base rate | **Yes.** 0.2701 (base rate 9.06%) and 0.0589 (2.56%) cannot be compared directly | No — comparable across periods and scopes |
| When to use it | building a watchlist of the top N banks | checking the model's direction, comparing across scopes |

"Lift over baseline" = PR-AUC ÷ base rate, i.e. how many times better than random.

**The models are distinguishable**: fold-to-fold σ = 0.037 against a between-model
spread of 0.142 — the spread is 3.8× the noise.

**More features help, plateauing around 50** — the signal is diffuse, many weak
features rather than a few dominant ones. XGBoost leads on PR-AUC while GP leads on
ROC-AUC; the two metrics pick different winners, which places the difference at the
top of the ranking rather than in the overall ordering.

The table above re-ranks features per fold. With the fixed list (§4), `gp50@fixed`
scores **PR-AUC 0.2701 / ROC-AUC 0.7664**, marginally ahead of the per-fold gp50 on
PR-AUC and marginally behind it on ROC-AUC. The fixed list is not chosen for being the
highest-scoring variant — it is the only one that can be deployed, since a quarterly
collector cannot pull a different 50 fields each time.

### 5.2 Per-fold performance

`gp50@fixed`:

| Fold | Test rows | Positives | Base rate | PR-AUC | ROC-AUC | Lift |
|---|---|---|---|---|---|---|
| 2020 | 20,588 | 1,707 | 8.29% | 0.2161 | 0.7506 | 2.61× |
| 2021 | 20,037 | 1,411 | 7.04% | 0.2467 | 0.7735 | 3.50× |
| 2022 | 19,348 | 1,897 | 9.80% | 0.2668 | 0.7427 | 2.72× |
| 2023 | 18,846 | 1,800 | 9.55% | 0.2871 | 0.7751 | 3.01× |
| 2024 | 18,430 | 1,832 | 9.94% | 0.3064 | 0.7820 | 3.08× |
| 2025 | 9,062 | 984 | 10.86% | 0.3053 | 0.7848 | 2.81× |
| **Weighted** | **106,311** | **9,631** | **9.06%** | **0.2701** | **0.7664** | **2.95×** |

`Lift = PR-AUC ÷ base rate`, used to compare folds whose base rates differ — the 2025
fold's PR-AUC is 24% above 2021's, but mostly because its base rate is higher; on lift,
2021 (3.50×) is the strongest. The weighted row's base rate is the six test sets
combined, not the panel's 9.61% (which includes training rows).

PR-AUC climbs from 0.216 to 0.306 across the first five folds, tracking the growth of
the training set, and holds at 0.305 in the sixth; ROC-AUC stays between 0.743 and
0.785 throughout. **No sign of decay over time.**

### 5.3 Against baselines

Four baselines, each answering a different question:

| Baseline | What it is | PR-AUC | ROC-AUC | gp50 ÷ it |
|---|---|---|---|---|
| **gp50@fixed** | the model currently selected | **0.2701** | **0.7664** | — |
| Logistic regression, 12 vars | linear model on the top 12 features | 0.1932 | 0.6909 | 1.40× |
| Best single feature | highest-AUC single field among the 529 | 0.1332 | 0.6356 | 2.03× |
| Random | random ordering | 0.0932 | 0.5028 | 2.90× |
| **Naive −tier1** | `risk = −tier-1 capital ratio` from `fact_call_report`, one line, no fitting | **0.0823** | **0.4208** | **3.28×** |

- **Random 0.0932 ≈ the test base rate 0.0906** — the evaluation pipeline is self-consistent
- **2.03× the best single feature** — combining features is worth something
- **1.40× a 12-variable logistic regression** — non-linearity is worth something
- **3.28× the naive rule, winning all six folds** — the machine learning earns its complexity

The naive `−tier1` baseline scores ROC-AUC 0.42 — below 0.5, i.e. inverted — and grows
**more** inverted among larger banks (0.31 on the top decile). Capital adequacy is not a
risk indicator for this target: distress events here are driven by deposit movement,
which relates only weakly to capital levels.

### 5.4 Operating table (2023 fold)

That fold covers 4,766 banks, **717 of which (15.0%) trigger a distress event within
the year**.

| List | Banks | Share | Caught | Recall | Precision |
|---|---|---|---|---|---|
| top 200 rows | 115 | 2.4% | 68 | 9.5% | **59.1%** |
| top 500 rows | 247 | 5.2% | 128 | 17.9% | 51.8% |
| top 1000 rows | 462 | 9.7% | 204 | 28.5% | 44.2% |
| top 2000 rows | 802 | 16.8% | 311 | 43.4% | 38.8% |

Watching 115 banks, roughly 59% do trigger within the year — **3.9×** the 15.0% base
rate.

**Recall is structurally low** — 717 banks will have an event and the list holds 115;
no amount of precision fits them in. Reaching 43.4% recall takes a list of 802. When
events are common and diffuse, high recall requires a large list.

### 5.5 Raw probabilities need calibration

`gp50@fixed` by decile on the 2023 fold:

| Decile | Mean prediction | Actual rate |
|---|---|---|
| lowest | 4.6% | 0.9% |
| fifth | 22.4% | 5.0% |
| ninth | 51.4% | 19.5% |
| highest | 68.0% | 31.7% |

A systematic overstatement, 2.1× at the top of the ranking and 5.1× at the bottom. **Raw `distress_prob` must not be displayed**; it has
to pass through `score.py`'s Platt calibration and quantile anchors. Ranking is
unaffected.

---

## 6. Model selection

**Currently selected: `gp50@fixed`.**

### 6.1 Why GP

**One — the dashboard needs intervals.** The four interval columns on
`bank_index_score` depend on the GP's `latent_mean` and `latent_var`, which XGBoost
does not produce; supplying them would mean bolting on a bootstrap ensemble or
Venn-Abers predictor.

**Two — the gap to XGBoost is small, and comes from data volume rather than the model.**
PR-AUC 0.2701 vs 0.2940 (8.1% behind), while ROC-AUC is higher for the GP. The GP is
capped at 5,000 rows per fold by its O(n³) cost (2,000 positive + 3,000 negative), where
XGBoost uses every training row — 145,021 by the last fold. Running XGBoost on the
**same 5,000-row subsample**:

| | PR-AUC | ROC-AUC | Training rows |
|---|---|---|---|
| gp50 | **0.2690** | **0.7678** | 5,000 |
| xgb100@sub | 0.2681 | 0.7610 | 5,000 |
| xgb50@sub | 0.2588 | 0.7519 | 5,000 |
| xgb100 (full) | 0.2940 | 0.7581 | 40,034–145,021 |

**At equal data there is no difference**, and the GP is marginally ahead. Feeding
XGBoost 8–29× more data raises PR-AUC (0.268 → 0.294) but slightly lowers ROC-AUC —
the extra data improves precision at the top, not the overall ordering.

### 6.2 What GP dimensions cost

`latent_var` — the source of the interval width — inflates with dimension:

| Dimensions | 3 | 5 | 8 | 12 | 22 | 50 |
|---|---|---|---|---|---|---|
| median latent_var | 0.031 | 0.059 | 0.108 | 0.166 | 0.351 | **0.525** |

50 dimensions carry **17×** the variance of 3, roughly 4× the interval width — the
distance-concentration effect of kernel methods, where every point looks "far from the
training data" in high dimensions. **Raising GP accuracy costs exactly the interval
quality that motivates using a GP**: the 80% interval averages 16.2 points against a
band width of 10, so **66.8% of banks have an interval spanning two bands or more**
(§7.1). Dropping to 12 dimensions preserves the intervals but takes PR-AUC to 0.222
(17.9% lower).

### 6.3 Note: how XGBoost performs

`xgb100` leads on full-sample PR-AUC by 8.1% (0.2940 vs 0.2701) and is stronger at the
top of the 2023 operating table (70.8% precision over the first 106 banks, against the
GP's 59.1%). **If the product scope widens to all 5,994 banks, or the interval columns
stop being required, the choice should be revisited.**

The three scopes disagree, which is worth recording:

| Scope | Better |
|---|---|
| Full-sample PR-AUC | XGBoost (+8.1%) |
| Full-sample ROC-AUC | **GP** (+1.1%) |
| Top of the full-sample operating table | XGBoost (+12 points of precision) |

---

## 7. Deliverables

### 7.1 Output tables

`bank_index_score` and `bank_index_feature`, defined in
`db/migrations/012_index_tables.sql`. The latest quarter (2025Q1) has scores for 4,519
banks, including all 104 seed banks.

| Band | Latest quarter | Panel rows | Actual event rate |
|---|---|---|---|
| sound (≥90) | 3,083 | 110,561 | **3.9%** |
| neutral (80–90) | 919 | 38,587 | 14.6% |
| distress (≤80) | 517 | 18,724 | **33.3%** |

Band separation is **8.6×** from sound to distress, monotone. Scores span 50.0 to 100.0.

**Intervals cannot indicate band membership.** The 80% interval averages **16.2
points** and the 95% interval 23.3, while the neutral band spans only 10:

| Bands spanned by the 80% interval | Share |
|---|---|
| 1 | 33.2% |
| 2 | 35.8% |
| 3 (all) | 31.0% |

**66.8% of banks have an 80% interval spanning two bands or more** — a bank labelled
neutral routinely has an interval covering both distress and sound.

**Among the 104 banks the dashboard displays, the effect is larger.** Their 2025Q1 band
distribution is healthier than the full sample (sound 81.7% vs 68.2%, distress 3.8% vs
11.4%, median score 97.5 vs 94.9), but the interval averages **17.9 points** against
14.9 overall:

| IDRSSD | Score | Band | 80% interval |
|---|---|---|---|
| 963945 | 72.0 | distress | **50.0 – 97.2** |
| 962966 | 77.2 | distress | **50.0 – 99.5** |
| 1394676 | 77.5 | distress | 51.5 – 97.3 |

The three lowest-scoring banks have intervals **spanning all three bands**. The cause is
that `latent_var` measures distance from the training data, and these 104 banks are the
largest filers — barely represented in a 5,000-row sample dominated by small and
mid-sized banks.

**But the width itself carries meaning, and its direction is positive.** Splitting the
2023 fold's out-of-fold predictions into quartiles by `latent_var`:

| Width quartile | Rows | Positives | Actual event rate | ROC-AUC |
|---|---|---|---|---|
| narrowest 25% | 4,712 | 318 | **6.75%** | 0.7352 |
| narrow | 4,711 | 386 | 8.19% | 0.7683 |
| wide | 4,711 | 446 | 9.47% | 0.7648 |
| **widest 25%** | 4,712 | 650 | **13.79%** | **0.7869** |

**The widest quartile has twice the event rate of the narrowest, and the model
ranks it at least as accurately.** The Spearman correlation between `latent_var` and
absolute prediction error is +0.09 — essentially none — so "wide interval = the model is unsure"
does not hold. The accurate reading is: **a wide interval means this bank differs from
most of the training sample, which means higher risk.** It is not junk data; it simply
is not a band-membership confidence.

#### Two columns this model cannot fill (dashboard · Rita)

`threshold_text` and `status` on `bank_index_feature`. A Gaussian Process produces no
per-feature thresholds — its evidence is overall similarity to historically distressed
banks, so there is no "tier-1 must be ≥ 9.0" style rule. Leave both NULL and skip them
when rendering, consistent with 012's own comment (*consumers must treat any feature row
as optional and render only what is present*). **No schema change is needed; every
other column fills and displays as usual.**

**49 of the 50 features are raw MDRM item codes** (`RCL_3814`, `RCRII_S442`, …; the
fiftieth is the derived `log_assets`) with no
readable names. Expanding `bank_index_feature` per feature gives 50 rows per bank, which
is not suitable for end users as-is.

### 7.2 Scores for the backtest

Produced against the input contract in `evals/backtest_protocol.md`:

```
scores_gp50_fixed_v1.csv    65,686 rows / 4,925 banks / 2021-12-31 to 2025-03-31
  fdic_cert_number · quarter_end_date · risk_score · model_version
```

Training = prediction time ≤ 2021-12-31 (the protocol's `--split-date`; report quarters
2017Q1–2021Q3, 102,186 rows / 9,617 positives); scoring = everything after. `risk_score`
is the raw GP probability (higher = riskier), uncalibrated — the protocol's metrics are
all rank-based, so calibration would not change them and only adds a step where the two
sides could diverge.

**These scores predate the fixed list in this report and cannot be regenerated.** The
list they were produced with was never written to disk (§9), so the delivered file is
kept as an artefact of the run that produced it rather than as something the pipeline
reproduces. Nothing downstream depends on regenerating it: the backtest has been run,
and any future run should use `fixed_order.json`, which the pipeline now emits.

**Verified to run against `evals/backtest.py`**:

| model_version | n_test | n_pos | PR-AUC | precision@50 | recall@budget=10 |
|---|---|---|---|---|---|
| `gp50_fixed_v1` | 1,251 | **32** | 0.0589 | 0.0800 | 0.2500 |

`models.py` also writes `oos_predictions.parquet` (106,311 rows / 6 folds) — the
per-row out-of-fold predictions behind every number in §5, and the only leak-free
per-row historical scores this project has (the production model is in-sample for every
past quarter). Available if a different split is wanted; not delivered by default.

This report uses the full 2017–2025 window; restricting a formal evaluation to the
window where the sentiment axis has coverage will change the numbers.

### 7.3 Before running the backtest: this scope cannot measure the model

**Same model, same window (report quarters 2022Q1–2024Q4), same pooled metric — only
the bank scope changes:**

| Bank scope | Labels | Test rows | Positives | Base rate | PR-AUC | Lift | ROC-AUC | Naive −tier1 |
|---|---|---|---|---|---|---|---|---|
| All filers | this report | 56,280 | 5,592 | 9.94% | 0.2910 | **2.93×** | 0.7697 | 0.0881 |
| Top 50% | this report | 28,149 | 2,079 | 7.39% | 0.2899 | **3.93×** | 0.7864 | 0.0657 |
| Top 10% | this report | 5,635 | 358 | 6.35% | 0.3011 | **4.74×** | 0.8028 | 0.0444 |
| 104 seed banks | this report | 1,248 | **29** | 2.32% | 0.0755 | **3.25×** | 0.7041 | 0.1327 |

**The model works on the full sample, the top 50% and the top 10%, and gets better as
banks get larger (2.93× → 4.74×).** On the 104 seed banks it reads 3.25× — but that
number carries almost no information, for two reasons.

**First, it is unstable.** The scores delivered for backtesting came from an earlier
fixed feature list; everything else — panel, folds, method, labels — was identical.
Swapping that list for the current one moves the pooled PR-AUC like this:

| Scope | Delivered list | Current list | Change |
|---|---|---|---|
| All 5,994 | 0.2979 | 0.2910 | **−2.3%** |
| 104 seed banks | 0.0459 | 0.0755 | **+64%** |

A change that barely registers on 5,592 positives moves the 104-bank reading by
two-thirds. With 29 events the metric is measuring which handful of banks happened to
land near the top, not whether the model ranks risk.

**Second, a rule with no predictive power beats it on this scope.** `−tier1` scores
ROC-AUC 0.42 on the full sample — inverted, worse than random — yet on these 104 banks
it reaches PR-AUC 0.1327 against the model's 0.0755. Any ranking that puts a known-bad
rule first is not ranking anything.

**Their event rate is a third of comparably-sized peers:**

| Size tier | All | of which seed banks | Non-seed at the same size |
|---|---|---|---|
| Top 2% | 109 banks · 4.16% | 70 banks · **2.17%** | 39 banks · 8.65% |
| Top 5% | 263 banks · 5.25% | 104 banks · **2.33%** | 159 banks · 7.56% |
| Top 10% | 530 banks · 6.35% | 104 banks · **2.33%** | 426 banks · 7.50% |

The seed banks sit at a median asset percentile of 98.4. Even against non-seed banks of
the same size they have two-thirds fewer distress events — this set was chosen for
prominence and news coverage, which selects for stability.

**What `evals/backtest.py` reported** on the delivered scores, against
`distress_bank_quarter.csv`: PR-AUC 0.0589 on 1,251 rows / 32 positives. The
three-label difference from the row above traces to the deposit basis in §8.3. That
result stands as run; it is not reproducible from this repo, because the feature list
behind those scores was never persisted (§9).

**Conclusion: a backtest on this scope cannot judge the model.** Not because the result
came back low — with the current list it comes back higher — but because 29 events
cannot separate a good ranking from a lucky one, in either direction.

**Recommendation**: to judge this axis's predictive ability, evaluate on the full sample
or by size tier (the first three rows). Treat the 104-bank result as a product-surface
reference, not as evidence about the model.

---

## 8. Boundaries and observations

### 8.1 What the model can and cannot do

**Label-adjacent fields contribute 9–19%.** Re-ranking after dropping all 144 of them:

| | Full pool, 529 | Clean pool, 385 | Drop |
|---|---|---|---|
| gp50@fixed | 0.2701 | 0.2187 | −19.0% |
| xgb100 | 0.2940 | 0.2514 | −14.5% |
| gp12 | 0.2219 | 0.2015 | −9.2% |

What remains is PR-AUC 0.219 / ROC-AUC 0.719, still 2.35× the random baseline. gp12
drops only 9.2% against gp50's 19%, placing the label-adjacent fields mostly in ranks
12–50.

Keeping them is defensible — the current NPL level genuinely affects whether NPL will
rise 1.5×, a real economic relationship. But it bounds the claim: **the model ranks
banks by distress risk; it has not found a signal beyond the conventional indicators.**

**Other boundaries:**

1. **Recall is capped by event frequency.** At a 9.06% base rate, no ranking covers most
   positives. A product needing high recall must revise the alert budget, not the model.

2. **The GP uses a fraction of the training data, and the feature list was chosen on the
   smallest window.** 5,000 sampled rows per fold means the 2025 fold discards 85% of
   positives and 97.7% of negatives; the fixed list comes from seven quarters
   (2017Q1–2018Q3). The latter is the price of avoiding leakage; the former could be
   improved by ensembling several subsamples, which is not implemented here.
   `length_scale = sqrt(dim) × 1.5` is a heuristic tuned at 3–12 dimensions and never
   retuned at 50.

3. **Annual folding wastes the last three quarters of usable training data.** `cut` is
   computed once from the fold's earliest prediction time, so later test rows in the same
   fold get a staler model. Quarterly rolling would match production more closely, and
   the difference also determines how often production should refit.

4. **The 45-day offset.** The label's four quarters run from report date while the data
   arrives 45 days later, so a nominal one-year warning is effectively 320 days.

5. **The final model has no test set, and no crisis-period data.** The production model
   fits on every label-closed row and is therefore in-sample for all history; the only
   clean check is timestamped scoring revisited a year later. The 2017–2025 window covers
   COVID and the 2023 rate shock but no systemic credit crisis.

### 8.2 Three observations about the target itself

Stated as facts for the team to judge whether the thresholds need revising; not an
assessment of the definition.

**(a) The −10% threshold sits in the tail, not loose.** The 1st percentile of
quarter-over-quarter deposit change is −10.54%; 1.17% of bank-quarters fire.

**(b) No clear seasonal or size pattern.** Event rates run Q1 1.31% / Q2 1.33% /
Q3 0.93% / Q4 1.10%; by asset tier 1.40% / 0.85% / 1.33% / 0.68%, not monotone.

**(c) Deposit events barely relate to actual failure.**

| | |
|---|---|
| Banks that ever fired a deposit event | 1,017 |
| Of those, firing ≥3 times | **204 (20%)**, one bank 14 times |
| Deposit events, 2017+ | 2,070 |
| Falling within a year before a bank's failure | **6 = 0.29%** |
| Of 27 banks that failed, those firing a deposit event in the prior year | **5 = 19%** |

Firing 14 times in 37 quarters reads more as volatile deposits than as repeated distress.

### 8.3 Three reporting-basis issues in `fact_call_report`

Found while reconciling the FFIEC source archives against that table during modelling.

| # | Table / column | Current logic | Problem | Suggested |
|---|---|---|---|---|
| 1 | `total_deposits` | `first_value(RCON2200, RCFN2200, RCFD2200)` | Domestic and foreign offices are complements; taking one drops the other. Banks with foreign operations are understated by **up to 2.83×** (one bank holds 65% of deposits abroad) | `RCON2200 + RCFN2200` |
| 2 | `npl_ratio` | `RCON1407 + RCON1403` ahead of RCFD | Same issue; the consolidated basis should win | RCFD first, otherwise domestic + foreign |
| 3 | RSSD 694904 (Flagstar/NYCB) 2022Q1 | deposits $18.0B, NPL 5.56% | **Contradicts the source filing**: $38.09B deposits, 0.134% NPL, $60.96B assets. This single record manufactures a distress event | Trace the collection / transform error for that row |

**None of the three affects this axis.** The `index/fundamentals/` pipeline does not read the
database — all four scripts pull from FFIEC source archives and coalesce prefixes
themselves (§2). The table is used only indirectly, twice: the `rssd_id ↔ fdic_cert`
mapping when generating the delivery CSV, and `tier1_capital_ratio` for the naive
baseline in §5.3 (78.1% coverage; missing values filled with the fold median).

Listed here because a field-by-field reconciliation surfaced them, for whoever maintains
that table.

---

## 9. Code

| Script | What it does | Runtime |
|---|---|---|
| `extract.py` | Downloads quarterly FFIEC archives, parses 17 schedules, coalesces prefixes by MDRM item | ~60 min |
| `features.py` | Builds the labelled panel, five screens | ~15 min |
| `mdrm_names.py` | Extracts MDRM field descriptions from the archives | ~2 min |
| `models.py` | 6-fold walk-forward, 5 XGBoost tiers × 6 GP dimensions + the fixed list | ~30 min |
| `final_model.py` | Fits on label-closed rows, calibrates, scores | ~5 min |

```bash
python3 index/fundamentals/extract.py       # FFIEC_START=20170101 fetches only the 37 quarters needed
python3 index/fundamentals/features.py
python3 index/fundamentals/models.py        # DROP_SAMESOURCE=1 runs the label-adjacent-fields control
MODEL_DIM=50 python3 index/fundamentals/final_model.py
```

`mdrm_names.json` is committed, so `mdrm_names.py` only needs running when the field set
changes. Everything else in the table above is reproducible from source: same archives,
same seed, same numbers.

**Two inputs used to be built outside the pipeline, and both were lost.** They were
written to `/tmp` by throwaway scripts, and nothing in the repository generated them:

| | Used by | Effect when absent |
|---|---|---|
| `mdrm_names.json` | screen [4] | The description branch dropped nobody. The candidate pool came out 534 instead of 529, silently — no warning in any log |
| `fixed_order.json` | `final_model.py`, the `@fixed` variant | `final_model.py` could not run at all, and no code in this repository produced a fixed list |

Both are now part of the pipeline: `mdrm_names.json` is committed and `features.py`
exits with an error if it is missing, and `models.py` writes `fixed_order.json` from the
first fold's ranking. The `@fixed` variant is a model in the fold loop like any other.

The cost of the second gap is that the delivered backtest scores (§7.2) and the earlier
`gp50@fixed` figures cannot be rebuilt — the list behind them is unknown and was not the
first-fold ranking, since fold 2020's `@fixed` result must equal `gp50` by construction
and the earlier numbers differ (0.2280 against 0.2161). Every figure in this report is
from the reproducible list.
