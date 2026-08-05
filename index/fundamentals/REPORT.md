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
| [3] Missingness vs size: top 1% vs the rest, gap ≤10% | 866 (**24 caught only here**) |
| [4] Drop identifiers / dates / near-constants | −388 |
| [5] Correlation pruning \|ρ\| ≥ 0.95 (merged, not discarded) | 640 → 528 |
| Plus the derived column `log_assets` | **529** |

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
therefore fix 51 fields permanently (50 features plus total assets as denominator).

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
| Affected by base rate | **Yes.** 0.2755 (base rate 9.06%) and 0.0589 (2.56%) cannot be compared directly | No — comparable across periods and scopes |
| When to use it | building a watchlist of the top N banks | checking the model's direction, comparing across scopes |

"Lift over baseline" = PR-AUC ÷ base rate, i.e. how many times better than random.

**The models are distinguishable**: fold-to-fold σ = 0.037 against a between-model
spread of 0.142 — the spread is 3.8× the noise.

**More features help, plateauing around 50** — the signal is diffuse, many weak
features rather than a few dominant ones. XGBoost leads on PR-AUC while GP leads on
ROC-AUC; the two metrics pick different winners, which places the difference at the
top of the ranking rather than in the overall ordering.

The table above re-ranks features per fold. With the fixed list (§4), `gp50@fixed`
scores **PR-AUC 0.2755 / ROC-AUC 0.7701** — the highest ROC-AUC of any candidate.

### 5.2 Per-fold performance

`gp50@fixed`:

| Fold | Test rows | Positives | Base rate | PR-AUC | ROC-AUC | Lift |
|---|---|---|---|---|---|---|
| 2020 | 20,588 | 1,707 | 8.29% | 0.2280 | 0.7561 | 2.75× |
| 2021 | 20,037 | 1,411 | 7.04% | 0.2445 | 0.7821 | 3.47× |
| 2022 | 19,348 | 1,897 | 9.80% | 0.2720 | 0.7551 | 2.77× |
| 2023 | 18,846 | 1,800 | 9.55% | 0.2997 | 0.7798 | 3.14× |
| 2024 | 18,430 | 1,832 | 9.94% | 0.3078 | 0.7755 | 3.10× |
| 2025 | 9,062 | 984 | 10.86% | 0.3048 | 0.7784 | 2.81× |
| **Weighted** | **106,311** | **9,631** | **9.06%** | **0.2755** | **0.7701** | **3.00×** |

`Lift = PR-AUC ÷ base rate`, used to compare folds whose base rates differ — the 2025
fold's PR-AUC is 25% above 2021's, but mostly because its base rate is higher; on lift,
2021 (3.47×) is the strongest. The weighted row's base rate is the six test sets
combined, not the panel's 9.61% (which includes training rows).

PR-AUC climbs monotonically from 0.228 to 0.308, tracking the growth of the training
set; ROC-AUC holds between 0.755 and 0.782. **No sign of decay over time.**

### 5.3 Against baselines

Four baselines, each answering a different question:

| Baseline | What it is | PR-AUC | ROC-AUC | gp50 ÷ it |
|---|---|---|---|---|
| **gp50@fixed** | the model currently selected | **0.2755** | **0.7701** | — |
| Logistic regression, 12 vars | linear model on the top 12 features | 0.1932 | 0.6909 | 1.43× |
| Best single feature | highest-AUC single field among the 529 | 0.1332 | 0.6356 | 2.07× |
| Random | random ordering | 0.0932 | 0.5028 | 2.96× |
| **Naive −tier1** | `risk = −tier-1 capital ratio`, one line, no fitting | **0.0823** | **0.4208** | **3.35×** |

- **Random 0.0932 ≈ the test base rate 0.0906** — the evaluation pipeline is self-consistent
- **2.07× the best single feature** — combining features is worth something
- **1.43× a 12-variable logistic regression** — non-linearity is worth something
- **3.35× the naive rule, winning all six folds** — the machine learning earns its complexity

The naive `−tier1` baseline scores ROC-AUC 0.42 — below 0.5, i.e. inverted — and grows
**more** inverted among larger banks (0.31 on the top decile). Capital adequacy is not a
risk indicator for this target: distress events here are driven by deposit movement,
which relates only weakly to capital levels.

### 5.4 Operating table (2023 fold)

That fold covers 4,766 banks, **717 of which (15.0%) trigger a distress event within
the year**.

| List | Banks | Share | Caught | Recall | Precision |
|---|---|---|---|---|---|
| top 200 rows | 106 | 2.2% | 67 | 9.3% | **63.2%** |
| top 500 rows | 259 | 5.4% | 136 | 19.0% | 52.5% |
| top 1000 rows | 459 | 9.6% | 200 | 27.9% | 43.6% |
| top 2000 rows | 811 | 17.0% | 312 | 43.5% | 38.5% |

Watching 106 banks, roughly 63% do trigger within the year — **4.2×** the 15.0% base
rate.

**Recall is structurally low** — 717 banks will have an event and the list holds 106;
no amount of precision fits them in. Reaching 43.5% recall takes a list of 811. When
events are common and diffuse, high recall requires a large list.

### 5.5 Raw probabilities need calibration

`gp50@fixed` by decile on the 2023 fold:

| Decile | Mean prediction | Actual rate |
|---|---|---|
| lowest | 4.5% | 1.0% |
| median | 26.1% | 5.8% |
| ninth | 50.9% | 19.2% |
| highest | 68.1% | 32.6% |

A systematic 2–4× overstatement. **Raw `distress_prob` must not be displayed**; it has
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
PR-AUC 0.2755 vs 0.2940 (6.7% behind), while ROC-AUC is higher for the GP. The GP is
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
quality that motivates using a GP**: the 80% interval averages 16.3 points against a
band width of 10, so **67.2% of banks have an interval spanning two bands or more**
(§7.1). Dropping to 12 dimensions preserves the intervals but takes PR-AUC to 0.222
(19% lower).

### 6.3 Note: how XGBoost performs

`xgb100` leads on full-sample PR-AUC by 6.7% (0.2940 vs 0.2755) and is stronger at the
top of the 2023 operating table (75.2% precision over the first 101 banks, against the
GP's 63.2%). **If the product scope widens to all 5,994 banks, or the interval columns
stop being required, the choice should be revisited.**

The three scopes disagree, which is worth recording:

| Scope | Better |
|---|---|
| Full-sample PR-AUC | XGBoost (+6.7%) |
| Full-sample ROC-AUC | **GP** (+1.6%) |
| Top of the full-sample operating table | XGBoost (+12 points of precision) |

---

## 7. Deliverables

### 7.1 Output tables

`bank_index_score` and `bank_index_feature`, defined in
`db/migrations/012_index_tables.sql`. The latest quarter (2025Q1) has scores for 4,519
banks, including all 104 seed banks.

| Band | Latest quarter | Panel rows | Actual event rate |
|---|---|---|---|
| sound (≥90) | 3,110 | 108,949 | **3.8%** |
| neutral (80–90) | 888 | 38,988 | 13.8% |
| distress (≤80) | 521 | 19,935 | **32.9%** |

Band separation is **8.6×** from sound to distress, monotone. Scores span 50.3 to 100.0.

**Intervals cannot indicate band membership.** The 80% interval averages **16.3
points** and the 95% interval 23.3, while the neutral band spans only 10:

| Bands spanned by the 80% interval | Share |
|---|---|
| 1 | 32.8% |
| 2 | 35.5% |
| 3 (all) | 31.7% |

**67.2% of banks have an 80% interval spanning two bands or more** — a bank labelled
neutral routinely has an interval covering both distress and sound.

**Among the 104 banks the dashboard displays, the effect is larger.** Their 2025Q1 band
distribution is healthier than the full sample (sound 76.9% vs 68.8%, distress 2.9% vs
11.5%, median score 96.8 vs 95.0), but the interval averages **21.0 points** against
14.9 overall:

| IDRSSD | Score | Band | 80% interval |
|---|---|---|---|
| 962966 | 67.8 | distress | **50.0 – 98.6** |
| 3394278 | 72.6 | distress | 50.6 – 96.7 |
| 963945 | 74.4 | distress | 51.6 – 96.6 |

The three lowest-scoring banks have intervals **spanning all three bands**. The cause is
that `latent_var` measures distance from the training data, and these 104 banks are the
largest filers — barely represented in a 5,000-row sample dominated by small and
mid-sized banks.

**But the width itself carries meaning, and its direction is positive.** Splitting the
2023 fold's out-of-fold predictions into quartiles by `latent_var`:

| Width quartile | Rows | Positives | Actual event rate | ROC-AUC |
|---|---|---|---|---|
| narrowest 25% | 4,712 | 328 | **6.96%** | 0.7265 |
| narrow | 4,711 | 390 | 8.28% | 0.7570 |
| wide | 4,711 | 436 | 9.25% | 0.7842 |
| **widest 25%** | 4,712 | 646 | **13.71%** | **0.8003** |

**The widest quartile has nearly twice the event rate of the narrowest, and the model
ranks it more accurately.** The Spearman correlation between `latent_var` and absolute
prediction error is +0.07 — essentially none — so "wide interval = the model is unsure"
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

**All 50 features are raw MDRM item codes** (`RCL_3814`, `RCRII_S442`, …) with no
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

**Same model, same window (test 2022–2024), same pooled metric — only the bank scope
changes:**

| Bank scope | Labels | Test rows | Positives | Base rate | PR-AUC | Lift | ROC-AUC | Naive −tier1 |
|---|---|---|---|---|---|---|---|---|
| All 5,994 | this report | 56,280 | 5,592 | 9.94% | 0.2979 | **3.00×** | 0.7701 | 0.0881 |
| Top 50% | this report | 28,137 | 2,075 | 7.37% | 0.2868 | **3.89×** | 0.7815 | 0.0648 |
| Top 10% | this report | 5,633 | 358 | 6.36% | 0.3052 | **4.80×** | 0.8072 | 0.0431 |
| 104 seed banks | this report | 1,251 | **29** | 2.32% | 0.0459 | **1.98×** | 0.7184 | 0.1327 |
| 104 seed banks | `distress_bank_quarter.csv` | 1,251 | **32** | 2.56% | 0.0589 | 2.30× | — | 0.1342 |

The last row is what `evals/backtest.py` actually reports. The three-label difference
between the last two rows traces to the deposit basis in §8.3.

**The model works on the full sample, the top 50%, and the top 10% — and gets better
as banks get larger (3.00× → 4.80×). It falls to 1.98× only on these 104 banks**, which
are themselves inside the top 10%.

**Their event rate is a third of comparably-sized peers:**

| Size tier | All | of which seed banks | Non-seed at the same size |
|---|---|---|---|
| Top 2% | 109 banks · 4.16% | 71 banks · **2.16%** | 38 banks · 8.72% |
| Top 5% | 263 banks · 5.25% | 105 banks · **2.32%** | 158 banks · 7.57% |
| Top 10% | 530 banks · 6.35% | 105 banks · **2.32%** | 425 banks · 7.50% |

The seed banks sit at a median asset percentile of 98.4. Even against non-seed banks of
the same size they have two-thirds fewer distress events — this set was chosen for
prominence and news coverage, which selects for stability.

**Conclusion: a backtest on this scope cannot judge the model.** With 29–32 events, and
a population whose distress pattern differs from the bulk of the training sample, the
number is uninformative in either direction — within the same window a rule with no
predictive power (`−tier1`, ROC-AUC 0.42 on the full sample) scores *higher* (0.1342 vs
0.0589), which is itself evidence that rankings on this sample cannot be trusted.

**Recommendation**: to judge this axis's predictive ability, evaluate on the full sample
or by size tier (the first three rows). Treat the 104-bank result as a product-surface
reference, not as evidence about the model.

---

## 8. Boundaries and observations

### 8.1 What the model can and cannot do

**Label-adjacent fields contribute 15–20%.** Re-ranking after dropping all 144 of them:

| | Full pool, 529 | Clean pool, 385 | Drop |
|---|---|---|---|
| gp50@fixed | 0.2755 | 0.2093 | −24.0% |
| xgb100 | 0.2940 | 0.2514 | −14.5% |
| gp12 | 0.2219 | 0.2015 | −9.2% |

What remains is PR-AUC 0.209 / ROC-AUC 0.718, still 2.2× the random baseline. `RC_2200`
(total deposits) alone is worth about 4 points. gp12 drops only 9.2% against gp50's 24%,
placing the label-adjacent fields mostly in ranks 12–50.

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
| `models.py` | 6-fold walk-forward, 5 XGBoost tiers × 6 GP dimensions + an ensemble | ~25 min |
| `final_model.py` | Fits on label-closed rows, calibrates, scores | ~5 min |

```bash
python3 index/fundamentals/extract.py       # FFIEC_START=20170101 fetches only the 37 quarters needed
python3 index/fundamentals/features.py
python3 index/fundamentals/models.py        # DROP_SAMESOURCE=1 runs the label-adjacent-fields control
MODEL_DIM=50 python3 index/fundamentals/final_model.py
```
