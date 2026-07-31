# Distress event definition (v1)

Owner: Shu Han. Audience: Ming (training target) and anyone asking
"why is this bank-quarter labeled distressed?"

Evidence: [`eda/reports/2026-07-31_distress_threshold_eda.md`](../eda/reports/2026-07-31_distress_threshold_eda.md),
produced by [`eda/distress_threshold_eda.py`](../eda/distress_threshold_eda.py).

**Status:** thresholds, Ming column contract, and labeled CSV locked for v1.
Rebuild with `python3 evals/build_distress_labels.py` →
[`evals/items/distress_bank_quarter.csv`](items/distress_bank_quarter.csv).

---

## 1. What we are labeling

A **distress state**, not FDIC failure / closure.

Failure is unusable on the 104 seed banks (0 `distress_within_4q` positives
from failures in 2017+). The answer key is instead rule-based acute stress
from Call Report fundamentals.

Scope for v1:

| In | Out |
|---|---|
| 104 seed banks (`db/seed/banks.csv`) | Full 5k+ institution set |
| `report_date >= 2017-01-01` | Pre-2017 (no text-era overlap for the combined eval) |
| Fundamentals rules below | Regulator enforcement / OCC (deferred) |

---

## 2. Event rule (fires in quarter Q)

An **event quarter** is a seed-bank Call Report row where **either** leg is true:

1. **Deposit outflow:**  
   `(total_deposits_t − total_deposits_{t−1}) / total_deposits_{t−1} ≤ −0.10`  
   (requires a prior quarter for the same bank with `total_deposits > 0`)

2. **NPL spike:**  
   `npl_ratio_t / npl_ratio_{t−1} ≥ 1.5` **and** `npl_ratio_t > 2.0`  
   (requires prior `npl_ratio > 0`)

Units match [`unified_ffiec_fdic_dataset/`](../unified_ffiec_fdic_dataset/README.md):
ratios are percentages; dollar fields are thousands of USD.

### Empirically checked (2017+, seed set)

| Metric | Value |
|---|---|
| Event quarters | 33 |
| Distinct banks | 24 |
| Top-bank share | `sofi` 3/33 (9%) — under the 25% concentration flag |
| Events/year | 2017:4, 2018:4, 2019:1, 2020:8, 2021:4, 2022:4, 2023:5, 2024:3 |

### Explicitly not part of the v1 event OR

| Signal | Why excluded |
|---|---|
| Tier-1 capital &lt; 8% / 6% / 4% | Zero seed-bank breaches 2017+ (p1 ≈ 9.9%) |
| Liquidity low-tail | Tags structural business models more than acute stress; keep as a GP **feature** |
| Unrealized securities loss ≤ −5% of assets | 157 quarters / 18 banks — floods the label; keep as a GP **feature** |
| FDIC / Fed / OCC enforcement | OCC ingest landed (`occ_enforcement` → `raw_item`) but **not** used in the v1 event OR. Revisit before claiming enforcement-based distress. |

---

## 3. Lookahead label (what Ming trains on)

When an event fires in quarter **Q**:

| Column | Value |
|---|---|
| `is_event_quarter` | `1` on **Q** only |
| `distress_within_4q` | `1` on the **four prior** quarters Q−1 … Q−4 (not on Q itself) |

Rationale: the project is early warning. Ming’s GP should predict *upcoming*
distress from current fundamentals, not detect the event quarter contemporaneously.

Additional rules:

- If a quarter lies within 4 prior quarters of **multiple** events, it still gets
  `distress_within_4q = 1` once.
- An event quarter is **not** marked `distress_within_4q` for *its own* event.
  It **can** still be `distress_within_4q = 1` if a *later* event falls within
  four quarters after it (clustered events). Ming still trains on
  `distress_within_4q`; those rare overlaps are valid early-warning positives
  for the later event. Observed on this build: **4** such rows.
- Quarters with insufficient history to evaluate the event rule (no prior
  quarter) are never events; they may still receive `distress_within_4q = 1`
  if a later event falls within the horizon.
- Missing inputs for a leg → that leg is false (no silent `1`).

```text
Example (one bank):
  2022-03-31  is_event=0  within_4q=1   ← Q-4
  2022-06-30  is_event=0  within_4q=1   ← Q-3
  2022-09-30  is_event=0  within_4q=1   ← Q-2
  2022-12-31  is_event=0  within_4q=1   ← Q-1
  2023-03-31  is_event=1  within_4q=0   ← Q (event: deposits −12%)
  2023-06-30  is_event=0  within_4q=0
```

**Ming trains against `distress_within_4q`, not `is_event_quarter`.**

---

## 4. Ming column contract

Artifact (Unit 3 will emit): one row per seed-bank × quarter in the window.

| Column | Type | Meaning |
|---|---|---|
| `fdic_cert_number` | int | Join key ↔ `fact_call_report.fdic_cert_number` / `bank.fdic_cert` |
| `quarter_end_date` | date (`YYYY-MM-DD`) | Same as Call Report `report_date` (quarter-end) |
| `bank_id` | text | Convenience from `db/seed/banks.csv` (optional for joins) |
| `is_event_quarter` | int 0/1 | Rule fired this quarter |
| `distress_within_4q` | int 0/1 | **Training / backtest target** |
| `event_reason` | text, nullable | On event rows only: `deposit_outflow`, `npl_spike`, or `deposit_outflow\|npl_spike` |

**Primary key:** `(fdic_cert_number, quarter_end_date)`.

**Path:** [`evals/items/distress_bank_quarter.csv`](items/distress_bank_quarter.csv)  
(local CSV; a DB migration is *not* required for Ming to start. If Ming later
wants a table behind migration `012`, that is his lane.)

**Join reminder:** scoring joins via `bank.fdic_cert ↔ fdic_cert_number` (value
join, no FK). Same key here.

---

## 5. Out of scope here

- Eval metrics / `evals/backtest.py` — later Shu Han tasks
- GP features, bands, `012_index_tables.sql` — Ming
- Sentiment labels — unrelated; never mix with this table

---

## 6. Version

| Version | Date | Change |
|---|---|---|
| v1 | 2026-07-31 | Lock deposit ≤−10% OR NPL ≥1.5× &gt;2%; 4q lookahead; fundamentals only |
| v1.1 | 2026-07-31 | Emit `evals/items/distress_bank_quarter.csv`; note clustered-event overlap |
