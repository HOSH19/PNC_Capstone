# Unified Schema Dataset

This folder implements the unified local dataset for Shu Han's three core sources:

- `FDIC BankFind Suite API`
- `FFIEC Central Data Repository Call Reports`
- `FDIC Failed Bank List / Failures endpoint`

The tables are populated with **real source data** by `scripts/build_dataset.py`:

- `dim_bank.csv` and `fact_distress_event.csv` come from the FDIC API
- `fact_call_report.csv` now comes from the **real FFIEC Call Report bulk source**
- `fact_bank_quarter.csv` is derived locally from the two fact tables

The script can use a historical quarterly window from **2008 Q1 through 2026
Q1** so the modeling table can include bank-quarters that precede real failure
events. The files currently in `tables/` reflect the most recent build that was
run locally.

## Current Contents (real data)

| Table | Rows | Source |
|-------|------|--------|
| `dim_bank.csv` | 5,247 | FDIC BankFind institutions, with failure-backed bank coverage retained |
| `fact_call_report.csv` | 435,264 | FFIEC CDR bulk Call Reports, 73 quarters from 2008 Q1 to 2026 Q1 |
| `fact_distress_event.csv` | 4,115 | FDIC failures endpoint |
| `fact_bank_quarter.csv` | 435,264 | derived join |

### Important notes on the real data

- **Units:** all dollar amounts (`total_assets`, `total_deposits`, `cre_loans`)
  are in **thousands of USD**, matching the FFIEC/FDIC source files. Ratios
  (`tier1_capital_ratio`, `total_capital_ratio`, `npl_ratio`,
  `loan_loss_allowance_ratio`, `liquidity_ratio`) are percentages.
- **`fact_call_report.csv` is now FFIEC-based.** It is built from the real
  `Call Reports -- Single Period` bulk download files rather than the FDIC
  `/financials` endpoint.
- **`cre_loans`** is currently a CRE proxy built from FFIEC `RCCI` categories:
  construction/land development, 1-4 family construction, owner-occupied
  nonfarm nonresidential, and other nonfarm nonresidential loans.
- **`liquidity_ratio`** is currently a simple FFIEC-derived proxy:
  `(cash due from depository institutions + currency/coin + AFS securities) /
  total_assets * 100`.
- **`npl_ratio`** is currently derived as:
  `(90+ days past due accruing + nonaccrual loans) / total loans * 100`.
- **`loan_loss_allowance_ratio`** is currently derived as:
  `allowance for credit losses / total loans * 100`.
- **`securities_unrealized_loss`** is still left empty. Even with the FFIEC
  bulk files, that field needs more precise schedule-level handling than the
  current build uses.
- **The current historical build produces positive labels** in
  `fact_bank_quarter.csv`: 1,957 rows with `distress_within_4q = 1`, 3,562 rows
  with `distress_within_8q = 1`, and 6,594 rows with a non-null
  `days_to_distress`.
- **By default, the script does not replace the FDIC CSVs you already have.**
  A normal run rebuilds only `fact_call_report.csv` from FFIEC and then
  regenerates `fact_bank_quarter.csv`.

## Rebuilding

```bash
python3 scripts/build_dataset.py
```

Full historical rebuild:

```bash
python3 scripts/build_dataset.py --start-year 2008 --end-quarter 20260331
```

Optional FDIC refreshes:

```bash
python3 scripts/build_dataset.py --refresh-dim-bank
python3 scripts/build_dataset.py --refresh-distress
```

The script uses the `ffiec-data-collector` Python package for the FFIEC bulk
download workflow.

The dataset uses a bank-centric star schema:

- `tables/dim_bank.csv`: master bank dimension
- `tables/fact_call_report.csv`: quarterly fundamentals
- `tables/fact_distress_event.csv`: distress event labels
- `tables/fact_bank_quarter.csv`: derived modeling table
- `sql/schema.sql`: SQLite/Postgres-style DDL for the same structure

## Source Coverage Check

The current design covers all three sources assigned to Shu Han:

1. `FDIC BankFind Suite API`
   - represented by `tables/dim_bank.csv`
2. `FFIEC Central Data Repository Call Reports`
   - represented by `tables/fact_call_report.csv`
3. `FDIC Failed Bank List / Failures endpoint`
   - represented by `tables/fact_distress_event.csv`

`tables/fact_bank_quarter.csv` is not a fourth raw source. It is a derived
modeling table created by joining the FFIEC quarterly fundamentals to the FDIC
distress labels on `fdic_cert_number`.

## Table Design

### `dim_bank`
One row per bank institution.

Primary key:
- `fdic_cert_number`

### `fact_call_report`
One row per bank per quarter.

Primary key:
- `fdic_cert_number`, `report_date`

### `fact_distress_event`
One row per failure or assistance event.

Primary key:
- `fdic_cert_number`, `failure_date`

### `fact_bank_quarter`
One row per bank per quarter for modeling. This is derived by joining quarterly fundamentals to forward-looking distress labels.

Primary key:
- `fdic_cert_number`, `quarter_end_date`

## Recommended Population Order

1. Keep the existing FDIC-backed `dim_bank.csv` unless you explicitly refresh it
2. Keep the existing FDIC-backed `fact_distress_event.csv` unless you explicitly refresh it
3. Pull historical FFIEC Call Report bulk files
4. Build `fact_call_report.csv` from FFIEC schedules
5. Build `fact_bank_quarter.csv` by labeling each bank-quarter with future distress windows

## Notes

- Use `fdic_cert_number` as the main join key across all three assigned sources.
- Keep `rssd_id` where available to support joins to additional banking datasets later.
- `fact_bank_quarter.csv` should usually be generated, not manually entered.
- The CSVs are already populated with real data after running `scripts/build_dataset.py`.
