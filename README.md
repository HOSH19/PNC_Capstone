# Sentiment Analysis on Call Reports and Financial News for Active U.S. Banks

## Overview
This project builds a sentiment-driven early warning framework for active U.S. banks. It combines quarterly FFIEC Call Report fundamentals with public text signals such as financial news, SEC filings, and earnings commentary to classify bank financial health and highlight emerging risk narratives.

## Problem Statement
Quarterly filings update slowly, while public sentiment can shift quickly. The core question is whether text-based signals can improve early detection of bank distress beyond financial ratios alone.

## System Overview
```mermaid
flowchart TD
    A[Structured Data<br/>Call Reports, BankFind, Failures] --> D[Unified Dataset]
    B[Text Data<br/>News, SEC Filings, Earnings Commentary] --> C[Sentiment and Risk Features]
    D --> E[Modeling]
    C --> E
    E --> F[Financial Health Classification]
    E --> G[Dashboard and Alerts]
```

## Unified Data Architecture
The project uses a bank-centric schema with a shared institution key and separate fact tables for fundamentals, events, and derived modeling views.

```mermaid
erDiagram
    DIM_BANK ||--o{ FACT_CALL_REPORT : has
    DIM_BANK ||--o{ FACT_DISTRESS_EVENT : has
    DIM_BANK ||--o{ FACT_BANK_QUARTER : has

    DIM_BANK {
        int fdic_cert_number
        int rssd_id
        string bank_name
        string city
        string state
        boolean active_status
        float total_assets
    }

    FACT_CALL_REPORT {
        int fdic_cert_number
        int rssd_id
        date report_date
        float total_assets
        float total_deposits
        float tier1_capital_ratio
        float total_capital_ratio
        float npl_ratio
        float loan_loss_allowance_ratio
        float liquidity_ratio
        float securities_unrealized_loss
        float cre_loans
    }

    FACT_DISTRESS_EVENT {
        int fdic_cert_number
        date failure_date
        string bank_name
        string city
        string state
        string event_type
        int distress_label
    }

    FACT_BANK_QUARTER {
        int fdic_cert_number
        date quarter_end_date
        int distress_within_4q
        int distress_within_8q
        int days_to_distress
    }
```

## Financial Health Classification
- Stable: fundamentals and sentiment remain within normal ranges.
- Watch: sentiment deteriorates or risk themes emerge without strong fundamental confirmation.
- Elevated Risk: sentiment decline aligns with weakening fundamentals.
- Imminent Disruption: strong negative sentiment combines with critical fundamentals or confirmed distress events.

## Repository Structure
- `unified_schema_dataset/`: local dataset build pipeline, schema, and populated tables.
- `unified_schema_dataset/scripts/`: scripts for FDIC refreshes, FFIEC Call Report ingestion, and derived modeling-table generation.
- `unified_schema_dataset/sql/schema.sql`: relational schema definition.
- `unified_schema_dataset/tables/`: generated CSV tables.

## Repo layout & ownership

```
├── unified_ffiec_fdic_dataset/  # FFIEC/FDIC unified dataset build (Shu Han)
├── db/                          # SQL migrations + bank crosswalk seed (this slice)
│   ├── migrations/
│   └── seed/
├── pipeline/                    # incremental pollers (GDELT, EDGAR) + db client (this slice)
│   └── loaders/                 # full loaders: FRED, yfinance, fundamentals (Phase 2+)
├── scoring/                     # FinBERT + Gemini hybrid scorer (Phase 2)
├── index/                       # stability index library + recompute CLI (Phase 3)
│   └── config/                  # index parameters as versioned YAML
├── dashboard/                   # Streamlit app (Phase 5)
├── evals/                       # eval harness (Phase 6)
│   ├── items/                   # gold set CSVs
│   └── prompts/                 # bake-off entries, one file per person
└── .github/workflows/           # ingest.yml (this slice)
```

- `unified_ffiec_fdic_dataset/` — FFIEC/FDIC structured dataset; owner: Shu Han.
- `db/` — Postgres migrations and the bank crosswalk seed; adding a bank is one seed row, zero code changes.
- `pipeline/` — incremental pollers (watermark + overlap window) and the thin DB client; `loaders/` will hold stateless full loaders (owner: TBD).
- `scoring/` — hybrid text scorer consuming `raw_item` status columns (owner: TBD).
- `index/` — stability index computation; parameters in `index/config/` YAML (owner: TBD).
- `dashboard/` — read-only Streamlit views (owner: TBD).
- `evals/` — gold-set evaluation harness and prompt bake-off (owner: TBD).

The table schema in `db/migrations/` is the **only shared contract** between
modules — modules communicate through those tables, never by importing each
other's code.

## Data Sources
- FFIEC Central Data Repository Call Reports
- FDIC BankFind Suite API
- FDIC Failed Bank List / failures endpoint
- Financial news and SEC filing sources to be integrated in later phases
