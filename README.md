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
  `scripts/` builds it (FDIC refreshes, FFIEC Call Report ingestion, derived
  modeling table), `sql/schema.sql` is the schema definition, `tables/` holds
  the generated CSVs.
- `db/` — Postgres migrations and the bank crosswalk seed; adding a bank is one seed row, zero code changes.
- `pipeline/` — incremental pollers (watermark + overlap window) and the thin DB client; `loaders/` will hold stateless full loaders (owner: TBD).
- `scoring/` — hybrid text scorer consuming `raw_item` status columns (owner: TBD).
- `index/` — stability index computation; parameters in `index/config/` YAML (owner: TBD).
- `dashboard/` — read-only Streamlit views (owner: TBD).
- `evals/` — gold-set evaluation harness and prompt bake-off (owner: TBD).

The table schema in `db/migrations/` is the **only shared contract** between
modules — modules communicate through those tables, never by importing each
other's code.

## Ingestion architecture

Everything meets in one Supabase Postgres database. GitHub Actions runs the
jobs; modules never import each other — they communicate only through the
shared tables defined in `db/migrations/`. Solid arrows are live today,
dashed arrows are planned phases.

```mermaid
flowchart LR
    subgraph SRC_TEXT[Text signals]
        GDELT[GDELT DOC 2.0<br/>bank news]
        EDGAR[SEC EDGAR<br/>8-K / 10-Q / 10-K]
        RSS[RSS feeds<br/>planned]
    end

    subgraph SRC_STRUCT[Structured fundamentals - Shu Han]
        FFIEC[FFIEC CDR<br/>Call Reports]
        FDIC[FDIC BankFind]
        FAIL[FDIC failures]
    end

    subgraph SRC_MKT[Macro and market - planned]
        FRED[FRED]
        YF[yfinance]
    end

    subgraph ACTIONS[GitHub Actions - ingest workflow]
        SEED[seed job<br/>banks.csv → bank]
        PG[poll_gdelt]
        PE[poll_edgar]
        PRSS[poll_rss - planned]
        SEED --> PG
        SEED --> PE
        SEED -.-> PRSS
    end

    BUILD[build_dataset.py<br/>local build → tables/*.csv]
    LOADERS[full loaders - planned<br/>pipeline/loaders/]

    GDELT --> PG
    EDGAR --> PE
    RSS -.-> PRSS
    FFIEC --> BUILD
    FDIC --> BUILD
    FAIL --> BUILD
    BUILD -.->|CSV tables| LOADERS
    FRED -.-> LOADERS
    YF -.-> LOADERS

    subgraph DB[Supabase Postgres]
        BANKT[(bank crosswalk)]
        RAW[(raw_item)]
        OPS[(watermark<br/>pipeline_heartbeat)]
        FUND[(fundamentals tables<br/>planned)]
    end

    SEED -->|upsert| BANKT
    PG --> RAW
    PE --> RAW
    PG -->|state| OPS
    PE -->|state| OPS
    PRSS -.-> RAW
    LOADERS -.-> FUND

    RAW --> SCORE[scoring - Phase 2]
    SCORE --> IDX[stability index - Phase 3]
    FUND -.-> IDX
    IDX -.-> DASH[dashboard - Phase 5]
    FUND -.->|distress labels| EVAL[evals and backtest - Phase 6]
```

Five ideas explain the whole design:

**1. One landing table.** Every collected text item — news article or SEC
filing — becomes a row in `raw_item`, tagged with its `source` and the
`bank_id` it mentions. `UNIQUE (source, external_id, bank_id)` means
re-inserting something we already have is a silent no-op, so any run can be
safely repeated. Downstream phases (scoring, index) only ever read this
table; the `finbert_status` / `llm_status` columns are their work queue.

**2. Watermark + overlap.** Each (source, bank) pair remembers when it was
last polled (`watermark` table). The next run re-fetches from slightly
*before* that point (15 min for GDELT, 2 days for EDGAR) so nothing slips
through the crack between "published" and "indexed by the source" — and the
UNIQUE key throws away whatever the overlap re-fetched. This is why a poller
that crashes is self-healing: its watermark didn't move, so the next run
covers the same window again.

**3. Incremental pollers vs full loaders.** Sources that stream and won't
replay the past (news, filings, RSS) get an *incremental poller* in
`pipeline/` using the watermark pattern above. Sources that return their full
history every time (FRED, yfinance, fundamentals) will get a *stateless full
loader* in `pipeline/loaders/` that re-fetches everything and upserts into
its own table — no watermark, and it never writes `raw_item`.

**4. The structured side joins through the bank crosswalk.** Shu Han's
`unified_ffiec_fdic_dataset/` builds the fundamentals locally — FDIC BankFind
institutions, FFIEC Call Report ratios, and failure events with the
`distress_within_4q/8q` labels — into `tables/*.csv`. In Phase 2 a stateless
full loader (`pipeline/loaders/`) upserts those CSVs into Postgres; no code is
shared in either direction, the CSVs are the interface. The `bank` crosswalk
carries `fdic_cert` / `rssd_id` (verified against his `dim_bank`) so text
signals and fundamentals join on institution keys: the stability index reads
both, and the distress labels become the backtest ground truth for evals.

**5. Failures are isolated but never silent.** A bad document only loses its
excerpt; a bad bank (broken query, wrong CIK) is skipped and retried next
run while the other banks continue; a bad source fails its own workflow job
while the other source keeps running (`fail-fast: false` matrix). At every
level the failure still surfaces: the run exits non-zero naming the failed
banks, and `pipeline_heartbeat` records `ok = false`. Check
`SELECT * FROM pipeline_heartbeat ORDER BY run_at DESC` before trusting the
data.

Day-to-day operations (applying migrations, secrets, adding a bank or a new
source) are step-by-step in `RUNBOOK.md` — adding a bank is one CSV row (§5),
adding a source touches four well-marked places (§6).

## Data Sources

Structured fundamentals (built into `unified_ffiec_fdic_dataset/`):
- FFIEC Central Data Repository Call Reports
- FDIC BankFind Suite API
- FDIC Failed Bank List / failures endpoint

Text signals (polled into `raw_item` by the ingest workflow):
- GDELT DOC 2.0 — bank news articles
- SEC EDGAR — 8-K, 10-Q, 10-K filings (8-K text excerpts included)

Planned (Phase 2+): RSS feeds as an additional incremental source; FRED and
market data via stateless full loaders in `pipeline/loaders/`.
