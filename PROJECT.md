# Sentiment Analysis on Call Reports and Financial News for Active U.S. Banks

## Project Description

U.S. commercial banks operate in an environment where financial deterioration can unfold gradually through balance-sheet weakness or suddenly through confidence shocks, liquidity pressure, adverse news, deposit flight, or market reaction. Traditional bank surveillance depends heavily on quarterly FFIEC Call Reports, which provide valuable structured financial data but arrive with a delay and may not fully capture fast-moving disruption. The recent regional banking crisis demonstrated that public sentiment, news coverage, market perception, and confidence dynamics can materially accelerate distress long before the next regulatory filing becomes available.

This capstone project proposes a **real-time sentiment-driven early warning analytics platform** for active U.S. banks. The project will combine structured regulatory data from Call Reports with unstructured text sources such as financial news, SEC filings, earnings commentary, press releases, and potentially social media or public web signals. The goal is to determine whether negative sentiment, topic shifts, and emerging risk narratives can provide earlier warning of bank disruption than quarterly financial ratios alone.

The project can focus on active FDIC-insured U.S. commercial banks above a defined asset threshold, such as **$1 billion**, to ensure adequate data availability. For each bank, the system would collect quarterly Call Report indicators and align them with daily or weekly text-based sentiment indicators. Natural language processing methods such as FinBERT, domain-specific transformer models, topic modeling, named entity recognition, and event classification can be used to extract sentiment and risk themes from news articles and filings. Examples of relevant themes include deposit outflows, liquidity stress, unrealized securities losses, capital raises, regulatory actions, commercial real estate exposure, earnings deterioration, cyber incidents, executive turnover, and rating downgrades.

A major research question is whether sentiment deterioration precedes measurable financial deterioration. The project can test this by constructing bank-level sentiment time series and comparing them against future distress labels, including FDIC failures, enforcement actions, PCA capital triggers, abnormal equity declines, or severe risk-tier movement. The modeling approach may include logistic regression, XGBoost, time-series classification, survival analysis, or Bayesian updating. A particularly strong design would use Call Report fundamentals as the baseline risk layer and sentiment as a fast-moving signal layer that updates the prior risk estimate.

The real-time analytics component would be central to the capstone. The final output could be an interactive dashboard that monitors active banks, displays current sentiment trends, highlights major negative news events, shows bank-level risk narratives, and provides explainable alerts. For example, a bank may be flagged not only because its sentiment score dropped, but because the decline was driven by repeated references to deposit pressure, securities losses, or regulatory scrutiny. This creates a more interpretable and actionable early warning system.

---

## Problem Statement

Traditional bank monitoring relies on quarterly regulatory filings that lag real-world confidence dynamics. This project asks whether **unstructured text sentiment** can complement structured Call Report fundamentals to detect imminent bank disruption earlier and more interpretably.

**Primary objective:** Build a pipeline that ingests financial reports and text sources, extracts sentiment and risk themes, and outputs a **classification of financial health** for each bank.

---

## Potential Research Questions

1. Do negative financial news sentiment signals provide measurable lead time before formal bank distress events?
2. Which sentiment sources are most predictive: financial news, SEC 8-K filings, earnings transcripts, or public market commentary?
3. Does sentiment improve prediction performance when combined with quarterly Call Report fundamentals?
4. Can topic-specific sentiment, such as liquidity stress or capital weakness, outperform general positive/negative sentiment?
5. How can a real-time dashboard present sentiment alerts responsibly without creating misleading or alarmist conclusions?

---

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         DATA INGESTION LAYER                            │
├──────────────────────┬──────────────────────┬───────────────────────────┤
│  Regulatory (Core)   │  Text Sources        │  Labels & Events          │
│  • FFIEC Call Reports│  • GDELT / News APIs │  • FDIC Failures          │
│  • FDIC BankFind     │  • SEC EDGAR         │  • Enforcement actions    │
│                      │  • Earnings text     │  • Market stress proxies    │
└──────────┬───────────┴──────────┬───────────┴─────────────┬─────────────┘
           │                      │                         │
           ▼                      ▼                         ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    ENTITY RESOLUTION & COMMON SCHEMA                      │
│         Join on FDIC Certificate / RSSD ID / CIK / Ticker               │
└─────────────────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         NLP & FEATURE LAYER                             │
│  FinBERT / transformers • Topic modeling • NER • Event classification   │
│  Risk themes: liquidity, CRE, capital, deposits, regulatory scrutiny    │
└─────────────────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         MODELING LAYER                                  │
│  Baseline: Call Report fundamentals                                     │
│  Enhanced: fundamentals + sentiment time series                         │
│  Methods: logistic regression, XGBoost, survival analysis, Bayesian     │
└─────────────────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         OUTPUT                                          │
│  Financial health classification • Risk narratives • Dashboard alerts   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Data Architecture: Common Schema (Shu Han — Core Regulatory Sources)

Three **Core** sources are assigned to Shu Han. All three share **FDIC Certificate** and **RSSD ID** as join keys, making a unified bank-centric schema feasible.

| Source | Role | Grain | Primary Key |
|--------|------|-------|-------------|
| FDIC BankFind Suite API | Bank universe & reference | Institution (snapshot + history) | `fdic_cert_number` |
| FFIEC Call Reports (CDR) | Financial fundamentals (features) | Bank × quarter | `fdic_cert_number` + `report_date` |
| FDIC Failed Bank List | Distress labels (targets) | Bank × failure event | `fdic_cert_number` + `failure_date` |

### Recommended Schema Design

Rather than forcing all sources into one flat table, use a **star schema** with a shared bank dimension and role-specific fact tables.

#### `dim_bank` (from FDIC BankFind)

Master institution table. One row per bank (or per bank lifecycle if mergers are tracked).

| Column | Type | Source |
|--------|------|--------|
| `fdic_cert_number` | INT (PK) | BankFind |
| `rssd_id` | INT | BankFind / Call Reports |
| `bank_name` | STRING | BankFind |
| `city` | STRING | BankFind |
| `state` | STRING | BankFind |
| `active_status` | BOOLEAN | BankFind |
| `total_assets` | FLOAT | BankFind (latest) |
| `charter_type` | STRING | BankFind |
| `established_date` | DATE | BankFind |
| `last_updated` | DATE | BankFind |

#### `fact_call_report` (from FFIEC CDR)

Quarterly structured financials — the **baseline risk layer**.

| Column | Type | Source |
|--------|------|--------|
| `fdic_cert_number` | INT (FK → dim_bank) | Call Reports |
| `rssd_id` | INT | Call Reports |
| `report_date` | DATE | Call Reports (quarter-end) |
| `total_assets` | FLOAT | Call Reports |
| `total_deposits` | FLOAT | Call Reports |
| `tier1_capital_ratio` | FLOAT | Call Reports |
| `total_capital_ratio` | FLOAT | Call Reports |
| `npl_ratio` | FLOAT | Call Reports |
| `loan_loss_allowance_ratio` | FLOAT | Call Reports |
| `liquidity_ratio` | FLOAT | Call Reports |
| `securities_unrealized_loss` | FLOAT | Call Reports |
| `cre_loans` | FLOAT | Call Reports |
| `loan_mix_*` | FLOAT | Call Reports (as needed) |

> Call Reports contain thousands of fields. Start with a curated subset aligned to distress predictors (capital, asset quality, liquidity, CRE exposure) and expand as modeling requires.

#### `fact_distress_event` (from FDIC Failures endpoint)

Event-level distress labels for supervised learning and case studies.

| Column | Type | Source |
|--------|------|--------|
| `fdic_cert_number` | INT (FK → dim_bank) | Failures API |
| `failure_date` | DATE | Failures API |
| `bank_name` | STRING | Failures API |
| `city` | STRING | Failures API |
| `state` | STRING | Failures API |
| `acquiring_institution` | STRING | Failures API |
| `event_type` | STRING | `failure` / `assistance` |
| `distress_label` | INT | 1 = distress event |

#### `fact_bank_quarter` (modeling view — derived)

Analytical grain for training: one row per bank per quarter, joining fundamentals with forward-looking distress labels.

| Column | Type | Notes |
|--------|------|-------|
| `fdic_cert_number` | INT | Join key |
| `quarter_end_date` | DATE | Observation date |
| `*_financial_features` | FLOAT | From `fact_call_report` |
| `distress_within_4q` | INT | Label: failure within next 4 quarters |
| `distress_within_8q` | INT | Label: failure within next 8 quarters |
| `days_to_distress` | INT | For survival analysis |

### Schema Fit Assessment

| Criterion | Assessment |
|-----------|------------|
| **Shared identifiers** | Strong — all three sources use `fdic_cert_number`; Call Reports and BankFind also share `rssd_id` |
| **Complementary roles** | Strong — reference (universe) + features (fundamentals) + labels (failures) map cleanly to dim/fact pattern |
| **Temporal alignment** | Moderate — Call Reports are quarterly; failures are sparse events; labels must be constructed with forward windows |
| **Field overlap** | Moderate — BankFind and Call Reports both provide financials; prefer Call Reports for quarterly depth, BankFind for universe filtering |
| **Class imbalance** | Risk — failures are rare; supplement with enforcement actions, PCA triggers, or equity drawdowns (other team sources) |

**Conclusion:** Shu Han's three datasets fit a common schema well. Use `fdic_cert_number` as the spine, `dim_bank` as the dimension table, and separate fact tables for quarterly fundamentals and distress events. When text sources (GDELT, SEC EDGAR) are integrated by other team members, add `fact_sentiment` at bank × day/week grain joined through the same `dim_bank` keys (with CIK/ticker mapping as an extension).

---

## Modeling Approach

### Baseline Layer (Structured)

Quarterly Call Report ratios provide the **prior risk estimate**: capital adequacy, asset quality, liquidity, earnings, and CRE concentration.

### Signal Layer (Unstructured)

Daily or weekly sentiment indices from news, filings, and earnings commentary update the prior with fast-moving public perception signals.

### Classification Output

| Health Tier | Description |
|-------------|-------------|
| **Stable** | Fundamentals and sentiment within normal range |
| **Watch** | Sentiment deterioration or emerging risk themes without fundamental confirmation |
| **Elevated Risk** | Sentiment decline aligned with weakening fundamentals |
| **Imminent Disruption** | Strong negative sentiment + critical fundamentals or confirmed distress event |

### Evaluation

- Compare baseline (Call Reports only) vs. sentiment-enhanced models
- Measure lead time: how many quarters/days before distress does sentiment signal fire?
- Use precision/recall, AUC-ROC, and survival curves given rare positive class

---

## Expected Deliverables

1. **Curated bank-level dataset** aligned to the common schema above
2. **Reproducible NLP pipeline** for sentiment and topic extraction
3. **Sentiment and topic features** at bank × time grain
4. **Predictive modeling experiments** (baseline vs. enhanced)
5. **Real-time scoring framework** for ongoing monitoring
6. **Interactive dashboard** with explainable alerts
7. **Written research report** covering data sources, methodology, performance, ethics, and limitations

---

## Research Value

This project addresses a major limitation in traditional bank monitoring: the **timing gap** between quarterly regulatory filings and rapidly changing market confidence. It extends existing early warning frameworks by treating public sentiment as a measurable, fast-moving risk signal. The prototype can monitor active U.S. banks in near real time while remaining grounded in publicly available data and academic research objectives.

---

## Ethical Considerations & Limitations

- **False alarms:** Sentiment signals can be noisy; dashboard alerts must be framed as risk indicators, not accusations
- **Entity matching errors:** News may reference holding companies, subsidiaries, or similarly named institutions
- **Survivorship and class imbalance:** Bank failures are rare; model calibration and supplementary labels are essential
- **Publication bias:** News coverage intensity varies by bank size and geography
- **Lagged filings:** Call Reports release after quarter-end; sentiment may lead but fundamentals anchor ground truth

---

## Data Source Ownership (Team)

| Owner | Core Sources |
|-------|-------------|
| **Shu Han** | FFIEC Call Reports, FDIC BankFind API, FDIC Failed Bank List |
| **Jiwon** | SEC EDGAR, GDELT |
| **Yusheng** | FDIC/OCC/Fed enforcement actions, earnings text |
| **Rita** | NewsAPI, Alpha Vantage, Benzinga (supplement) |
| **Ming** | Reuters archive, FRED, CFPB complaints, yfinance, agency RSS (supplement) |

See `data_source.csv` for the full inventory with links, access methods, and validation status.
