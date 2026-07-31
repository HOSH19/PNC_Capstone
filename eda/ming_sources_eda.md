# EDA — Ming's sources (CFPB complaints, market prices, FRED macro)

Complements `2026-07-16_ingest_eda.md`, which covered the `raw_item` text corpus
+ fundamentals. This covers the three tables that EDA missed: `cfpb_complaint`,
`market_daily`, `fred_observation`. CFPB numbers are from a full-history local
extract (~1.2M matched complaints, 2011–2026); prices/FRED from the live DB.

**Framing.** Every source is read as a candidate **input to the per-bank
risk / sentiment score**, judged on (a) **coverage** — which banks can be
scored, (b) **signal** — does it move meaningfully, (c) **explainability** —
can it be decomposed into human-readable drivers.

---

## 1. CFPB consumer complaints

**Coverage — skewed to large, consumer-facing banks.**
- 92 of 104 banks have complaints; 30 have ≥1000, 22 have <100 (thin, unreliable
  to score).
- Top 10 banks = **87%** of all complaints (BofA 180k, Wells 168k, JPM 168k,
  Cap One 161k, Citi 134k, Synchrony 79k, Amex 57k, USB 48k, **PNC 31k**,
  Ally 29k).
- Commercial/regional banks are near-absent (Western Alliance ~0/month) — CFPB
  is a *consumer* signal, so it only scores consumer-facing banks.

**Volume signal — strong, up 2.6× since 2020.**
- Flat ~5k/mo 2012–2020, then accelerates: 2020 71k → 2023 117k → 2025 182k/yr
  (an industry-wide surge, largely credit-reporting). → volume is a live signal,
  but must be **normalized per bank** (raw levels are dominated by size).

![CFPB complaints/month (full history)](charts/cfpb_volume_trend.png)

**Category mix — the explainability dimension.**
- Products: checking/savings 21%, credit card 16%, mortgage 15%, credit
  reporting ~16%. Issues: managing an account 13%, incorrect report info 8.5%,
  loan mod/foreclosure 5.4%.
- → a score move can be attributed to a category ("driven by mortgage-servicing
  complaints").

![Complaints by product](charts/cfpb_product_mix.png)

**Narrative text — ~466k usable, feeds the BERT/sentiment track.**
- 38% overall have narrative; 43–54% for mature years, only 21% for 2026 (CFPB
  publishes narratives months late). Median 893 chars.
- → plenty of text to fine-tune a sentiment model, but fresh complaints have no
  narrative, so the **text (sentiment) signal lags the volume signal by months**.

![% of complaints with narrative, by year](charts/cfpb_narrative_rate.png)

**Company response.** 74% closed-with-explanation, 12% monetary relief, 11%
non-monetary, 1.2% no relief; timely 99% (not discriminating). Relief type is a
weak resolution-quality signal.

**Complaint keywords — what each bank is complained about (core of explainability).**

Overall, the top narrative terms are card / payment / money / charge / dispute /
fraud — complaints center on **cards, payments, fees, disputes, fraud**:

![Overall complaint-narrative word cloud](charts/cfpb_wordcloud_overall.png)

More useful are each bank's **distinctive keywords**: TF-IDF across banks (with
**each bank's own name stripped**), leaving the themes that set a bank apart and
revealing which business line its complaints concentrate on:

| Bank | Distinctive keywords (TF-IDF) | Business / theme revealed |
|---|---|---|
| wfc Wells Fargo | assistance center, **hamp**, everyday checking, safe deposit box | mortgage modification (HAMP), checking, safe deposit |
| bac Bank of America | **countrywide**, **lynch** (Merrill), identity theft, flood insurance | mortgage legacy (Countrywide), identity theft |
| jpm Chase | **sapphire**, amazon, **freedom**, marriott | co-brand credit cards (Sapphire/Freedom/Amazon) |
| cof Capital One | **auto finance**, walmart, efta requires, timely access | auto loans, Walmart card, EFT disputes |
| citi Citi | **best buy**, **home depot**, wayfair | retail co-brand cards (Best Buy/Home Depot) |
| syf Synchrony | **paypal, amazon, lowes, walmart, store card** | retail store cards (the whole line) |
| axp Amex | bluebird, membership rewards, platinum, delta | premium cards / rewards / Delta co-brand |
| usb U.S. Bancorp | fidelity, visa gift, **unemployment benefits**, cardmember | prepaid/gift & benefit cards |
| ally Ally | **auto finance, leased, gap insurance, buyout**, scams | auto lending/leasing (core), fraud |
| pnc PNC | **virtual wallet**, national city, link accounts, escrow | Virtual Wallet product, account linking |
| tfc Truist | **service finance, solar, installation**, obligor | solar/home-improvement financing (Service Finance) |
| gs Goldman | **apple card**, apple, marcus | Apple Card + Marcus consumer arm |

![Per-bank distinctive-keyword word clouds (bank names removed)](charts/cfpb_wordcloud_perbank.png)

→ **This is the basis for dashboard explainability**: not just a risk/sentiment
number, but "which business line drives this bank's negatives" (e.g. Wells Fargo
= mortgage servicing, Ally = auto lending, Synchrony = retail store cards). Add
BERT sentiment + category labels and the score decomposes into explainable
"which complaint type, what sentiment" drivers.

**Complaint topics over time — topics move on different clocks.**

Split complaints by topic (product category) and the dynamics differ completely;
an aggregate count flattens all of this:
- **Mortgage**: peaked 2013 (foreclosure-crisis aftermath), then **declines**
  (only ~1200/quarter by 2024).
- **Credit reporting**: ~0 in 2012 → **explodes after 2020**, peak ~12400/quarter
  in 2025.
- **Credit card / checking-savings**: steady growth.
- **Fraud/scam issues**: 775/yr (2018) → 2944/yr (2024), **~4×** — a
  risk-relevant, rising topic.

![CFPB complaints by topic, quarterly](charts/cfpb_topic_trends.png)

The mix shift is clearer as shares — the **topic composition keeps drifting**
(mortgage recedes, credit-reporting/account complaints rise):

![Complaint mix over time (share by topic)](charts/cfpb_topic_mix.png)

→ Modeling implication: **topic-level signal carries far more information than a
generic positive/negative or an aggregate count.** "More complaints" could be a
mortgage-servicing problem or a fraud surge — very different risk meanings. CFPB's
built-in categories make it a natural testbed for **topic-specific sentiment**.

---

## 2. market_daily (price-based risk)

**Coverage — dense (its differentiator).** 103/104 banks have ≥5yr of daily
prices back to 2008. Unlike news/complaints, the **price signal can score every
bank**, including small commercial ones.

**Signal correctly flags real distress.** 3-yr metrics: median annual vol 0.30;
worst drawdowns FLG −0.80 (Flagstar/NYCB), LOB −0.54, SOFI −0.53; most volatile
SOFI, FLG, WAL (Western Alliance) — exactly the banks under real 2023–24 regional
stress. Drawdown/vol are inherently explainable.

![Per-bank volatility and max-drawdown distributions](charts/price_risk_dist.png)

![Big-bank prices (from 2008; crisis dips visible)](charts/big_bank_prices.png)

---

## 3. fred_observation (macro backdrop)

5 series, deep history (10y–FedFunds spread back to 1962). The financial-stress
index, HY credit spread, and 10y–FedFunds spread all spike at 2008/2020/2023.
Role: a **systemic-risk backdrop / common multiplier**, not a per-bank signal
(when macro stress is high, all banks' risk rises together).

![Macro stress signals (spikes at 2008/2020/2023)](charts/macro_stress.png)

---

## 4. How these signals relate to "distress": leading-ness and complementarity

### 4.1 Leading-ness — complaints are *reactive*, price *leads*

Two natural experiments test whether these signals give early warning of distress.

**Case A — Wells Fargo 2016 fake-accounts scandal (announced 2016-09).**
Complaints were flat at ~770/mo, and **only jumped to 1590 (2.1×) in the month
the scandal broke (2016-09)**, then decayed. → complaints **react to the public
event, they don't lead it** (consumers tend to file after media coverage —
inherently lagging).

![Wells Fargo complaints/mo (2016 scandal)](charts/distress_wells2016.png)

**Case B — 2023 regional-bank crisis (SVB collapse 2023-03).**

| Bank | Price drawdown (Mar–May) | Complaint change (vs 2022) |
|---|---|---|
| Western Alliance | **−74%** | 35× (but ~0 baseline → noise) |
| Zions | **−61%** | 12× (~6/mo baseline → noise) |
| KeyCorp | **−52%** | only 1.3× |
| Citizens | **−40%** | only 1.3× |

Prices **cratered 40–74% within days**, while banks with real complaint volume
(KeyCorp/Citizens) rose only 1.3×. → **price reacts fast / leads; complaints
barely move** — because this was a solvency/liquidity (bank-run) event, not a
consumer-conduct one.

![Price vs complaints (2023 crisis): price craters, complaints don't move](charts/distress_price_vs_complaints.png)

**Takeaways:**
- **Complaint volume = a "reactive / severity" signal**: useful for
  publicly-surfaced **consumer-conduct** events (severity confirmation, e.g.
  Wells 2.1× and sustained), but **not leading**, and near-silent for
  solvency crises.
- **Price = a "leading / market" signal**: reacts fastest, covers all banks.
- → Clear division of labor: **price as the "leading" component, complaints as
  the "conduct-risk / severity" component.**
- *Caveat*: two case studies give directional evidence; a rigorous answer needs
  an event-study lead-lag analysis across many events.

### 4.2 Complementarity — complaints/price are ~uncorrelated with fundamentals

Put each bank's **own signals** (complaint rate, price drawdown, vol) next to the
**fundamentals** (NPL, Tier1 capital, liquidity) and compute cross-bank
correlations (all 104 banks join via `bank.fdic_cert → fact_bank_quarter`):

| Signal | vs NPL | vs Tier1 capital | vs Liquidity |
|---|---|---|---|
| Complaint rate (per assets) | 0.18 | −0.11 | 0.06 |
| Price drawdown | −0.01 | 0.16 | 0.16 |
| Price vol | 0.16 | −0.19 | −0.18 |

![Signal correlations — complaints/price vs fundamentals](charts/signal_correlation.png)

- All correlations between my three signals and the fundamentals are **low
  (|corr| ≤ 0.19)**; the only strong ones are internal (drawdown vs vol −0.80,
  capital vs liquidity 0.31).
- → **The complaint/price signals occupy a different "information space" than
  fundamentals — complementary, not redundant.** Adding them to a
  fundamentals-only model should surface information the fundamentals miss (and
  fundamentals are **quarterly, lagged** while price is **daily, leading**).
- *Caveat*: low correlation is evidence of non-redundancy; whether it actually
  improves prediction still needs a model to confirm.

---

## 5. Cross-cutting takeaways for the scoring dashboard

1. **Coverage is complementary, not uniform — it dictates how to combine the
   score.** Price = all banks; complaints = large consumer banks; news = large
   banks only. So weight each component by its per-bank availability: a small
   commercial bank is scored mostly on price + fundamentals; a large consumer
   bank gets the full sentiment + complaints + price stack.

2. **The distress label barely overlaps our universe.** FDIC failures are small
   community banks; ~0 of our top-100 appear. → confirms the plan to **broaden
   the label to near-distress** (enforcement actions, severe drawdowns), which
   our own enforcement + price-drawdown data can supply.

3. **Complaints are "reactive", price is "leading".** Wells 2016 complaints
   jumped only when the scandal broke; in 2023 prices fell 40–74% while
   complaints didn't move. → **price as the "leading" component, complaints as
   the "consumer-conduct-risk / severity" component.**

4. **Our signals are ~uncorrelated with fundamentals (|corr|≤0.19) —
   complementary, not redundant.** Complaint rate / drawdown / vol are
   independent of NPL / capital / liquidity → adding them to a fundamentals-only
   model should bring new information (and fundamentals are quarterly/lagged,
   price is daily/leading).

5. **Topic-level beats aggregate.** Complaint topics run on different clocks
   (mortgage down since 2013, credit-reporting exploded post-2020, fraud ~4×).
   "More complaints" can mean very different things → topic-level signal +
   sentiment carries far more information than one aggregate count.

6. **The text signal inherently lags.** ~466k narratives are great for training,
   but fresh complaints have no narrative for months — **sentiment is inherently
   delayed vs the complaint-volume signal** (matters for a real-time dashboard).

---

*Charts are inlined per section; source images in `eda/charts/`.*
