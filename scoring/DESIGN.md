# scoring — design (Phase 2)

Companion to [README.md](README.md). Methodology agreed with the mentor
2026-07-12: **LLM-assisted labeling → fine-tuned BERT-based model → daily
batch prediction**, with keyword-level explainability. This document fixes
the concrete contracts; implementation comes after team review.

## Three labeling events, not one

All three emit the same three classes (`positive | negative | neutral`), which
makes them easy to conflate. They differ in who produces them and what for:

| | Producer | Input | Purpose | Status |
|---|---|---|---|---|
| ① **Labeling** | Llama (Kaggle GPU) | 8,360 collected rows | training answer key for FinBERT | done — 2026-07-22 batch |
| ② **Verification** | 5 teammates × 50 rows | 250-row gold slices | is ① trustworthy? (quality gate) | in progress |
| ③ **Scoring** | fine-tuned FinBERT (serving) | new rows, daily | the risk score the dashboard shows | not built |

① and ② ask a single question — *what direction does this text imply?* — which
is bank-agnostic and correct as such. ③ has a second step: its output is rolled
up per bank, and that rollup is the only place bank attribution matters.

    ① ②   text → direction                    bank-agnostic; no attribution problem
    ③     text → direction → per-bank rollup  attribution required here

Terminology used throughout this document: **labeling** = ①, **verification**
= ②, **scoring** = ③. The "Bank attribution" section concerns ③ only — it has
no effect on the labeling batch or on verification work in flight.

## Decision log

| Date | Decision | Rationale |
|---|---|---|
| 2026-07-15 | Labeler: **Llama on Kaggle free GPU, alone to start**. Gemini API is a contingency challenger, not set up initially. | Kaggle env is reused for FinBERT fine-tuning (stage 2); no API cost/ToS/rate-limit dependency. See "Quality gate" for when Gemini gets pulled in. |
| 2026-07-15 | Storage: **two new tables** (`item_label`, `item_score`), `raw_item` unchanged. | Labeling output is per-(item, labeler) history (champion-challenger + human rows must coexist); serving output is one latest score per item. Different lifecycles → different tables. Migration: `db/migrations/011_scoring_tables.sql`. |
| 2026-07-15 | Scope of this doc: design + schema contract only, no executable code yet. | Lets scoring design proceed in parallel with teammates' data-source integration. |
| 2026-07-17 | Risk lexicon is **descriptive only — never an eligibility filter** for labeling. Eligibility = language + noise filters, nothing else. | Mentor feedback: important banking-risk stories often carry no obvious negative words ("bank explores strategic alternatives", "deposits fall for third consecutive quarter", "chief risk officer departs"); a lexicon filter would drop exactly those. |
| 2026-07-17 | Corpus accounting uses a **six-stage funnel**; the export dry-run must report all six counts. | Mentor feedback: raw totals hide cross-source duplication (same story via GDELT / Alpha Vantage / NewsAPI / RSS / syndication domains); the funnel exposes the real working-set size. Stages in "Target selection" below. |
| 2026-07-25 | **Bank attribution is gated at the aggregation layer, not in `eligibility`.** Every eligible item is still labeled and scored; a separate check decides whether that score counts toward the bank it was filed under. | Measured on the 2026-07-22 batch: only **726 of 7,756** GDELT titles (9.4%) name the bank they were fetched under, and **21 of 30** signal-bearing gold-slice rows are attributed to the wrong bank. Gating this inside `eligibility` would kill the false signals but drop ~90% of the training corpus — and those rows are *valid* text→direction training examples ("Bandhan Bank Q1 crash → negative" teaches what bank distress reads like; it just isn't JPMorgan's). Keeping the two axes separate preserves the corpus and removes the false signal. Note the split by treatment: analyst/holder wire templates are a *different* problem — their directional labels are wrong-entity by construction, so they are **dropped** at Stage 1 (target selection item 3), not merely left unattributed. Only the "bank not named in its own title" class is deferred to attribution. See "Bank attribution" below. |
| 2026-07-24 | **`bank_id` is provenance, not a model feature.** FinBERT scores `text` (title, or title+excerpt for EDGAR) → direction only; the bank is attributed from `raw_item.bank_id` (the query the row was fetched under) at the aggregation layer, never from the model input. Text-only train/serve is intentional, not a gap. | Risk direction is largely bank-agnostic — "deposit outflows" is negative for any bank — so `bank_id` in the input would not improve classification. The real failure mode is *attribution*: a row fetched under bank X whose title is actually about entity Y (bank-as-analyst / holder / CEO cases — "TD issues forecast for Louisiana-Pacific", "HSBC invests in MSC", "Dimon warns UK"). That is an upstream relevance question (is the bank the subject?), not something a small encoder learns reliably from noisy labels; the human labels already mark these `neutral`. Revisit only if a stratified eval shows attribution errors a relevance gate can't fix. |
| 2026-08-07 | **Quality gate computed** over gold slices 1–6 (`pipeline/quality_gate.py` → `evals/gate_report_2026-08-07.md`). Headline (random slices 1–5): 221/250 = 88.4%. **The ≥85% placeholder is not a usable threshold** — 91.6% of human labels in the random sample are `neutral`, so an all-`neutral` labeler scores 91.6% and beats both the threshold and Llama. The gate must be restated in chance-corrected terms: **kappa 0.473 random / 0.564 pooled, macro-F1 0.648 / 0.700**. Per-class verdict still open. | Slice 6 is a deliberate directional oversample, so it strengthens per-class cells (negative n 7→14, positive 14→34) but is excluded from the headline. Decision-relevant finding: llama-side negative precision 13/36 = 36.1%, and **the training-set hygiene filters remove 0 of those 21 mislabels** — 11 are analyst/rating rows in the `keep for now` class, the rest are tone-negative general news. So this is a labeling-prompt defect, not a filter gap, and it lands directly on the ~188 `negative` training rows. |

## Shared contract (the only cross-team surface)

- `db/migrations/011_scoring_tables.sql`: `item_label` (3-class `label`,
  `label_source`, `model_meta`, one row per item×labeler) and `item_score`
  (3-class `label`, `probs`, `keywords`, `model_version`, one row per item),
  plus a partial index on `raw_item(finbert_status = 'pending')`.
- `raw_item.finbert_status` value contract (convention, no CHECK — matches
  existing free-text usage):
  - `'pending'` (default, set by ingestion) → item awaits scoring
  - `'done'` → scored, row exists in `item_score`
  - `'failed'` → scoring errored; `last_error` holds the message
  - `'skipped'` → intentionally not scored (e.g. non-English); reason in `last_error`
- `label_source` values: `'llama_kaggle' | 'gemini' | 'human'` (CHECK in 011;
  adding a labeler is the same one-line ALTER pattern as `raw_item.source`).

## Stage 1 — Labeling (build the training corpus)

Labels are assigned **from the article text alone** — never validated against
subsequent real-world events (mentor's instruction; that comparison belongs to
the backtest, which uses the distress labels in the fundamentals tables).

### Target selection (what gets labeled)

Applied at export time, not persisted. The export dry-run reports the
six-stage funnel (mentor, 2026-07-17): **total retrieved → unique after
dedup → eligible English → bank-relevant → event-relevant → selected for
labeling** — each stage's count, so the real working-set size is visible.

1. Cross-source dedup: the same story arrives via GDELT, Alpha Vantage,
   NewsAPI, RSS and syndication domains; dedupe by `title_hash` **across
   sources**, not just within one. Exact rule to be settled in the export
   script alongside the noise predicate.
2. English items only (~70% of corpus per 2026-07-16 EDA). FinBERT is
   English-only; non-English handling is a Phase-3 question.
3. Drop analyst/holder wire templates — this is the `is_syndication_noise`
   predicate (13F/holdings spam, ~14% of GDELT per EDA). Measured 2026-07-25
   against the 2026-07-22 batch: title-template matching flags **1,578 of
   7,756 GDELT rows (20.3%)** — `"X Purchases N Shares of Y"`, `"X Has $N
   Million Stake in Y"`, `"X Raises Price Target for Y"`.

   These must be **dropped**, not merely left unattributed. 196 of them carry a
   *directional* Llama label, and that direction belongs to the other entity
   ("Goldman Sachs Raises AMD Price Target" → `positive` is AMD's, not
   Goldman's). Trained on, they teach "X lifts Y's target = positive"; that
   pattern then fires on genuine bank rows and manufactures the very
   misattribution the attribution gate exists to remove. Actively harmful, not
   inert filler.

   ⚠️ **Predicate not final.** A naive `Upgrad|Downgrad` term also catches a
   bank's *own* rating change ("Commerce Bancshares Downgraded by Wall Street
   Zen to Sell" — a legitimate `negative`, confirmed in the gold slices). The
   pattern must require a **second entity as the object** before it fires.
   `n_duplicates` and `domain` remain useful signals. Settle in the export
   script.
4. Text field: GDELT items use `title` (no body is ingested); EDGAR items use
   `title` + `text_excerpt` (8-K excerpt, ≤ ~4000 chars).

The EDA risk lexicon is **not** a selection criterion (see decision log
2026-07-17): stories like "bank explores strategic alternatives" or "chief
risk officer departs" are highly risk-relevant with zero lexicon hits.
Every eligible item gets labeled regardless of lexicon match.

**EDGAR 10-Q/10-K carry no scorable text.** `poll_edgar` fetches a
primary-document excerpt for **8-K only**, and that is the right call: a 10-Q
primary document is iXBRL, so its first 4,000 characters are taxonomy URIs
rather than prose (verified 2026-07-25 on a 7.6 MB Goldman Sachs 10-Q —
readable text starts around 180,000 chars, MD&A around 654,000). But EDGAR
titles are synthesized as `"{holding_name} {form}"`, so they are never empty,
the `empty_text` guard never fires, and 107 contentless rows (103× 10-Q, 4×
10-K/A) reached the 2026-07-22 labeling batch. The whole prompt for those rows
was `Article: "Popular, Inc. 10-Q"`; Llama labeled all 107 `neutral` — a
degenerate pattern, and wasted human labeling budget. `eligibility` should
require a non-empty excerpt for EDGAR. Quarterly financial state is already
covered properly, as numbers, by the Call Report fundamentals — extracting it
from 10-Q prose would duplicate that badly.

### Kaggle round-trip contract (manual steps are explicit)

1. **Export** (repo script, later): SELECT eligible `raw_item` rows →
   `labeling_batch_<date>.csv` with columns
   `raw_item_id, source, bank_id, published_at, title, text_excerpt`.
2. **Upload** CSV as a private Kaggle dataset (manual).
3. **Label** (Kaggle notebook, Llama 8B-class): prompt yields exactly one of
   `positive|negative|neutral` per row → `labels_<date>.csv` with columns
   `raw_item_id, label, model_meta` (`model_meta` JSON: model id, quantization,
   prompt version, run date).
4. **Import** (repo script, later): upsert into `item_label` with
   `label_source='llama_kaggle'`; `ON CONFLICT (raw_item_id, label_source)`
   update — relabeling with a new prompt overwrites the labeler's row, and the
   old prompt version stays visible in `model_meta` history via git, not DB.

`llm_status` / `llm_attempts` on `raw_item` are available for labeling
bookkeeping (e.g. `llm_status='labeled'`), but the source of truth is
`item_label`; no schema change either way (README's open option preserved).

### Hand verification & quality gate

- Randomly sample **200–300 labeled items**, stratified by class and source;
  a human labels them blind (`label_source='human'` rows in `item_label`).
- Measure per-class disagreement vs `llama_kaggle`. Random noise is tolerable
  (fine-tuning is robust to it); **systematic bias is not** (e.g. all
  regulatory news labeled negative) — the review focuses on finding patterns.
- **Quality gate (threshold TBD by team, placeholder: ≥85% agreement overall,
  no class below 75%).** If the gate fails: revise prompt → relabel → if it
  still fails, activate the **Gemini challenger**: label the same corpus with
  `label_source='gemini'`, hand-review focuses on Llama/Gemini disagreement
  rows (`GROUP BY raw_item_id HAVING count(DISTINCT label) > 1`), and the team
  picks the champion on measured agreement with human labels.

## Stage 2 — Training (fine-tune FinBERT)

Pretrained-only FinBERT is not sufficient per the mentor; fine-tune on the
stage-1 labels. Same Kaggle GPU environment as labeling (workflow reuse).

- Training set: `item_label` rows from the champion labeler, with `human`
  rows overriding the champion's where both exist.
- Split: **time-based** train/validation (e.g. last N weeks held out) rather
  than random — matches serving reality and avoids syndication near-duplicates
  leaking across the split. Watch per-bank concentration (a few banks dominate
  GDELT volume).
- Report accuracy + per-class F1 on the held-out slice; compare against
  pretrained-only FinBERT as the baseline to demonstrate the fine-tune helped.
- Artifact: model weights versioned outside the repo (Kaggle dataset or HF
  hub, TBD); `item_score.model_version` records which weights scored each row.

### Training-set hygiene (applied at training-set construction, not in the pipeline)

Stage 2 is gated on Stage 1: the training set is the **champion** labeler's
rows, and the champion is chosen by the quality gate. A failed gate means
prompt revision → relabel → every label changes, so training before the gate
resolves risks throwing the run away.

When the gate does resolve, some rows should be excluded from the training set.
This is a **filter where the training set is assembled** — it needs no change
to `eligibility`, no re-export, and no re-labeling:

| Class | Rows | Directional labels | Action |
|---|---|---|---|
| Contentless EDGAR (10-Q / 10-K/A, no excerpt) | 107 | 0 | exclude — degenerate, all `neutral` |
| Holdings/13F wire spam ("X Purchases N Shares of Y") | 723 | 51 | exclude — directions are wrong-entity |
| Rating / price-target templates | 375 | 123 | **keep for now** — see below |

The third class is **not** safely droppable yet. The same surface form covers
opposite roles:

    "Pearson downgraded by JP Morgan"            bank_id=jpm   → jpm is the actor  → drop
    "Commerce Bancshares Downgraded by WSZ"      bank_id=cbsh  → cbsh is the target → keep

Dropping the class wholesale would discard 123 directional labels including
genuine bank rating changes. A rule must first distinguish whether the
attributed bank is the *subject* or the *agent* (see Stage 1 item 3).

⚠️ **This filter must land in `eligibility` before Stage 3 serving ships.**
Filtering at training only, while serving still scores those rows, is exactly
the train/serve skew the shared-eligibility design exists to prevent.

**Known risk — tone-prior vs risk-direction.** FinBERT is pretrained on
financial *tone*; our target is *risk direction*. The two diverge exactly on
the euphemistic cases the labeling guide flags as most valuable — "exploring
strategic alternatives" (calm tone, negative), "consent order lifted"
("regulator" wording, positive). Because FinBERT is a distillation of the
stage-1 Llama labels, it cannot exceed them on these cases. Therefore the
quality gate (§ Hand verification) must be read **stratified on the
tone≠direction subgroup**, not only overall: overall agreement can pass while
the high-value euphemistic cases fail. This — not the absence of `bank_id`
(see decision log 2026-07-24) — is the primary correctness risk of the
FinBERT stage.

## Stage 3 — Serving (daily batch)

Runs on the existing scheduled infra (GitHub Actions, CPU-only — FinBERT-size
models are fine on CPU; this is why serving is not an LLM, see QnA discussion).

1. SELECT from `raw_item` WHERE `finbert_status='pending'` (partial index).
2. Apply the same eligibility filter as labeling; ineligible → `'skipped'`.
3. Score in batches → upsert `item_score` (`label`, `probs`, `model_version`).
4. Mark `finbert_status='done'` (or `'failed'` + `last_error`); the two writes
   should happen in one transaction per batch.
5. Heartbeat via existing `write_heartbeat` (`pipeline/db.py`) like pollers.

**Keywords / explainability (v2):** standout keywords per sentiment cluster
(PCA / keyword clustering) fill `item_score.keywords` in a later iteration;
the column exists now so the dashboard contract is stable.

## Bank attribution — why a scored item may not count for its bank

### The problem, concretely

GDELT matches a query against the **full article body**, but its `artlist` API
returns only the **title** (verified 2026-07-25: each article object carries
`domain, language, seendate, socialimage, sourcecountry, title, url,
url_mobile` — no body, no excerpt, no matched snippet). So a row filed under
`gs` can arrive looking like this:

    bank_id : gs
    title   : "Chipmaker SK Hynix raises $26.5bn in US stock market debut"

The match was legitimate — the body said Goldman Sachs was a bookrunner — but
the headline is about SK Hynix. Score that headline `positive` and credit it to
`gs`, and Goldman gets a good-news point it never earned.

This is not an edge case:

| Measurement (2026-07-22 batch / gold slices) | |
|---|---|
| GDELT titles that name their own `bank_id` | **726 / 7,756 (9.4%)** |
| Signal rows (non-neutral) attributed to the wrong bank | **21 / 30 (70%)** |

Neutral noise is harmless — direction 0 either way. Misattributed **signal** is
not: it pushes a real direction onto the wrong bank.

### Why this is not an `eligibility` filter

`eligibility` is the shared **text** gate: "is there anything here to score?"
Attribution answers a different question: "does this score belong to *this*
bank?" Collapsing the two is tempting and expensive:

- Requiring the bank to be named in the title would drop **~90% of the GDELT
  corpus** before labeling ever happens.
- Those rows are *valid* training data. "Bandhan Bank Q1 results beat
  estimates, but stock crashes" is a textbook bank-distress headline and its
  `negative` label is correct — it simply is not JPMorgan's `negative`.

So the two axes stay separate:

    labeling / training    every eligible item        → corpus preserved
    scoring                every eligible item        → for now; see below
    bank-level rollup      attributable items only    → false signal removed

**Scoring non-attributable rows is a phase, not a permanent rule.** A row is
bank-specific — `raw_item` is UNIQUE on `(source, external_id, bank_id)`, so an
article matched by two banks' queries becomes two rows — and a row's score is
only ever read by its own bank's rollup. Once a row fails attribution, its
score is read by nobody.

It is still worth computing *while the gate is young*. The gate was validated
on 6 rows. Asking "is it discarding real signal?" requires scores on the
discarded side; skip the scoring and the rejected pile is unreadable, so there
is no evidence with which to widen or tighten it.

    gate unvalidated (now)   score everything          → gate stays measurable
    gate stable              finbert_status='skipped'  → stop paying for it

Cost is not the deciding factor either way — FinBERT is CPU-sized and daily
increments are small; the 7,756-row figure is one-time backlog. The switch is
cheap in both directions: `'skipped'` is already in the status contract, and
resetting it to `'pending'` puts rows back into the daily batch.

### The check

**Is the bank named in its own title?** Word-boundary match against
`bank.aliases` (already seeded). `\b` is required — bare `"Citi"` otherwise
matches "Citizens". Rows that fail are still labeled and scored; they are
excluded from that bank's rollup only.

That is the whole attribution check. Analyst/holder wire templates ("Goldman
Sachs Raises AMD Price Target") are **not** handled here — they are dropped
earlier, at Stage 1 target selection item 3, because their directional labels
are wrong-entity by construction and poison training. The two predicates are
split by *treatment*, deliberately:

    harmful to train on          → drop at Stage 1        (wire templates)
    valid to train on, misfiled  → don't attribute here   (bank not named)

Validated against `gold_slice_3` (42 GDELT rows — the slice carrying per-row
reasoning): all 3 misattributed signal rows blocked, all 3 correctly-attributed
signal rows kept. One analyst-template row cleared this check, which is exactly
why that class is handled at Stage 1 instead. n is small — re-measure once more
slices are labeled.

## Backtest linkage (out of scoring's scope, for orientation)

Sentiment scores join fundamentals via
`bank.fdic_cert ↔ fact_bank_quarter.fdic_cert_number` (no FK, join by value).
Distress labels: `fact_distress_event`, `fact_bank_quarter.distress_within_4q/8q`.
Caveat from EDA: only ~27 of ~3,627 failure events fall in the GDELT era
(2017+), and 4-quarter positives are ~0.45% of bank-quarters — the backtest is
a rare-event evaluation and must be designed accordingly.

## Open items (team decisions — README: "teammates own all design decisions")

- [ ] Owner of scoring/ (currently TBD)
- [ ] Quality-gate thresholds (placeholders above)
- [ ] Exact syndication-noise predicate for target selection
- [ ] Model-weights hosting (Kaggle dataset vs HF hub)
- [ ] `probs` as jsonb vs three numeric columns (jsonb chosen for now; revisit
      if the dashboard needs to sort/filter on probabilities directly)
- [ ] Whether the inference-time LLM escalation tier (low-confidence items →
      LLM) is pursued — `llm_status`/`llm_attempts` kept unused for it
