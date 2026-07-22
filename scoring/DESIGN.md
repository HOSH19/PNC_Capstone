# scoring — design (Phase 2)

Companion to [README.md](README.md). Methodology agreed with the mentor
2026-07-12: **LLM-assisted labeling → fine-tuned BERT-based model → daily
batch prediction**, with keyword-level explainability. This document fixes
the concrete contracts; implementation comes after team review.

## Decision log

| Date | Decision | Rationale |
|---|---|---|
| 2026-07-15 | Labeler: **Llama on Kaggle free GPU, alone to start**. Gemini API is a contingency challenger, not set up initially. | Kaggle env is reused for FinBERT fine-tuning (stage 2); no API cost/ToS/rate-limit dependency. See "Quality gate" for when Gemini gets pulled in. |
| 2026-07-15 | Storage: **two new tables** (`item_label`, `item_score`), `raw_item` unchanged. | Labeling output is per-(item, labeler) history (champion-challenger + human rows must coexist); serving output is one latest score per item. Different lifecycles → different tables. Migration: `db/migrations/011_scoring_tables.sql`. |
| 2026-07-15 | Scope of this doc: design + schema contract only, no executable code yet. | Lets scoring design proceed in parallel with teammates' data-source integration. |
| 2026-07-17 | Risk lexicon is **descriptive only — never an eligibility filter** for labeling. Eligibility = language + noise filters, nothing else. | Mentor feedback: important banking-risk stories often carry no obvious negative words ("bank explores strategic alternatives", "deposits fall for third consecutive quarter", "chief risk officer departs"); a lexicon filter would drop exactly those. |
| 2026-07-17 | Corpus accounting uses a **six-stage funnel**; the export dry-run must report all six counts. | Mentor feedback: raw totals hide cross-source duplication (same story via GDELT / Alpha Vantage / NewsAPI / RSS / syndication domains); the funnel exposes the real working-set size. Stages in "Target selection" below. |

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
3. Drop 13F/holdings syndication noise (~14% of GDELT per EDA): exclude rows
   whose `title_hash` cluster is holdings-spam; `n_duplicates` and `domain`
   help identify these. Exact predicate to be settled in the export script.
4. Text field: GDELT items use `title` (no body is ingested); EDGAR items use
   `title` + `text_excerpt` (8-K excerpt, ≤ ~4000 chars).

The EDA risk lexicon is **not** a selection criterion (see decision log
2026-07-17): stories like "bank explores strategic alternatives" or "chief
risk officer departs" are highly risk-relevant with zero lexicon hits.
Every eligible item gets labeled regardless of lexicon match.

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
