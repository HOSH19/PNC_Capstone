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
| 2026-08-07 | **Quality gate computed** over gold slices 1–6 (`pipeline/quality_gate.py` → `evals/gate_report_2026-08-07.md`). Headline (random slices 1–5): 221/250 = 88.4%. **The ≥85% placeholder is not a usable threshold** — 91.6% of human labels in the random sample are `neutral`, so an all-`neutral` labeler scores 91.6% and beats both the threshold and Llama. The gate must be restated in chance-corrected terms: **kappa 0.473 random / 0.564 pooled, macro-F1 0.648 / 0.700**. Per-class verdict still open. | Slice 6 is a deliberate directional oversample, so it strengthens per-class cells (negative n 7→14, positive 14→34) but is excluded from the headline. Decision-relevant finding: llama-side negative precision 13/36 = 36.1%, and **the training-set hygiene filters remove 0 of those 21 mislabels** — 11 are analyst/rating rows in the `keep for now` class, the rest are tone-negative general news. So this is a labeling-prompt defect, not a filter gap, and it lands directly on the ~186 `negative` training rows. |
| 2026-08-07 | **Acceptance criteria are fixed before a labeling run, not after** (`quality_gate.CRITERIA`), and every directional criterion is **paired** with its opposite (precision with recall). Kappa is primary; raw agreement is not a criterion at all. | A single-sided criterion is gameable in the direction the fix pushes: told to raise `negative` precision, a labeler passes by ceasing to emit `negative`. Fixing the bar beforehand also stops the threshold from being renegotiated once the result is visible, which matters because we already retired one threshold for being unusable. |
| 2026-08-07 | **Gold rows are split dev / holdout** (`quality_gate.stratum_for`); prompts are tuned against dev only, and holdout rows are excluded from the FinBERT training set. Slice 6 is halved by file order rather than assigned whole. | Otherwise the human rows scatter into train/val by publication date and no human-truth evaluation survives — measured, 168 of 300 went to train. Slice 6 is the directional oversample, so assigning it whole to dev left the holdout with 3 `negative` rows; halving by file order keeps both sides usable and cannot be re-cut favourably later. The holdout is **semi-blind**: the gate report lists every disagreeing row. |
| 2026-08-07 | **Relabel before training, and pilot the relabel on the 300 gold rows before running all 8,360.** | Training first cannot answer whether relabeling is needed: validation labels come from the same labeler, so a high `negative` F1 would mean the model reproduced the error faithfully, and a low one confounds label noise with class scarcity (186 rows) and FinBERT's tone prior. The pilot costs ~2 min of GPU and no human time, because the 300 human labels already exist and `quality_gate.py` scores the result immediately. |
| 2026-08-07 | **Prompt v3** (`evals/prompts/jiwon_llama_v3.md`) leads with a subject test, closes `negative`'s open-ended euphemism clause, and rebalances the examples to neutral 9 / negative 6 / positive 3. v2 is kept as the provenance of `labels_2026-07-22.csv`. | v2 opened with "news articles about US banks", which asserted the premise the labeler most often got wrong — only 9.4% of GDELT titles name their own bank, and 13 of 25 dev disagreements are bank-as-analyst rows. "US" also contradicted the labeling guide. The examples are the real lever: `max_tokens=4` with guided choice means step-by-step instructions cannot execute as reasoning, and v2's negative-heavy example set was itself pushing the negative prior. |
| 2026-08-14 | **Live collection moves to GKG too** (`pipeline/poll_gkg.py`, scheduled in `ingest.yml` with a service-account key). This supersedes the "live poller stays on the DOC API" clause of the backfill entry above, decided the same day once the DOC API turned out to be refusing the poller as well. | poll_gdelt had not completed a run since ~2026-08-01. Spacing was not the variable: 8 s, 25 s and 60 s all 429'd, because the limit is on cumulative volume. Moving live as well has a second benefit — the backtest and production now read the same corpus, which is the mismatch that mattered most, since the backtest is the evidence for the whole claim. Volume falls from ~1,467 raw rows/day to ~313, but *attributed* volume rises from 87 to 313: the 5x the DOC API added was articles that never counted for a bank anyway. Operationally this needed a personal-account GCP project — the university one expires credentials within hours, survivable by hand and fatal for a cron — and the billing project is now `BIGQUERY_PROJECT` rather than a constant. |
| 2026-08-14 | **Retraining was considered for the corpus change and rejected, on measurement.** The model keeps its DOC-API-trained weights while serving GKG rows. | The concern was calibration: the model learned on a corpus where ~90% of rows were not about their bank, and GKG is ~100% self-naming, so a neutral-leaning model might under-call. Tested instead of assumed, by splitting the 300 gold rows on whether the title names its own bank and scoring the champion labeler on each half. It does **better** on the GKG-like half — kappa 0.678 / macro-F1 0.846 against 0.486 / 0.633 — because the analyst-attribution cases that prompts v3 and v4 both stumbled on live almost entirely in the half the new corpus drops. Raw agreement points the other way (84.6% vs 95.9%) and is the usual trap: the non-self-naming half is 94.7% `neutral`, so it scores well by doing nothing. **What this does not fix**: the training corpus and the gold slices are still DOC API draws, so the quality gate no longer samples what we collect. It still measures the labeler, which is its job; it no longer measures the corpus. Adding GKG-drawn gold rows closes that, and is the same kind of work as Yusheng's slices 7-9. |
| 2026-08-14 | **The 2020-2024 backfill comes from GKG in BigQuery, not the DOC API** (`pipeline/backfill_gkg.py`; 407,032 rows loaded). ~~The live poller stays on the DOC API.~~ — superseded later the same day, see the entry above. | The DOC API rate-limits on cumulative volume rather than spacing — 8 s, 25 s and 60 s all 429'd identically, and its own error body tells high-traffic users to switch datasets. A five-year pull is ~3,000 requests, exactly what is being refused; the poller's ~104 per run is not. BigQuery does all five years in one query per year for 356 GiB, inside the 1 TB monthly free tier, reached through the `bq` CLI so nothing is added to requirements. Worth recording that the DOC API had been failing since ~2026-08-01 on its own, before any backfill work — nothing we did caused it, and it went unnoticed because nothing consumes live GDELT yet. |
| 2026-08-14 | **The attribution gate is now load-bearing for the backtest, not an improvement.** It must be applied before any backtest number is reported, and the backtest write-up must say it was. | The two corpora differ where it matters. GKG has no usable company field (V2Organizations returned zero hits for "Bank of America" on a day JPMorgan had 177), so backfill rows are matched on the title naming the bank — ~100% self-naming. The DOC API searched article bodies, so only 9.4% of live rows do. Ungated, the backtest would score a clean corpus while production scores one where 21 of 30 directional rows are attributed to the wrong bank (measured, see § Bank attribution) — **the backtest would read better than production for a reason that has nothing to do with the model**. Measured once the gate existed (`pipeline/attribution.py`, 2026-08-14): it attributes **99.4%** of the 401,061 GKG backfill rows and **6.5%** of live rows over a healthy collection fortnight. Note the direction reverses rather than the gap merely shrinking — ungated, live is 1,352 rows/day against the backfill's 220, so **live is 6.2x larger**; gated, live is 87 against 218, so **the backfill is 2.5x larger**. The 5x the gate removes from live is almost entirely articles that were never about the bank. The residual is the different matching paths and should be stated, not explained away. Note the live figure is lower than the 9.4% this document measured in July: that number matched raw `bank.aliases`, while the gate also applies the seed's `generic` exclusion (22 banks fall back to holding name only) — the gate is stricter than the estimate it replaced. `index/` is Ming's lane, but for the *fundamentals* axis — that role guide says to leave `scoring/` alone — so the sentiment-side aggregation this gate lives in is assigned to nobody. Someone has to own it before the backtest means anything. |
| 2026-08-09 | **The fine-tune works, and its evidence is macro-F1, not accuracy** (`evals/finbert_metrics_2026-08-09.json`; weights outside the repo in the private Kaggle dataset **`chloejiwon/finbert-ft-2026-08-09`**, which `item_score.model_version` names as `finbert-ft-2026-08-09`). On the 132-row human holdout, macro-F1 goes **0.410 → 0.656** against pretrained-only FinBERT, with every class improving. **Do not report accuracy**: 0.841 fine-tuned sits *below* the 0.848 an all-`neutral` labeler scores there, the same trap that retired the original gate threshold, and val is worse still (0.928 against a 0.952 baseline). | Pretrained FinBERT behaves exactly as the tone-prior risk predicted — `neutral` recall 0.446, i.e. it calls direction on everything that sounds financial — and the fine-tune fixes that (0.884) without collapsing the directional classes. **The `negative` figure stays unreportable**: F1 0.545 rests on 6 holdout rows, so 3 correct calls. One mild signal worth re-checking when more human rows exist: the model's `negative` precision is 0.273 against val (labeler-scored) but 0.600 against the human holdout, i.e. the negatives it adds beyond the labeler tend to be right — consistent with the 46x class weight, and far too small a sample to lean on. |
| 2026-08-09 | **Both stages are reproducible, measured rather than assumed.** Labeling: prompts v3 and v4 were each run twice over the 300 gold rows (pilot, then as part of the full 8,360-row pass) and disagreed on **0 of 300** both times — `temperature=0` with constrained decoding is exact. Training: the notebook was re-executed on identical data when the Kaggle version was saved, and all **49 fields of `metrics.json` matched to the last decimal**. | Worth stating because neither was guaranteed. GPU kernel non-determinism usually moves fine-tune metrics slightly; `TrainingArguments`' default `seed=42` evidently pins this configuration completely. The practical consequence is that the committed `metrics.json` describes exactly the weights that were published, rather than a sibling run of them — and the write-up can claim end-to-end reproducibility with a measurement behind it. |
| 2026-08-09 | **The champion is a per-class ensemble, not a single prompt**: `negative` comes from prompt v4, every other label from v3 (`labels_ensemble_full.csv`, `model_meta.prompt_version = "v3+v4-neg"`). It beats both parents on every headline measure — kappa 0.683 (v3 0.650, v4 0.601), macro-F1 0.828 (0.808, 0.766) — and is the first labeler to clear the `negative` recall bar (0.857 ≥ 0.85) at 0.923 precision. It still misses `positive` recall (0.676 against 0.70), so the deviation recorded below stands, narrowed. | Each prompt is better on a different axis and the axes are separable: v4's subject-test wording recovers negatives v3 drops, while v3 keeps the `positive` behaviour v4 destroyed (recall 0.676 vs 0.353). Checked for dev-set fitting, since the combination was chosen after seeing the gold rows: on the **holdout alone** the ensemble scores `negative` 0.833 precision / 0.833 recall against v3's 1.000 / 0.667, so the gain is not an artefact of the rows the choice was made on. Cost: every corpus must now be labeled twice, once per prompt. This is outside DESIGN's champion-challenger framing, which assumes one labeler wins; recorded here rather than bent to fit it. |
| 2026-08-09 | **The `negative` class stays scarce, and the FinBERT run ships with that limitation rather than working around it.** The training set holds **38** `negative` rows (v2 had 186, v3 30, v4 35). Labeling backfill data to raise that count was considered and **rejected**. | The 186 under v2 were an artefact of over-calling — precision 0.361, so most were wrong. Correct labeling of this corpus yields 45-63 corpus-wide negatives against roughly 234 implied by the human rate on the random gold rows (a point estimate on 7 human negatives, so the interval is wide; the shortfall is nonetheless clear). No prompt closes it: the 2026-04-13 onward window simply contains few genuine bank-distress stories. The GDELT 2020-2024 backfill would supply them, but training on it **destroys the leak-free separation the design currently gets for free** — the distress events the backtest scores are 2017-2024 while the training corpus starts 2026, so there is currently zero overlap. Recovering that separation would need event-level holdout windows keyed to Shu Han's distress table, which does not exist yet, and would make the training set depend on a blocking artefact in another lane. The team chose the simpler, honest option: accept the ceiling, state it, and let the backtest judge. **What this means concretely: the fine-tuned model will not learn `negative` from 38 examples, its `negative` F1 on the 132-row holdout (6 negatives) cannot measure it either, and neither number should be reported as evidence about distress detection.** |
| 2026-08-09 | ~~**Prompt v3 is the champion**~~ — superseded the same day by the ensemble row above; kept for the record because the reasoning below still explains why a labeler that missed the criteria was accepted at all. | (original entry follows) |
| 2026-08-09 | **Prompt v3 is the champion**, chosen over v2 and v4 on kappa (0.650 vs 0.564 / 0.601) and macro-F1 (0.808 vs 0.700 / 0.766). Prompt iteration stops here, at the pre-agreed v4 cap. **v3 does not meet the acceptance criteria** — `negative` recall 0.714 (bar 0.85) and `positive` recall 0.676 (bar 0.70) — so this is a documented deviation, not a pass, and the write-up must say so. | v3 fixed what the gate was opened for: `negative` precision 0.361 → 1.000, and the 21 analyst-attribution mislabels went to **zero**. That is the defect that poisons the training set, and no later run beat it. v4 traded it away — it lifted `negative` recall 0.714 → 0.786 but drove `positive` recall 0.765 → 0.353 and moved the whole distribution toward the all-`neutral` degenerate labeler (274 predicted `neutral` against 252 actual), which is why its raw agreement rose to 95.2% while kappa *fell*. The remaining options are the Gemini challenger or two-stage decoding, both deferred; re-open this if either is taken up. |
| 2026-08-09 | **`positive recall` added to the criteria table** (`quality_gate.CRITERIA`), after the v4 run exposed that it was missing. | Every directional precision was supposed to be paired with its recall; `positive` shipped unpaired, and v4 walked through the gap — passing `positive precision` at 0.857 precisely *because* it had almost stopped predicting `positive` (43 → 14 predictions, recall 0.353). Four of five criteria read green while the failure migrated from one class to another. The threshold (0.70) is rounded down from the v2 baseline that set the rest of the table, not fitted to any run: the champion is below it. A test now asserts that no directional precision ships without its recall. Changing the table after seeing a result is exactly what the fixed-in-advance rule forbids, so it is recorded here as closing a hole rather than moving a bar — no existing threshold was touched. |
| 2026-08-07 | **The prompt file is the only source of its own version**; `kaggle_llama_labeling.py` reads the `<!-- prompt_version -->` marker and validates it before the GPU work. | The driver previously carried a `PROMPT_VERSION` constant beside a marker in the prompt file, so pointing it at a v3 file would have written `"v2"` into `model_meta` — a provenance field that lies is worse than no field. |

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
- ~~Quality gate: ≥85% agreement overall, no class below 75%.~~ **Retired
  2026-08-07 — do not use.** Measured on the 250 random gold rows, 91.6% of
  human labels are `neutral`, so a labeler that answers `neutral` to every row
  scores 91.6% and passes a threshold Llama's real 88.4% also passes. Raw
  agreement cannot separate a working labeler from a broken one here.
- **Read the gate chance-corrected instead.** `pipeline/quality_gate.py`
  reports Cohen's kappa, macro-F1, and the majority baseline alongside raw
  agreement, plus per-class figures in **both** directions — the old wording
  said "no class below 75%" without saying below 75% *of what*, and the two
  denominators disagree sharply (v2 `negative`: 92.9% human-side, 36.1%
  llama-side).
- **Acceptance criteria live in `quality_gate.CRITERIA`**, fixed before a run
  so they cannot be renegotiated after seeing its result. Every directional
  criterion is **paired**: precision alone is passed by a labeler that stops
  emitting the class, recall alone by one that emits it everywhere. Kappa is
  primary. Current bar: kappa ≥0.60, `negative` precision ≥0.60 *with* recall
  ≥0.85, `positive` precision ≥0.60, `neutral` recall ≥0.82.
- **Gold rows are split dev / holdout** (`quality_gate.stratum_for`, fixed
  2026-08-07). A prompt may be tuned against `dev` only; `holdout` is read
  once and is excluded from the FinBERT training set. Slice 6 is halved by
  file order rather than assigned whole — it is the directional oversample,
  and giving it entirely to dev left the holdout with 3 negative rows. The
  holdout is **semi-blind, not blind**: the gate report lists every
  disagreeing row, holdout included. Say so in the write-up.
- If the gate fails: revise prompt → relabel → if it still fails, activate the
  **Gemini challenger**: label the same corpus with `label_source='gemini'`,
  hand-review focuses on Llama/Gemini disagreement rows (`GROUP BY
  raw_item_id HAVING count(DISTINCT label) > 1`), and the team picks the
  champion on measured agreement with human labels.
- **Relabeling is piloted on the 300 gold rows first** (≈2 min of GPU), and
  the full 8,360-row run happens only if the pilot clears the criteria. The
  human labels already exist, so the pilot costs no human time and answers
  the only question the full run would.

### The finding that gates Stage 2 (measured 2026-08-07)

The headline number is not the result. **When Llama says `negative`, it is
right 13 times out of 36 — 36.1%.** Humans read two thirds of those rows as
`neutral`. That single figure decides whether Stage 2 can start, because
directional labels are the entire training signal and `neutral` accuracy is
free on a corpus that is ~90% `neutral`.

Three things make it a decision rather than an observation:

1. **It is structural, not sampling noise.** On the 250 random rows the figure
   was 6/16 = 37.5%. Slice 6 more than doubled the directional sample, to
   13/36, and it barely moved — 36.1%. A 16-row reading invites "wait for more
   data"; the same value at 36 rows does not.
2. **The training-set filters do not touch it.** Of the 21 rows where Llama
   said `negative` and a human said `neutral`, the hygiene filters remove
   **zero**. Eleven are analyst/rating rows — the bank is the *analyst* and
   another company is the subject ("CAE cut to Underweight at Morgan Stanley")
   — which is exactly the `keep for now` class in the table below. The rest
   are ordinary tone-negative news ("IBM stock sinks 22%", a branch robbery).
   So this is a **labeling-prompt defect, not a filter gap**, and no amount of
   filter work fixes it.
3. **It lands where the data is thinnest.** The assembled training set holds
   186 `negative` rows. If roughly two thirds carry the wrong direction, the
   real `negative` signal is on the order of 60 rows, and the fine-tune learns
   the analyst-attribution error as a rule.

This is the "systematic bias is not tolerable" case named two bullets up, and
it is why prompt v3 (`evals/prompts/jiwon_llama_v3.md`) leads with the subject
test. Note also that training on the current labels cannot *diagnose* this:
the validation labels are the same Llama labels, so a high `negative` F1 there
would mean the model faithfully reproduced the error.

**What the human side is, exactly.** Slices 1–5 were labeled once and then
*reviewed* by a second person; slice 6 was labeled fresh. No blind independent
second pass exists, so the project has **no inter-annotator agreement
statistic and cannot derive one from this** — a reviewer who sees the first
answer mostly agrees with it. The labels are better than a single pass, but
the ceiling these kappa numbers sit under is unknown. State this as a
methodology limitation rather than implying an IAA number exists.

## Stage 2 — Training (fine-tune FinBERT)

Pretrained-only FinBERT is not sufficient per the mentor; fine-tune on the
stage-1 labels. Same Kaggle GPU environment as labeling (workflow reuse).

Assembled by `pipeline/export_training_set.py` (local, pure and tested), then
fine-tuned by `pipeline/kaggle_finbert_train.py` (GPU glue only).

- Training set: `item_label` rows from the champion labeler, with `human`
  rows overriding the champion's where both exist.
- Split: **time-based** train/validation rather than random — matches serving
  reality and avoids syndication near-duplicates leaking across the split.
  Watch per-bank concentration (a few banks dominate GDELT volume).
  Measured in **days, not "last N weeks"**: GDELT polling went live
  2026-07-09, so 94% of the 2026-07-22 batch falls in its final 13 days and a
  two-week holdout put 82% of the corpus in validation. Default 3 days.
- **Three outputs, not two.** The human holdout stratum leaves train *and*
  val and is written as its own CSV. Without it the gold rows scatter across
  the split by publication date (measured: 168 into train, 48 into val, 84
  dropped by hygiene), so more than half the human ground truth is spent on
  training and val is ~30:1 Llama-labeled. **Metrics on val measure agreement
  with the labeler, not accuracy** — and the labeler is known to be wrong two
  times in three on `negative`. Only the holdout answers "is it right".
- Report accuracy + per-class F1 on **both** val and the human holdout,
  labeled so the two are not confused; compare against pretrained-only
  FinBERT as the baseline to demonstrate the fine-tune helped.
- Class weights are inverse-frequency: the corpus is ~90% `neutral` while the
  distress index depends on the rare directional classes.

### Training on a champion that missed the criteria (decided 2026-08-09)

The champion is the v3+v4 ensemble and is **below one of the acceptance
criteria** — `positive` recall 0.676 against 0.70. (The single-prompt v3 that
this section was first written for missed two, `negative` recall as well; the
ensemble closed that one at 0.857.) Training proceeds anyway. The reasoning,
because a reader is entitled to ask why a bar was set and then not met:

**Which criterion failed matters more than how many.** v2 failed on
*precision*: it labeled analyst rows about other companies `negative`, so
training on it teaches the rule "an analyst downgrade is bank distress", and
that rule then fires across the whole corpus and manufactures signal. v3 fails
on *recall*: the rows it misses land in `neutral`. That withholds signal
rather than inventing it — the "random noise is tolerable, systematic bias is
not" distinction this document draws at the gate. v3 fixed the bias: the 21
analyst-attribution mislabels went to zero.

**And one of the two is recoverable downstream.** `item_score.probs` stores
the 3-class distribution, so a model that under-calls direction can have its
directional threshold lowered at serving — trading precision back for recall
without retraining. A precision defect could not be undone that way: the wrong
rule is in the weights. The choice was therefore between a fixable failure and
an unfixable one.

Three things follow, and none are optional:

1. **The recall ceiling is inherited, not incidental.** FinBERT distills the
   stage-1 labels, so it cannot exceed v3 on the rows v3 gets wrong. The index
   will under-react to real distress. Say this in the write-up; do not report
   the model's recall as if the labeler were ground truth.
2. **Serving must not hard-code `argmax`.** Keep `probs` populated and leave
   the directional threshold tunable — that is the mitigation this decision
   depends on.
3. **The backtest is the real gate.** If `evals/backtest.py` shows the index
   missing known distress events, the fix is upstream at the labeler, and the
   remaining options are the Gemini challenger or two-stage decoding.

The misses are not uniformly random either — they cluster on "the bank is the
target of an analyst action", multi-bank sector stories, and results that beat
while the shares fell. Threshold tuning recovers such rows only partly, which
is why (3) is a real gate and not a formality.

**A fourth constraint, added when the corpus-wide counts came in.** Gold-set
recall does not transfer to the corpus. The ensemble reads 0.857 `negative`
recall on 14 human negatives, but labels only 63 of roughly 234 implied
negatives corpus-wide, because half the gold negatives come from slice 6 —
which sampled rows v2 already called `negative` *and* that name their own bank,
i.e. the easy ones. **Quote the gate's per-class recall as a property of the
gold rows, never as a corpus-wide rate.** The training set ends up with 38
`negative` rows, and the decision log entry above records why that is accepted
rather than fixed.
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

Counts below are as **implemented and re-measured 2026-08-07** by
`export_training_set.py` against the 2026-07-22 batch:

| Class | Rows | Directional labels | Action |
|---|---|---|---|
| Contentless EDGAR (10-Q / 10-K/A, no excerpt) | 107 | 0 | exclude — degenerate, all `neutral` |
| Holdings/13F wire spam ("X Purchases N Shares of Y") | **983** | 43 | exclude — directions are wrong-entity |
| Rating / price-target templates | 571 | 157 | **keep for now** — see below |
| Duplicate titles (cross-source, case-insensitive) | 385 | — | exclude — leakage across the split |
| Human holdout stratum | 132 | 20 | exclude from train/val — see Stage 2 |

The spam row read 723/51 in the original estimate. That predicate was never
kept, and this section's own instruction was to "settle in the export
script"; the settled predicate is structural — it fires only on
two-entity templates ("X purchases N shares of Y", "Y shares sold by X",
"X invests $N in Y"), and its overlap with the rating class was measured at
**zero**. The rating class is correspondingly larger than the earlier 375.

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

⚠️ **The backtest depends on this check, as of the 2026-08-14 backfill.** The
historical rows come from GKG matched on the title naming the bank, so they
are ~100% self-naming; live rows are 9.4%. Skip the gate and the backtest
scores a corpus that production never sees — flatteringly, since the 90% it
drops is where the misattribution lives. Apply it to both, and say in the
write-up that it was applied.

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
- [ ] **Quality-gate thresholds — the team still has to confirm these, but the
      old placeholders are retired, not pending.** Confirm the chance-corrected
      bar in `quality_gate.CRITERIA` (kappa ≥0.60 and the paired per-class
      figures), not "≥85% overall / ≥75% per class", which a do-nothing labeler
      passes.
- [x] Exact syndication-noise predicate for target selection — settled in
      `export_training_set.HOLDINGS_SPAM` (two-entity templates only, 983 rows,
      zero overlap with the rating class). Still to move into `eligibility`
      before Stage 3 serving ships.
- [ ] Whether the rating / price-target class (571 rows, 157 directional) can
      be split by subject-vs-agent instead of kept wholesale — 11 of the 21
      `negative` mislabels live here
- [ ] Model-weights hosting (Kaggle dataset vs HF hub)
- [ ] `probs` as jsonb vs three numeric columns (jsonb chosen for now; revisit
      if the dashboard needs to sort/filter on probabilities directly)
- [ ] Whether the inference-time LLM escalation tier (low-confidence items →
      LLM) is pursued — `llm_status`/`llm_attempts` kept unused for it
