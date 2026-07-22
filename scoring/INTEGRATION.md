# INTEGRATION — plugging into the scoring pipeline, step by step

**Status: skeleton (framework not yet implemented).** This is the scoring
counterpart of `DATA_SOURCES.md`: Jiwon owns the framework (export/import
scripts, eligibility module, serving batch, quality-gate harness — see
[DESIGN.md](DESIGN.md)); everyone else plugs in through one of the four
patterns below. Script names marked *(TBD)* are fixed when the framework
lands; the **contracts are fixed now** (migration `011`, DESIGN.md).

## Step 0 — pick your pattern (decision tree)

1. **You have a model or prompt that classifies articles?**
   (Gemini challenger, a Qwen bake-off entry, a revised Llama prompt)
   → **Pattern A: add a labeler.** Same one-line ALTER rhythm as adding a
   `raw_item` source in Phase 1.
2. **You integrated a `raw_item` source and want its items scored?**
   → **Pattern B: source adapter.** Tell the shared eligibility module which
   fields are text and what to skip. Without this your source sits at
   `finbert_status='pending'` forever — ingestion alone does NOT opt a
   source into scoring.
3. **You want to read sentiment downstream?** (backtest, stability index,
   dashboard) → **Pattern C: consumer.** Read-only on `item_score`.
4. **You were asked to hand-verify labels?** → **Pattern D: blind labels.**
   No code, ~1–2 hours, but it is the quality gate's ground truth.

## Contract recap (details in DESIGN.md — this is the short version)

- `item_label`: one row per (item, labeler); `label_source` CHECK lists the
  allowed labelers; `UNIQUE (raw_item_id, label_source)`.
- `item_score`: one row per item, latest score only. **Only the serving
  batch writes it.**
- `raw_item.finbert_status`: `'pending' → 'done' | 'failed' | 'skipped'`.
  **Only the serving batch changes it.**
- Labeling CSV round-trip: export `labeling_batch_<date>.csv`
  (`raw_item_id, source, bank_id, published_at, title, text_excerpt`) →
  labeler produces `labels_<date>.csv` (`raw_item_id, label, model_meta`).

## Walkthrough A — add a labeler (example: Qwen on Kaggle)

1. **Migration** — next free `db/migrations/` number: copy the one-line
   ALTER pattern from the header of `011_scoring_tables.sql` and add
   `'qwen_kaggle'` to the `label_source` CHECK. (`'llama_kaggle'`,
   `'gemini'`, `'human'` are already in 011 — no migration needed for those.)
2. **Prompt entry** — `evals/prompts/<yourname>_<model>.md`, one file per
   person (README contract). Prompt version goes in the file, not the DB.
3. **Label** — consume the export CSV as-is; emit exactly one of
   `positive|negative|neutral` per row into `labels_<date>.csv`, with
   `model_meta` JSON carrying model id, quantization, prompt version, run
   date. The canonical Llama notebook *(TBD)* is your copy template — same
   role `poll_gdelt.py` played in Phase 1.
4. **Import** — run the import script *(TBD)* with
   `--label-source qwen_kaggle`. `ON CONFLICT (raw_item_id, label_source)`
   means relabeling overwrites *your* row and touches nobody else's.
5. **Verify** —
   `SELECT label, count(*) FROM item_label WHERE label_source='qwen_kaggle' GROUP BY 1;`
   and the disagreement view vs the champion:
   `SELECT raw_item_id FROM item_label GROUP BY 1 HAVING count(DISTINCT label) > 1;`

## Walkthrough B — make your source scoreable (example: alpha_vantage)

1. **Adapter entry** — add your source to the shared eligibility module
   *(TBD: one dict/function per source)*: which columns form the text
   (`title` only? `title + text_excerpt`?), and any source-specific skip
   rule (e.g. Alpha Vantage ships its own sentiment in `meta` — it is
   ignored, never blended into ours).
2. **Do NOT re-implement shared filters.** English-only and the
   syndication-noise predicate live in the eligibility module and are used
   by both labeling export and serving — one implementation, or the
   training and serving distributions silently diverge. New rule → add it
   there, with a test.
3. **Verify** — export dry-run *(TBD flag)* prints eligible/skipped counts
   per source; after a serving run, every item of your source has left
   `'pending'` (→ `'done'` or `'skipped'` + reason in `last_error`).

## Walkthrough C — consume scores (backtest, index, dashboard)

- Read-only. Join to banks via `raw_item.bank_id`; to fundamentals via
  `bank.fdic_cert ↔ fact_bank_quarter.fdic_cert_number` (join by value, no FK).
- An item absent from `item_score` is normal (`'skipped'`/`'pending'`) —
  outer-join, don't assume coverage.
- Filter or group by `model_version` when comparing across time; scores made
  by different weights are not directly comparable.
- Never write `item_score` or `item_label` from a consumer.

## Walkthrough D — blind human verification (everyone, ~100–150 items each)

1. The gate owner sends you a sample CSV (stratified by class and source).
   It contains **no model labels** — that's the "blind" part; don't look
   them up.
2. Label from the article text alone — never from what later happened to
   the bank (mentor's instruction; that comparison belongs to the backtest).
3. Return the CSV; it is imported as `label_source='human'`. Human rows
   override the champion labeler's in the training set, so sloppy human
   labels do direct damage — when genuinely torn, `neutral` beats guessing.

## Definition of done

Pattern A: - [ ] CHECK migration (next number, existing files untouched)
           - [ ] prompt file in `evals/prompts/`
           - [ ] labels imported; per-class counts look sane
           - [ ] disagreement-vs-champion query run and eyeballed
Pattern B: - [ ] adapter entry + test in the eligibility module
           - [ ] export dry-run counts reviewed
           - [ ] post-run: zero stuck `'pending'` for your source
Pattern C: - [ ] reads only; outer-joins; `model_version` handled
Pattern D: - [ ] blind, text-only, returned on time

## Rules that apply to everyone

- **Single-writer rule**: serving writes `item_score` + `finbert_status`;
  import scripts write `item_label`; nothing else writes either table.
- **Eligibility logic exists once.** If you find yourself copying the
  language or noise filter into your own script, stop — that's the
  train/serve skew bug factory.
- **Labels never peek at outcomes.** Article text in, sentiment out.
- Repo working agreements (`CLAUDE.md`) apply: small gated steps, no new
  dependencies/tables/enum values outside the plan without asking.
- Questions → DESIGN.md first, then the `011` header comments, then the
  team channel — before inventing a fifth pattern.

## For coding agents (and the humans driving them)

Read, in order: repo root `CLAUDE.md`, `scoring/DESIGN.md`, this file,
`db/migrations/011_scoring_tables.sql`. The agent must NOT redesign the
framework: if the task seems to require editing the export/import scripts,
the eligibility module's interface, or an applied migration, the design is
being misread — stop and ask. Pattern A and B changes should each fit in
one small PR (one migration + one file + one test).
