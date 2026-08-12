# Design — quality gate computation + FinBERT fine-tune prep (2026-08-07)

Owner: Jiwon. Spec sources: `docs/roles/jiwon.md` §2/§4, `scoring/DESIGN.md`
(Hand verification & quality gate, Stage 2, Training-set hygiene).

## Scope

1. Compute the Stage-1 quality gate over gold slices 1–6 and write a report.
2. Assemble the FinBERT training set locally and finish the Kaggle GPU
   training script. The GPU run itself happens later, by hand, on Kaggle.

## Quality gate

New: `pipeline/quality_gate.py` (pure `compute_gate()` + argparse CLI, same
shape as `export_labeling.py`) and `tests/test_quality_gate.py`.

- Inputs: `evals/items/gold_slice_*.csv` (`id,source,title,text_excerpt,label,comment`)
  joined on `id = raw_item_id` to `labels_2026-07-22.csv` (`label_source='llama_kaggle'`).
- Outputs per run: overall agreement, 3×3 confusion, per-class agreement in
  **both directions** (human-side recall and llama-side), strata by source,
  by slice, and by the tone≠direction subgroup (title-pattern heuristic from
  the `labeling_guide.md` trap tables; flagged rows are listed in the report
  for eyeball review — the heuristic is not a verdict).
- **Sampling caveat baked into the report**: slices 1–5 are the random sample
  (250 rows, 221 agree = 88.4%, ≥85% headline PASS). Slice 6 was deliberately
  stratified by `export_new_gold_slice.py` (20 neg / 20 pos / 10 neu on
  Llama's label, own-bank-in-title only), so it strengthens the per-class
  cells (negative n 7→14, positive 14→34) but must not be pooled into the
  headline number — naive 300-row pooling gives 84.3% and is biased by design.
- Verdict language: overall gate passes on the random sample; per-class
  thresholds stay **pending team confirmation** (placeholders ≥85%/≥75%,
  DESIGN.md "threshold TBD by team"). Key finding to surface: llama-side
  negative precision 13/36 = 36.1% — Llama over-calls negative.
- Report lands at `evals/gate_report_2026-08-07.md`; one decision-log line in
  `scoring/DESIGN.md`.

Also: the labeled `gold_slice_6.csv` moves into `evals/items/` (the copy on
PR #16 has blank labels; the labeled version supersedes it — expect a merge
conflict there and resolve toward the labeled file).

## FinBERT fine-tune prep

New: `pipeline/export_training_set.py` (+ `tests/test_export_training_set.py`)
and `pipeline/kaggle_finbert_train.py`.

**Local assembly** (`export_training_set.py`, pure `build_training_set()`):
join labels+batch CSVs on `raw_item_id`; text via `pipeline.eligibility.text_for`
(the single source of truth); apply DESIGN.md training-set hygiene — drop
contentless EDGAR (expected 107) and 13F/holdings wire-spam titles (expected
723), keep rating/price-target templates (375); cross-source title dedup as in
`export_labeling.py`; human gold labels (300 rows) override Llama's;
time-based split on `published_at` with the last `--holdout-weeks` (default 2)
as validation. Funnel counts print and must reconcile against the DESIGN.md
table (107/723/375) — mismatches get reported, not silently absorbed.

**Kaggle script** (`kaggle_finbert_train.py`): GPU glue only, mirroring
`kaggle_llama_labeling.py` — run procedure in the docstring, argparse
`--train --val --output-dir --run-date [--epochs --lr]`. Evaluates
pretrained-only `ProsusAI/finbert` on the holdout as the baseline (id2label
order calibrated against `pipeline.labeling.LABELS`), fine-tunes with HF
Trainer using inverse-frequency class weights (corpus is 89.8% neutral),
re-evaluates (accuracy + per-class F1), saves weights + `metrics.json` to
`--output-dir`. Weights are versioned outside the repo (Kaggle dataset);
torch/transformers stay out of `requirements.txt` (same precedent as vllm).

## Ordering vs the gate

DESIGN.md forbids training before the gate resolves. This work stops at code +
report; the per-class verdict is explicitly deferred to the team's threshold
decision, and the GPU run happens after that — no conflict.

## Verification

- `pytest` (all existing + 2 new test files, including `test_structure.py`)
- `ruff check .`
- Gate report numbers reconcile: 221/250 (slices 1–5), 253/300 pooled,
  human-side class totals 252/34/14.
- Training-set funnel reconciles with DESIGN.md's 107/723/375 table.
