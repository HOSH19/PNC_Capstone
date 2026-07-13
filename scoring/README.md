# scoring — sentiment scoring pipeline (Phase 2, not yet implemented)

Owns: sentiment scoring of ingested text, per the mentor-agreed methodology
(2026-07-12): **LLM-assisted labeling → fine-tuned BERT-based model → daily
batch prediction**, with keyword-level explainability.

## Pipeline (three stages)

1. **Labeling** — build the training corpus. An LLM/SLM (candidates: Qwen via
   Foundry Local, Llama via Kaggle free GPU, Gemini API) labels collected
   articles positive / negative / neutral; run a champion-challenger between
   labelers and hand-verify a sample. Labels are assigned **from the article
   text alone** — do not validate against subsequent real-world events
   (mentor's instruction; that comparison belongs to the backtest, which uses
   the separate distress labels from the fundamentals dataset).
2. **Training** — fine-tune a BERT-based model (FinBERT is the leading
   candidate) on those labels: train/validation split, tuning, accuracy
   evaluation. Pretrained-only FinBERT is not sufficient per the mentor.
3. **Serving** — the fine-tuned model scores new `raw_item` rows daily via the
   status-column queue (`finbert_status = 'pending'` → score → mark done).
   Output is 3-class plus **standout keywords** per sentiment cluster
   (PCA / keyword clustering) for explainability in the dashboard.

`llm_status` / `llm_attempts` / `last_error` remain available for either
labeling bookkeeping or an optional inference-time LLM escalation tier
(undecided — kept as an open option, no schema change needed).

- Phase: 2
- Reads: `raw_item` (status columns are the queue)
- Writes: label/score tables or columns to be added under `db/migrations/`
  (3-class label, label_source, keywords — the only shared contract)
- Owner: TBD

Nothing in this directory is executable yet; teammates own all design decisions here.
