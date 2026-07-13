# evals — evaluation harness (Phase 6, not yet implemented)

Owns: evaluation of the scoring pipeline against a gold set, and the labeling
model comparison.

## Two kinds of labels — don't confuse them

- **Sentiment labels** (positive / negative / neutral per article): generated
  by LLM-assisted labeling in `scoring/`, from article text alone. The gold
  set here is the **human-verified sample** of those labels (mentor's sample
  verification) — it judges whether the labeler/model reads text correctly.
- **Distress labels** (`distress_within_4q/8q` in the fundamentals dataset):
  historical facts from FDIC failure records — nothing to label. They are the
  ground truth for the **backtest**, which judges whether the combined index
  actually warned ahead of real distress.

Automated labeling applies to the first kind only.

- `items/` — gold set CSVs (human-verified labeled raw items)
- `prompts/` — bake-off entries, one file per person: labeling prompts and the
  champion-challenger comparison between labeler models (Qwen / Llama / Gemini)
- Phase: 6
- Reads: `raw_item` and score tables (schema in `db/migrations/` is the only shared contract)
- Writes: eval reports (artifacts, not DB)
- Owner: TBD

Nothing in this directory is executable yet; teammates own all design decisions here.
