# Role guide — Jiwon: labeling + FinBERT

> Owns: Stage 1 (labeling / quality gate) and Stage 2 (fine-tuning) of
> `scoring/`, plus the historical GDELT backfill that gives the backtest
> something to score.

## Dependency graph

Slices 2–5 are all on `main` (200 rows). **`gold_slice_1` is the only thing
left before the gate**, and it is mine — no one else to wait on.

```mermaid
flowchart LR
    RITA["Rita · gold_slice_4 ✔"]
    YU["Yusheng · gold_slice_5 ✔"]
    J1["<b>1 · gold_slice_1</b><br/>sole remaining blocker"]
    J2["<b>2 · quality gate</b><br/>250 rows, tone≠direction"]
    J3["<b>3 · training-set hygiene</b><br/>+ eligibility predicates"]
    J4["<b>4 · FinBERT fine-tune</b>"]
    J5["<b>5 · GDELT backfill 2020–24</b><br/>25s spacing, chunked"]
    SCORE["backfill scored on CPU<br/>no labeling cost"]
    SH["Shu Han · evals/backtest.py"]
    SERVE(["Stage 3 serving<br/>deferred"])

    RITA --> J2
    YU --> J2
    J1 --> J2 --> J4
    J3 --> J4
    J4 --> SCORE
    J5 --> SCORE --> SH
    J3 -.->|must move into eligibility first| SERVE
    J4 -.->|weights| SERVE
```

Tasks 3 and 5 are **not** behind the gate — the predicates and the backfill can
run while slices 4 and 5 are still in review. Only tasks 2 and 4 wait.

## Sequence

Stage 2 depends on Stage 1 in a way that is easy to get wrong: the training set
is the **champion** labeler's rows, and the quality gate decides who the
champion is. If the gate fails, the fix is prompt revision → re-labeling, which
changes every label. Training before the gate concludes risks throwing that
training away.

### 1. Label `gold_slice_1` — 50 rows

0 of 50 filled. This is my share of the 5 × 50 human verification set.

### 2. Compute the quality gate

Once slices 4 (Rita) and 5 (Yusheng) land on `main`, all 250 rows exist.

- Per-class disagreement vs `llama_kaggle`
- Placeholder thresholds: ≥85% overall, ≥75% every class — **the team still has
  to confirm these**
- **Stratify by tone ≠ direction.** This is the point of the whole exercise.
  FinBERT is pretrained on financial *tone* while our target is *risk
  direction*, and they diverge exactly on the euphemism cases the labeling
  guide flags as most valuable — "exploring strategic alternatives" (calm tone,
  negative), "consent order lifted" (regulator language, positive). An overall
  pass rate can hide failure on precisely those rows.

If the gate fails: revise the prompt, re-label, and only then consider standing
up the Gemini challenger.

### 3. Training-set hygiene

Filters applied when **assembling the training set** — no `eligibility` change,
no re-export, no re-labeling:

| Class | Rows | Directional | Action |
|---|---|---|---|
| Contentless EDGAR (10-Q / 10-K/A, no excerpt) | 107 | 0 | exclude — degenerate, all `neutral` |
| Holdings / 13F wire spam | 723 | 51 | exclude — the direction belongs to another company |
| Rating / price-target templates | 375 | 123 | **keep for now** — same surface form covers both roles |

Also finish the two `eligibility` predicates, since they gate both this and
serving:

- `is_syndication_noise` — must fire only when a **second entity** is the
  object. A bare `Upgrad|Downgrad` term also catches a bank's *own* rating
  change ("Commerce Bancshares Downgraded by Wall Street Zen to Sell" — a
  legitimate `negative`, confirmed in a gold slice).
- EDGAR empty `text_excerpt` → `eligible=False, reason="empty_text"`. EDGAR
  titles are synthesized as `"{holding_name} {form}"` and are never empty, so
  the existing guard never fires.
- **Boilerplate-only 8-K excerpts are a third class**, found while labeling
  `gold_slice_1`: `"First Financial Bancorp. 8-K"` has a long, non-empty
  excerpt that is entirely the SEC cover page — address, phone number, the
  Rule 425 / 14a-12 checkboxes, the registered-securities table — with no
  event text. An emptiness check will not catch it. Count how many of these
  exist while labeling slice 1, then widen the predicate from "is the excerpt
  empty" to "is there body text past the cover page". Same degenerate pattern
  as the 107 contentless 10-Q rows: nothing to read, so every labeler writes
  `neutral`.

⚠️ Both predicates must move into `eligibility` **before Stage 3 serving
ships**. Filtering only at training time while serving keeps scoring those rows
creates exactly the train/serve skew the shared filter exists to prevent.

### 4. Fine-tune FinBERT

Kaggle GPU, same environment as the labeling batch.

- Training set: champion labeler's `item_label` rows, with `human` rows
  overriding where both exist
- **Time-based** train/validation split, not random — matches serving reality
  and stops syndicated near-duplicates leaking across the split
- Report accuracy + per-class F1 on the holdout, **against a pretrained-only
  FinBERT baseline**. Without that comparison there is no evidence fine-tuning
  did anything
- Decide weight hosting (Kaggle dataset vs HF hub) and record which weights
  scored each row in `item_score.model_version`

### 5. GDELT backfill, 2020–2024

Measured problem: our distress events run 2017–2024 and the most recent is
2024-12-31, while the GDELT corpus starts 2026-04-13. **Zero overlap** — the
backtest has no scorable history without this.

- 104 seed banks, no bank-set expansion (probe showed sub-$10B banks have
  essentially no news; Heartland Tri-State produced 5 English articles in the
  year it failed)
- 2020–2024 covers 24 of the 33 distress events
- ~140k rows expected
- **Request spacing must be ~25s, not the current 8s.** The sizing probe hit
  429s on 2 of 6 banks at 8s and succeeded immediately at 25s. At 25s the run
  is ~15 hours, so chunk it across Actions jobs — the per-job limit is 6h
- **These rows are not labeled.** The fine-tuned FinBERT scores them on CPU.
  Backfill volume does not touch the Kaggle labeling budget at all

## What I do not own

- `evals/backtest.py` and the metric protocol — Shu Han
- The distress event definition — Shu Han, with Ming
- Stage 3 serving harness — deferred until the weights exist; wiring it against
  pretrained weights now means writing it twice

## Reference

- `scoring/DESIGN.md` — stages, contracts, decision log; `DESIGN.ko.md` is the
  Korean translation and English is canonical
- `scoring/labeling_guide.md` — the rules the human labels follow
- `evals/prompts/jiwon_llama.md` — the labeling prompt (v2)
- `pipeline/eligibility.py` — the shared filter both stages call
