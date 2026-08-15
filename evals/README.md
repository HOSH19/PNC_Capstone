# evals — evaluation harness

Owns: (1) sentiment gold-set / labeling bake-off artifacts, and (2) the
**distress backtest** that grades risk scores against the fundamentals answer
key.

## Two kinds of labels — don't confuse them

- **Sentiment labels** (positive / negative / neutral per article): generated
  by LLM-assisted labeling in `scoring/`, from article text alone. The gold
  set here is the **human-verified sample** of those labels — it judges
  whether the labeler/model reads text correctly.
- **Distress labels** (`distress_within_4q` in
  [`items/distress_bank_quarter.csv`](items/distress_bank_quarter.csv)):
  rule-based acute stress from Call Reports (not FDIC failure). Definition:
  [`distress_definition.md`](distress_definition.md). They are the ground
  truth for the **backtest**, which judges whether risk scores warned ahead
  of real distress.

Automated labeling applies to the first kind only. Never mix the two.

## Layout

| Path | Role |
|---|---|
| `items/gold_slice_*.csv` | Human-verified sentiment gold slices |
| `items/distress_bank_quarter.csv` | Distress answer key (rebuild via `build_distress_labels.py`) |
| `distress_definition.md` | What counts as a distress event / lookahead |
| `backtest_protocol.md` | Metrics, time split, eval windows (locked) |
| `build_distress_labels.py` | Rebuild the distress CSV |
| `backtest.py` | Harness — `python3 evals/backtest.py --smoke` (naive −tier1) |
| `prompts/` | Labeling bake-off prompts |

- Reads: local distress CSV + score CSVs / Call Reports (and later score tables)
- Writes: eval reports under `evals/reports/` (artifacts, not DB)
- Owner: Shu Han (distress / backtest); labeling gold slices are shared
