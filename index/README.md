# index — stability index library + recompute CLI (Phase 3, not yet implemented)

Owns: the bank stability index computation — a library plus a recompute CLI that
combines scored text signals with fundamentals into a per-bank index time series.
Parameters live in `index/config/` as versioned YAML (approved design decision).

## Method (mentor-agreed, 2026-07-12)

- **Fundamentals axis**: sklearn **GaussianProcessClassifier** over Call Report
  features (liquidity ratio, fee income ratio, loans vs capital, NPL, …) with
  min/max threshold-based distress indicators derived from descriptive analysis.
- **Score bands**: ≥ 90 positive (sound) / 80–90 neutral / ≤ 80 negative
  (distress signal) — i.e. distress yes / no / neutral, three classes like the
  sentiment side.
- **Combination**: the GP fundamentals score is combined with the sentiment
  score from `scoring/` — this module is where the two axes meet (documents are
  aggregated to bank × period here before joining quarterly fundamentals).

- Phase: 3
- Reads: scored `raw_item` output, fundamentals tables
- Writes: index tables to be added under `db/migrations/` (the only shared contract)
- Owner: Ming (fundamentals axis) — see docs/roles/ming.md. The sentiment-side
  document aggregation, including the bank-attribution gate, is unassigned.

Nothing in this directory is executable yet; teammates own all design decisions here.
