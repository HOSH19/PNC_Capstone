# eda — exploratory data analysis (active now)

Owns: exploratory analysis of the collected data, requested by the mentor
(2026-07-12) ahead of any dashboard or model work. Notebooks and analysis
scripts live here; findings go out as email updates to the mentor.

Scope:
- descriptive statistics over `raw_item` (volume per bank/source, coverage)
- keyword extraction and PCA / clustering — which keyword groups drive
  positive vs negative framing
- input for the labeling-approach decision (corpus size → manual vs
  LLM-assisted labeling)

- Phase: now (precedes scoring/dashboard design)
- Reads: `raw_item`, fundamentals CSVs (read-only)
- Writes: notebooks/reports here — no DB writes, no shared code
- Owner: TBD
