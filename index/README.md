# index — stability index library + recompute CLI (Phase 3, not yet implemented)

Owns: the bank stability index computation — a library plus a recompute CLI that
combines scored text signals with fundamentals into a per-bank index time series.
Parameters live in `index/config/` as versioned YAML (approved design decision).

- Phase: 3
- Reads: scored `raw_item` output, fundamentals tables
- Writes: index tables to be added under `db/migrations/` (the only shared contract)
- Owner: TBD

Nothing in this directory is executable yet; teammates own all design decisions here.
