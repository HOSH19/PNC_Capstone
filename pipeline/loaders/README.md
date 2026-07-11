# pipeline/loaders — full loaders (Phase 2+, not yet implemented)

Owns: stateless FULL LOADERS for sources that return complete history on every
fetch — FRED macro series, yfinance market data, FFIEC/FDIC fundamentals → DB.
Pattern: re-fetch everything, upsert by natural primary key (no watermark).

- Phase: 2+
- Reads: external APIs only
- Writes: dedicated tables to be added under `db/migrations/` (the only shared contract)
- Owner: TBD

Nothing in this directory is executable yet; teammates own all design decisions here.
