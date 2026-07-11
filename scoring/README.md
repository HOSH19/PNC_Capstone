# scoring — FinBERT + Gemini hybrid scorer (Phase 2, not yet implemented)

Owns: sentiment/risk scoring of ingested text. Consumes `raw_item` rows via the
status columns (`finbert_status = 'pending'` → score → mark done; `llm_status`
gates escalation to the LLM scorer, `llm_attempts` / `last_error` track retries).

- Phase: 2
- Reads: `raw_item` (status columns are the queue)
- Writes: score tables/columns to be added under `db/migrations/` (the only shared contract)
- Owner: TBD

Nothing in this directory is executable yet; teammates own all design decisions here.
