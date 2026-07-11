# evals — evaluation harness (Phase 6, not yet implemented)

Owns: evaluation of the scoring pipeline against a hand-labeled gold set.

- `items/` — gold set CSVs (labeled raw items)
- `prompts/` — prompt bake-off entries, one file per person
- Phase: 6
- Reads: `raw_item` and score tables (schema in `db/migrations/` is the only shared contract)
- Writes: eval reports (artifacts, not DB)
- Owner: TBD

Nothing in this directory is executable yet; teammates own all design decisions here.
