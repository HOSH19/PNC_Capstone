# RUNBOOK — ingestion slice (GDELT + EDGAR)

## 1. Apply migrations to Supabase

Get the **session pooler** connection string from Supabase dashboard →
Project Settings → Database → Connection string, then run the migrations
in order (they are not idempotent — run each once):

```bash
export SUPABASE_DB_URL='postgresql://postgres.<project-ref>:<password>@aws-0-<region>.pooler.supabase.com:5432/postgres'

psql "$SUPABASE_DB_URL" -f db/migrations/001_bank.sql
psql "$SUPABASE_DB_URL" -f db/migrations/002_raw_item.sql
psql "$SUPABASE_DB_URL" -f db/migrations/003_watermark_heartbeat.sql
psql "$SUPABASE_DB_URL" -f db/migrations/004_fundamentals.sql
```

After 004, load Shu Han's fundamentals CSVs (stateless full loader — safe to
re-run any time the CSVs are rebuilt):

```bash
python -m pipeline.loaders.load_fundamentals
```

## 2. Set GitHub secrets

Repo → Settings → Secrets and variables → Actions → New repository secret:

| Secret | Value |
|---|---|
| `SUPABASE_DB_URL` | the pooler DSN above |
| `SEC_USER_AGENT_EMAIL` | a real contact email (SEC blocks requests without one) |

## 3. Trigger the workflow

GitHub → Actions → **ingest** → Run workflow (branch `main`).
The `seed` job upserts `db/seed/banks.csv` into the `bank` table, then one
`poll` matrix job per source runs in parallel (a failing source never blocks
the others). Each poller logs `<bank>: N seen, M inserted`, and a failing
bank is logged as `<bank>: FAILED: <cause>` without stopping the other banks.

**Failure notifications**: GitHub emails on failed workflow runs — for
scheduled runs the recipient is the last committer of the workflow file, and
every teammate can opt in via GitHub → Settings → Notifications → Actions →
"Only notify for failed workflows". No webhook is configured (team decision:
email for now).

Local alternative (needs `.env` values exported, see `.env.example`):

```bash
pip install -r requirements.txt
python -m pipeline.seed_banks
python -m pipeline.poll_gdelt
python -m pipeline.poll_edgar
```

## 4. Verify rows

```sql
-- items per source/bank
SELECT source, bank_id, count(*), max(published_at)
FROM raw_item GROUP BY 1, 2 ORDER BY 1, 2;

-- EDGAR forms breakdown
SELECT bank_id, meta->>'form' AS form, count(*)
FROM raw_item WHERE source = 'edgar' GROUP BY 1, 2 ORDER BY 1, 2;

-- poller state and run history
SELECT * FROM watermark ORDER BY source, bank_id;
SELECT * FROM pipeline_heartbeat ORDER BY run_at DESC LIMIT 10;
```

**Idempotency check**: run the workflow a second time immediately — the new
`pipeline_heartbeat` rows should show `items_inserted` at or near 0 (the
overlap windows — 15 minutes for GDELT, 2 days for EDGAR — may legitimately
re-see a handful of items; the UNIQUE constraint drops them).

## 5. Add a bank (no code changes)

1. Add one row to `db/seed/banks.csv` (aliases are `;`-separated; `cik` is
   the zero-padded 10-digit string from
   https://www.sec.gov/files/company_tickers.json).
2. Set `is_live` to `true`.
3. Re-run the workflow — the seed step upserts the row and both pollers
   pick it up (first run for a new bank looks back 72 h on GDELT, 90 days
   on EDGAR).

## 6. Add a new data source (for teammates)

Shared infrastructure (`pipeline/db.py`, `watermark`, `pipeline_heartbeat`,
the `UNIQUE(source, external_id, bank_id)` key) is source-agnostic — you only
touch the four places below.

**Start with `DATA_SOURCES.md`** — it maps every source in the team inventory
(enforcement actions, news APIs, FRED, yfinance, CFPB, RSS, …) to the right
pattern, with the `source` value, `external_id`, and bank-matching strategy to
use for each, plus full walkthroughs.

**Incremental poller** (source won't replay the past — e.g. RSS):

1. Migration `db/migrations/00X_add_<source>.sql`: extend the `raw_item.source`
   CHECK (exact ALTER statements are in the header of `002_raw_item.sql`).
2. `pipeline/poll_<source>.py`: copy the `main()` skeleton from
   `pipeline/poll_gdelt.py` (the canonical template) and swap the
   fetch/transform parts. Use `pipeline.http.throttled_get` for all requests
   (headers/retry_statuses are parameters — don't fork the retry loop).
   Keep: watermark + overlap window, per-bank failure containment,
   pre-insert dedup where the source syndicates, heartbeat on both paths.
3. `.github/workflows/ingest.yml`: add the module name to the `poll` job's
   matrix (one line, marked spot). Job-level `needs: seed` +
   `fail-fast: false` isolate failures for you — no per-step guards needed.
4. Only if the source needs its own per-bank identifier: add a `bank` column
   via migration + mirror it in `db/seed/banks.csv` and
   `pipeline/seed_banks.py` COLUMNS (see header of `001_bank.sql`).

**Full loader** (source returns full history — FRED, yfinance, fundamentals):
lives in `pipeline/loaders/`, is stateless (no watermark), re-fetches
everything and upserts by natural PK into its OWN table added under
`db/migrations/` — it does not write `raw_item`. See
`pipeline/loaders/README.md`.

## Notes

- First-run lookbacks are code constants: GDELT 72 h
  (`pipeline/poll_gdelt.py`), EDGAR 90 days (`pipeline/poll_edgar.py`).
  Anything older is backfill territory (out of scope for this slice).
- The current 5 seed banks are marked **PENDING TEAM RE-SELECTION** —
  revisit `gdelt_query` strings when the team finalizes the bank list
  (e.g. "PNC Bank" also matches PNC-sponsored venues).
