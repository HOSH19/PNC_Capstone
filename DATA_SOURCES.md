# DATA_SOURCES — adding the inventory sources, step by step

This guide maps the **not-yet-integrated** sources from the team's Integrated
Public Data Source Inventory to the two integration patterns this repo
supports, and walks through adding one of each. The mechanical checklist
lives in `RUNBOOK.md` §6; this document tells you **which pattern your source
needs and what to use for each blank**.

## Step 0 — pick your pattern (decision tree)

1. **Does the source give you the full history every time you ask?**
   (FRED series, yfinance prices, CFPB complaint dumps)
   → **Full loader** in `pipeline/loaders/`: stateless, re-fetch everything,
   upsert by natural key into its **own table**. Never writes `raw_item`.
2. **Does the source stream items you'd miss if you stopped polling?**
   (news APIs, RSS feeds, enforcement announcements)
   → **Incremental poller** in `pipeline/`: watermark + overlap window,
   items land in `raw_item`.
3. **Is the data text that scoring should read?** → it belongs in `raw_item`
   (pattern 2, or a one-off loader that inserts into `raw_item`).
   **Is it numbers/labels?** → own table (pattern 1).
4. **Can't automate the fetch at all?** (library archives, manual exports)
   → collect files by hand, commit a CSV under `db/seed/` or `eda/`, and write
   a one-off loader. Don't force a poller onto a source you scrape manually.

## Source-by-source mapping

| Source (owner) | Pattern | `source` value / table | `external_id` / natural key | Bank matching | Notes & gotchas |
|---|---|---|---|---|---|
| OCC Enforcement Actions (Yusheng) | poller → `raw_item` | `occ_enforcement` | action number | charter number → needs new `bank` column, or name match | Export may be manual at first — CSV + one-off loader is fine to start; check for a structured download first (the Fed has one, see below) |
| Fed Enforcement Actions (Yusheng) | CSV fetch → `raw_item` | `fed_enforcement` | document URL from the CSV | holding name → `aliases` | **Easiest of the three**: the Fed publishes the full history as plain CSV at `federalreserve.gov/supervisionreg/files/enforcementactions.csv` (effective date, org, action type, doc URL) — no scraping. Fetch whole file, filter to tracked banks, upsert into `raw_item`; the UNIQUE key replaces the watermark |
| Earnings releases / transcripts (Yusheng) | poller → `raw_item` | `earnings` | document URL | per-bank IR page → new `bank.ir_url` column | Most fragmented source; start with 5–10 banks, not all 104 |
| NewsAPI (Rita) | poller → `raw_item` | `newsapi` | article URL | query per bank (reuse `aliases`) | Needs `NEWSAPI_KEY` secret; free tier has short history — watermark from day 1 |
| Alpha Vantage News (Rita) | poller → `raw_item` | `alphavantage` | article URL | `bank.ticker` (already in the table) | Needs `ALPHAVANTAGE_KEY`; its own sentiment score goes into `meta`, don't treat it as ours |
| Benzinga (Rita) | poller → `raw_item` | `benzinga` | article ID | ticker | Commercial — confirm access before writing any code |
| Reuters archive via UCLA Library (Ming) | manual corpus | n/a (decide later) | n/a | n/a | **Check license/export rights first**; likely an offline backtest corpus, not a pipeline source |
| FRED (Ming) | full loader | table `fred_observation` | `(series_id, date)` | none (macro controls) | Needs `FRED_API_KEY`; series list can be a YAML/CSV config |
| CFPB Complaints (Ming) | full loader | table `cfpb_complaint` | `complaint_id` | company-name mapping (messy — keep a mapping column or table) | Structured signal, not text for scoring |
| yfinance (Ming) | full loader | table `market_daily` | `(ticker, date)` | `bank.ticker` | Unofficial API — wrap fetches in try/except per ticker |
| Agency press releases RSS (Ming) | poller → `raw_item` | `agency_rss` | item URL/GUID | scan title/summary against `aliases`; skip items matching no tracked bank | The canonical RSS example from RUNBOOK §6 |

## Walkthrough A — incremental poller (example: NewsAPI)

1. **Migration** — `db/migrations/004_add_newsapi.sql` (next free number):
   copy the two ALTER statements from the header of
   `db/migrations/002_raw_item.sql` and add `'newsapi'` to the CHECK list.
   Apply it: `psql "$SUPABASE_DB_URL" -f db/migrations/004_add_newsapi.sql`.
2. **Poller** — copy `pipeline/poll_gdelt.py` → `pipeline/poll_newsapi.py`
   (it is the canonical template; its `main()` shape is the contract):
   - keep: the per-bank loop with its try/except containment, watermark ±
     overlap, `db.upsert_raw_items`, `db.set_watermark(run_start)`,
     heartbeat on both paths, the final `sys.exit` on partial failure;
   - replace: `fetch_window()` with your API call — **always through
     `pipeline.http.throttled_get`** (headers / retry_statuses / throttle_s
     are parameters; do not write your own retry loop);
   - build rows with `source="newsapi"`, `external_id=<article URL>`, and put
     source-specific extras in `meta` (jsonb — no schema change needed);
   - if the source syndicates (news does), keep the
     `db.existing_title_hashes` pre-insert dedup.
3. **Secret** — API keys are env vars, never code: add `NEWSAPI_KEY=` to
   `.env.example`, set it in GitHub → Settings → Secrets → Actions, and add
   one line under `env:` in `.github/workflows/ingest.yml`.
4. **Workflow** — add `poll_newsapi` to the `poll` job's matrix in
   `ingest.yml` (one line, marked spot). `needs: seed` + `fail-fast: false`
   give you failure isolation automatically.
5. **Verify** — run the workflow twice (RUNBOOK §3–4): first run inserts,
   second run's `pipeline_heartbeat.items_inserted` ≈ 0. Check
   `SELECT count(*) FROM raw_item WHERE source='newsapi';`.

Only if your source needs a per-bank identifier that isn't in the `bank`
table yet (OCC charter number, IR page URL): add the column in your
migration, mirror it in `db/seed/banks.csv` and the `COLUMNS` tuple of
`pipeline/seed_banks.py` (see the header of `db/migrations/001_bank.sql`),
and start your bank loop with `if not bank["your_column"]: continue`.

## Walkthrough B — full loader (example: FRED)

1. **Migration** — `db/migrations/00X_fred.sql`: create your own table with
   the natural key, e.g.
   `fred_observation(series_id text, date date, value numeric, PRIMARY KEY (series_id, date))`.
   Full loaders never touch `raw_item` and need no CHECK change.
2. **Loader** — `pipeline/loaders/load_fred.py`: stateless. Fetch every
   configured series in full (through `pipeline.http.throttled_get`), then
   `INSERT ... ON CONFLICT (series_id, date) DO UPDATE`. No watermark —
   re-running always converges to the source. Write a heartbeat
   (`db.write_heartbeat(conn, "load_fred", ...)`) so runs are observable.
3. **Secret** — same as poller step 3 (`FRED_API_KEY`).
4. **Workflow** — add `loaders.load_fred` as a matrix entry (module path
   works: `python -m pipeline.loaders.load_fred`).
5. **Verify** — run twice; row count stays identical the second time.

## Walkthrough C — sources you can't fully automate: the FDIC enforcement case

FDIC Enforcement Orders are **implemented** (`pipeline/poll_fdic_enforcement.py`,
migration `005`) and illustrate the pattern for portal-only sources. The ED&O
portal (`orders.fdic.gov`) is a Salesforce Lightning SPA with no export, no
API, and no stable server-rendered DOM (verified: plain GET and headless
Chrome both return the loading shell), so the fetch step is **manual by
design** and everything after it is automated:

- **Monthly routine (~10 min, owner: Jiwon)**: open the month's FDIC press
  release (fdic.gov/news) → follow its link to the portal's order list →
  append one CSV row per order to `db/seed/fdic_enforcement.csv`
  (`order_date,institution_name,city,state,order_type,docket_number,pdf_url`)
  → commit and push.
- **Every workflow run**: the poller re-reads the whole CSV, matches
  institutions to tracked banks by normalized legal/holding/alias names
  (orders for untracked community banks are skipped with a log line —
  expected), and upserts into `raw_item`; `UNIQUE(source, external_id,
  bank_id)` makes unchanged-CSV runs free no-ops, so it rides the normal
  schedule with no watermark.
- PDF text extraction is deferred (would need a `pypdf` dependency —
  needs approval); rows are metadata-only for now.

Reuse this shape for any source where the fetch can't be automated cheaply:
hand-maintained CSV in `db/seed/` as the interface, an idempotent poller
doing everything else. CAPTCHA is not the real obstacle at public-records
volumes (one page a month, no login) — markup churn and missing exports are.

**Do the Fed source first.** Its full-history CSV (see table above) makes it
the cheapest regulatory-action source by far — a good first contribution
before tackling OCC.

## For coding agents (and the humans driving them)

If you point a coding agent at this task, have it read, in order: the repo
root `CLAUDE.md` (working agreements — small gated steps, never touch
`unified_ffiec_fdic_dataset/`, no new dependencies/tables without asking),
then this file, then `RUNBOOK.md` §6, then the template it will copy
(`pipeline/poll_gdelt.py` for pollers, `pipeline/loaders/load_fundamentals.py`
for loaders). The agent must NOT redesign shared infrastructure — if the
task seems to require editing `pipeline/db.py`, `pipeline/http.py`, or an
already-applied migration, the design is being misread: stop and ask.

A `raw_item` row is a dict with exactly these keys (see `db.RAW_ITEM_COLUMNS`):
`source, external_id, bank_id, published_at (tz-aware datetime), title, url,
domain, text_excerpt, title_hash, n_duplicates, meta (dict)` — put anything
source-specific inside `meta`.

Definition of done for a new source (check every box before the PR):

- [ ] CHECK migration (or own-table DDL) added as the next `db/migrations/`
      number; existing migration files untouched
- [ ] module runs locally: first run inserts, immediate second run inserts ~0
      (RUNBOOK §3–4)
- [ ] one bank/row failing does not stop the others (poller pattern)
- [ ] all HTTP goes through `pipeline.http.throttled_get`
- [ ] new secrets in `.env.example` + GitHub secrets + `ingest.yml env:`
- [ ] one line added to the `ingest.yml` matrix
- [ ] this file updated: your source's row removed from the table above

## Rules that apply to every source

- **Never fork shared infrastructure**: `pipeline/db.py` and
  `pipeline/http.py` take `source`/`headers`/`retry_statuses` as parameters —
  adding a source requires **zero changes** in them.
- **One bank's failure must not stop the others** — keep the per-bank
  try/except from the template. A failed bank's watermark stays put and heals
  on the next run.
- **Idempotency is the contract**: `UNIQUE(source, external_id, bank_id)` for
  `raw_item`, natural PK for loader tables. Pick an `external_id` the source
  guarantees stable (URL, accession number, order number).
- **Rate limits**: pass a source-appropriate `throttle_s` (GDELT taught us:
  when in doubt, slower). Respect documented limits before the API teaches
  you the hard way.
- Enforcement/press sources are **event-driven and sparse** — a bank with no
  events is normal; don't treat empty results as failure.
- Questions or something the checklist doesn't cover → RUNBOOK §6, then the
  header comments of `001_bank.sql` / `002_raw_item.sql`, then ask in the
  team channel before inventing a third pattern.
