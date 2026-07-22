# Draft — EDA update email to mentor (supersedes unsent 2026-07-13 draft)

Subject: Data pipeline live + first EDA findings (corpus size, noise profile, label scarcity)

Hi [Mentor],

Quick update as promised. The ingestion pipeline is now fully automated
(GitHub Actions every 6 hours, 104 banks — the top-100 list you asked for,
filtered to US-listed holding companies so the text sources actually exist
for every bank), and we ran an EDA over what it has collected (re-run
2026-07-16; full report attached). Highlights below.

**1. Corpus size (your question from the call).** GDELT news is arriving at
roughly 1,200–1,300 articles/day across 86 of the 104 banks, and Alpha
Vantage news (integrated by Ming this week, along with NewsAPI and an
agency-RSS poller) adds ~500/day more. 8-K/10-Q/10-K filings run ~150 per
quarter-month. At this rate a 6-month window is 250k+ articles — well above
our earlier 80–150k projection — so manual labeling is definitively off the
table, and we'll go with the LLM-assisted labeling + sample verification
approach you suggested, then fine-tune a BERT-based model on those labels.

**2. The noise profile is the real first finding.** Three things dominate
before any modeling:
- ~14% of headlines are institutional-holdings syndication spam ("XYZ LLC
  sells 4,921 shares…"), and the top 3 domains (all one press-release
  network) account for 33% of articles. The ratio has held steady as the
  corpus grew 5x, so it's structural, not a backfill artifact.
- ~29% of articles are non-English (Chinese 12%, Spanish 6%) — we already
  store the language tag, so we'll filter to English for the FinBERT track.
- Only ~1.4% of headlines hit an obvious risk lexicon (decline, downgrade,
  fraud, probe, fine…) — the corpus is overwhelmingly neutral, which matches
  your point that all three sentiment classes need to be first-class.
Our take: labeling should run on a filtered corpus (English, holdings-spam
tagged), otherwise we'd spend most of the label budget on noise.

**3. News coverage is heavily skewed to large banks.** Median GDELT volume
is 4 articles per bank while the max is ~1,600 — half our universe has
almost no news at all. So the news-sentiment signal will realistically only
be dense for the larger banks, and we're prioritizing the remaining source
integrations by whether they cover that long tail (or contribute
label-adjacent events like enforcement/litigation), not by raw volume.

**4. Fundamentals: feature-extraction bug found, fixed, and backfilled.**
The min/max scan first showed `tier1_capital_ratio`, `total_capital_ratio`,
and `securities_unrealized_loss` 100% NULL across all 435k bank-quarters.
Root causes (verified against real FFIEC archives from 2010/2014/2026):
ratio fields carry a percent sign since 2015 that the numeric parser
rejected, the RC-R schedule changed names across eras (RCR → RCRIA/RCRIB →
RCRI), and the unrealized-loss column had never been wired up. All three are
fixed and the full 73-quarter dataset is rebuilt and re-verified as of
today: capital ratios are populated for ~90% of bank-quarters (100% for
2008–2019, ~60% after 2020 — CBLR-electing small banks are legally exempt,
so that's expected), unrealized loss 100%. Sanity checks line up with public
disclosures (e.g. JPM's −$70.7B securities unrealized loss). With this, the
GP thresholds can use the real capital ratios: e.g. latest-quarter NPL
median 0.46 / p95 3.40, liquidity median 23.6 / p95 58.8.

**5. Label reality check — the one constraint more data can't fix.**
`distress_within_4q` positives are 0.45% of bank-quarters (the
rare-positive-class issue you warned about), and of ~3,600 failure events
only ~27 fall in 2017+ where news coverage exists — the 2008 cluster
predates GDELT. Adding news sources doesn't change this number, so we'd
like your read on our two-part plan:
- (a) broaden the positive class beyond outright failure to near-distress
  events inside the news era — regulatory enforcement actions (the Fed's
  full history back to 1990 is now ingested: 89 actions across 34 of our
  banks), and potentially dividend cuts / severe drawdowns;
- (b) treat the failure backtest as a rare-event evaluation and lean on
  qualitative case studies (SVB, First Republic, NYCB) rather than claiming
  statistical significance from 27 events.
Does that split sound right to you, and would you weight (a) toward
enforcement actions only, or include the market-based events too?

Next steps this week: we're designing the automated-labeling architecture
that will generate the training data for the model fine-tune — the pipeline
around the LLM labeler (eligibility filters, label storage, verification
sampling) plus the labeling bake-off itself (Qwen vs Llama vs Gemini on a
human-verified sample).

One logistics note: rather than a separate touchpoint this week, we'll send
a progress update by email on Sunday (7/19). Of course, happy to jump on a
call if anything above needs discussion before then.

Best,
Jiwon

[Attachment: 2026-07-16_ingest_eda.md]

<!-- TODO before sending (do not include in email):
  1. FDIC enforcement orders: migration 005 + loader are committed, but the
     live DB currently has 0 fdic_enforcement rows. Verify/fix the load or
     keep FDIC out of the email (currently only Fed is claimed).
  2. Ming integration: confirm Ming is OK being named for alpha_vantage /
     newsapi / agency_rss, and that those branch migrations are merging.
  3. The 7/13 draft mentioned the 8-K boilerplate-window issue and a
     "tier1 median ~14" figure; dropped here (8-K window: still true but we
     may have a fix by send time — re-add if not; tier1 median: not in the
     attached report, re-verify before quoting).
-->
