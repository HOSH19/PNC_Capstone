# Draft — EDA update email to mentor

Subject: Data pipeline live + first EDA findings (corpus size, noise profile, feature gaps)

Hi [Mentor],

Quick update as promised. The ingestion pipeline is now fully automated
(GitHub Actions every 6 hours, 104 banks — the top-100 list you asked for,
filtered to US-listed holding companies so the text sources actually exist
for every bank), and we ran a first EDA over what it has collected. Full
report attached; highlights below.

**1. Corpus size (your question from the call).** After the initial
backfill, GDELT news is arriving at roughly 450–500 articles/day across the
banks collected so far, and 8-K/10-Q/10-K filings at ~150 per quarter-month.
Projecting to all 104 banks over a 6-month window we expect on the order of
**80–150k articles** — so manual labeling is off the table, and we'll go
with the LLM-assisted labeling + sample verification approach you suggested,
then fine-tune a BERT-based model on those labels.

**2. The noise profile is the real first finding.** Three things dominate
before any modeling:
- ~14% of headlines are institutional-holdings syndication spam ("XYZ LLC
  sells 4,921 shares…"), and the top 3 domains (all one press-release
  network) account for 31% of articles.
- 35% of articles are non-English (Spanish 12%, Chinese 12%) — we already
  store the language tag, so we'll filter to English for the FinBERT track.
- Only ~1.2% of headlines hit an obvious risk lexicon (downgrade, fraud,
  probe, fine…) — so the corpus is overwhelmingly neutral, which matches
  your point that all three classes need to be first-class.
Our take: labeling should run on a filtered corpus (English, holdings-spam
tagged), otherwise we'd spend most of the label budget on noise.

**3. 8-K excerpts need a smarter window.** The first ~4,000 chars are mostly
cover-page boilerplate ("pursuant to Section 13…"); we'll shift extraction
to start at the Item sections before scoring.

**4. Fundamentals: found and fixed a feature-extraction bug.** The min/max
scan first showed `tier1_capital_ratio`, `total_capital_ratio`, and
`securities_unrealized_loss` 100% NULL across all 435k bank-quarters. Root
causes (verified against real FFIEC archives from 2010/2014/2026): ratio
fields carry a percent sign since 2015 that the numeric parser rejected, the
RC-R schedule changed names across eras (RCR → RCRIA/RCRIB → RCRI), and the
unrealized-loss column had never been wired up. All three are fixed and the
full 73-quarter dataset is rebuilt: capital-ratio coverage is now 100% for
2008–2019 and ~60% after 2020 (CBLR-electing small banks are legally exempt
— expected), unrealized loss 100%. Sanity checks line up with public
disclosures (e.g. JPM's −$70.7B securities unrealized loss). Shu Han — the
change is in your module (utils.numeric + ffiec_call_reports); commit
message has the details, please review. With this, the GP thresholds can
use the real capital ratios: e.g. latest-quarter NPL median 0.46 / p95 3.40,
liquidity median 23.6 / p95 58.8, tier1 median ~14 across reporting banks.

**5. Label reality check for the backtest.** `distress_within_4q` positives
are 0.45% of bank-quarters (the rare-positive-class issue you warned about),
and of 3,627 failure events only ~27 fall in 2017+ where GDELT news coverage
exists — the 2008 cluster predates it. So the news-sentiment backtest will
lean on the recent failures (SVB-era and after) plus non-failure distress
signals (enforcement actions — we already ingest FDIC orders, and the Fed
publishes its full history as CSV, which we'll add next).

Next steps this week: preprocessing filters → labeling approach bake-off
(Qwen vs Llama vs Gemini on a human-verified sample) → capital-ratio fix
with Shu Han. Happy to walk through any of this at the end-of-week
touchpoint.

Best,
Jiwon

[Attachment: 2026-07-13_ingest_eda.md]
