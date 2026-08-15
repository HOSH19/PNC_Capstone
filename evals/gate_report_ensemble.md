# Quality gate — 2026-08-09

Human labels: `evals/items/gold_slice_*.csv` (300 rows). Llama labels: `labels_ensemble_full.csv` (`label_source='llama_kaggle'`, prompt v2).

## Headline (random sample, slices 1–5)

Overall agreement **230/250 = 92.0%** — placeholder threshold ≥85%: **PASS** on the letter of the threshold. **Read the next section before acting on that word** — the threshold sits below the do-nothing baseline, so passing it carries no information.

Slice 6 is a deliberate directional oversample (20 neg / 20 pos / 10 neu on Llama's label, own-bank titles only — see `export_new_gold_slice.py`), so it is excluded here; pooled over all 300 rows the number would read 273/300 = 91.0%, biased by sampling design, not labeler quality.

## Chance-corrected reading (the number that matters)

| | random (1–5) | pooled (1–6) |
|---|---|---|
| raw agreement | 0.920 | 0.910 |
| all-neutral baseline | 0.916 | 0.840 |
| **Cohen's kappa** | 0.541 | 0.683 |
| macro F1 | 0.733 | 0.828 |

**The gate threshold is set below the trivial baseline.** On the random sample, 91.6% of human labels are `neutral`, so a labeler that answers `neutral` to every row scores 91.6% — above both the ≥85% threshold and Llama's own 92.0%. Raw agreement cannot distinguish a working labeler from a broken one on this corpus; the team's threshold decision should be restated in kappa or macro-F1.

Kappa 0.541 (random) / 0.683 (pooled) is *moderate* agreement on the conventional reading — not a failure, not a pass.

**Methodology limit worth stating in the report.** The human side is a second-pass *review* of slices 1–5, not a blind independent re-label, so the labels are better than a single pass but no inter-annotator agreement statistic exists — and a review cannot produce one, because a reviewer who sees the first answer mostly agrees with it. The ceiling these kappa numbers are measured against is therefore unknown.

## Acceptance criteria for the next labeling run

Fixed before the run so they cannot be renegotiated afterwards. Every directional criterion is **paired**: precision alone is passed by a labeler that stops saying the class at all, recall alone by one that says it everywhere. Raw agreement is deliberately not a criterion.

| criterion | target | measured | n | status |
|---|---|---|---|---|
| kappa (primary) | ≥0.60 | 0.683 | 300 | meets |
| negative precision | ≥0.60 | 0.923 | 13 | meets |
| negative recall — guards the above | ≥0.85 | 0.857 | 14 | meets |
| positive precision | ≥0.60 | 0.622 | 37 | meets |
| positive recall — guards the above | ≥0.70 | 0.676 | 34 | **below** |
| neutral recall — guards over-correction | ≥0.82 | 0.944 | 252 | meets |

Two of these demand improvement (kappa and negative precision — the defects v3 exists to fix); the rest sit at or just under today's value as **no-regression floors**, so a run that fixes negative by breaking positive or neutral fails. A floor already marked `meets` is not slack — it is the level that must survive.

The thresholds were fixed on 2026-08-07 from the prompt-v2 baseline, so reading them against v2 itself shows what was wrong rather than a verdict. Note the sample sizes: on the directional rows one row moves the number by several points, so these detect a large effect, not a small one.

## Per-class agreement (all rows, both directions)

Placeholder threshold ≥75% per class — **the denominator direction is undefined in DESIGN.md, so both are reported and the per-class verdict is deferred to the team's threshold decision.**

| class | human-side (recall) | llama-side (precision) |
|---|---|---|
| positive | 23/34 = 67.6% | 23/37 = 62.2% |
| negative | 12/14 = 85.7% | 12/13 = 92.3% |
| neutral | 238/252 = 94.4% | 238/250 = 95.2% |

**Key finding:** llama-side negative precision is 12/13 = 92.3% — when Llama says `negative`, humans usually see `neutral`. Llama over-calls negative; that is the systematic-bias pattern DESIGN.md asks this review to find.

This is the finding that gates Stage 2, not the headline number. Directional labels are the whole training signal, and the training set carries only a few hundred `negative` rows to begin with — if most of them are `neutral` in truth, the fine-tune learns the error.

### Negative over-call (0 rows: llama `negative`, human `neutral`)

0 of the 0 match the analyst/rating title heuristic — the bank is the *analyst* and another company is the subject. Those rows are the `keep for now` class in DESIGN.md's training-set hygiene table, so **the training-set filters do not remove them**: this is a labeling-prompt problem, not a filter problem. The remainder are ordinary news where the tone is negative but the bank's risk is not (`labeling_guide.md` trap tables).

| id | slice | source | human | llama | title |
|---|---|---|---|---|---|

## Confusion (llama → human)

| llama \ human | positive | negative | neutral |
|---|---|---|---|
| positive | 23 | 0 | 14 |
| negative | 1 | 12 | 0 |
| neutral | 10 | 2 | 238 |

## Dev vs holdout

`dev` is what a prompt may be tuned against; `holdout` is read once, at the end, and is the only human-truth evaluation the FinBERT training set is kept away from. A run that improves on dev while holdout drops is overfitted to the rows the prompt was written against.

| stratum | n | agreement | kappa | macro F1 |
|---|---|---|---|---|
| dev | 125 | 113/125 = 90.4% | 0.715 | 0.853 |
| holdout | 175 | 160/175 = 91.4% | 0.648 | 0.799 |

⚠️ The holdout is **not fully blind**: this report lists every disagreeing row, holdout rows included, so anyone who read an earlier revision has seen them. Prompt revisions must be written from the dev rows and `labeling_guide.md` only, and the final write-up should call this holdout semi-blind rather than blind.

## By slice

| slice | agreement |
|---|---|
| gold_slice_1 | 45/50 = 90.0% |
| gold_slice_2 | 47/50 = 94.0% |
| gold_slice_3 | 47/50 = 94.0% |
| gold_slice_4 | 45/50 = 90.0% |
| gold_slice_5 | 46/50 = 92.0% |
| gold_slice_6 | 43/50 = 86.0% (stratified — see headline note) |

## By source

| source | agreement |
|---|---|
| edgar | 60/75 = 80.0% |
| gdelt | 213/225 = 94.7% |

## Tone≠direction subgroup (title heuristic)

Agreement 44/48 = 91.7%. DESIGN.md requires reading the gate stratified on this subgroup — the heuristic flags candidate rows; eyeball them:

| id | slice | source | human | llama | title |
|---|---|---|---|---|---|
| 293 | 1 | gdelt | neutral | neutral | Ramsay Stattman Vela & Price Inc . Cuts Stake in Illinois Tool Works Inc . $ITW |
| 2501 | 1 | gdelt | neutral | neutral | Sumitomo Mitsui Trust Group Inc . Has $113 . 93 Million Stake in Zoetis Inc . $ZTS |
| 17441 | 1 | gdelt | neutral | neutral | CubeSmart ( NYSE : CUBE ) Receives $43 . 18 Average Price Target from Analysts |
| 4068 | 1 | gdelt | neutral | neutral | Labcorp ( NYSE : LH ) Price Target Raised to $320 . 00 |
| 5426 | 1 | gdelt | neutral | neutral | ClearShares Ultra - Short Maturity ETF ( NYSEARCA : OPER ) Sees Large Decline in Short Interest |
| 14736 | 1 | gdelt | neutral | neutral | EPR Properties ( NYSE : EPR ) Hits New 1 - Year High After Dividend Announcement |
| 17754 | 1 | gdelt | neutral | neutral | BMO Capital Markets Upgrades Amcor ( NYSE : AMCR ) to Hold |
| 10644 | 1 | gdelt | neutral | neutral | Apple , IBM downgraded : Wall Street top analyst calls |
| 9474 | 1 | gdelt | neutral | neutral | Comfort Systems united states ( NYSE : FIX ) Downgraded by Erste Group Bank to Hold |
| 20381 | 2 | gdelt | neutral | neutral | Arcus Biosciences ( NYSE : RCUS ) Stock Price Up 6 . 6 % on Analyst Upgrade |
| 19745 | 2 | gdelt | neutral | neutral | JPMorgan Chase & Co . Cuts Redwood Trust ( NYSE : RWT ) Price Target to $6 . 00 |
| 14622 | 2 | gdelt | neutral | neutral | PayPal ( NASDAQ : PYPL ) Trading Up 2 . 2 % After Analyst Upgrade |
| 21116 | 2 | gdelt | negative | negative | Commerce Bancshares ( NASDAQ : CBSH ) Downgraded by Wall Street Zen to Sell |
| 289 | 2 | gdelt | neutral | neutral | Fidelity National Information Services ( NYSE : FIS ) Stock Rating Upgraded by Barclays |
| 16330 | 2 | gdelt | neutral | neutral | Short Interest in Global X SuperIncome Preferred ETF ( NYSEARCA : SPFF ) Declines By 38 . 0 % |
| 19801 | 2 | gdelt | negative | neutral | Bank Stocks Underperform post Q1 : ICICI Bank gains while HDFC Bank , Axis Bank , Kotak Mahindra , Yes Bank decline |
| 6405 | 3 | gdelt | neutral | neutral | Ardent Health ( NYSE : ARDT ) Shares Gap Down on Analyst Downgrade |
| 10072 | 3 | gdelt | neutral | neutral | Short Interest in Select Medical Holdings Corporation ( NYSE : SEM ) Declines By 52 . 1 %  |
| 5164 | 3 | gdelt | neutral | neutral | WisdomTree Continuous Commodity Index Fund ( NYSEARCA : GCC ) Sees Large Decline in Short Interest |
| 23233 | 3 | gdelt | neutral | neutral | Rekor Systems Q4 EPS Forecast Cut by Northland Securities |
| 11808 | 3 | gdelt | neutral | neutral | Jim Cramer Didnt Hold Back With SanDisk Corporation ( SNDK ) Price Target Upgrade |
| 17047 | 4 | gdelt | neutral | neutral | Madison Square Garden ( NYSE : MSGS ) Upgraded at Morgan Stanley |
| 14605 | 4 | gdelt | neutral | neutral | AllianceBernstein Global High Income Fund , Inc . ( NYSE : AWF ) Sees Significant Decline in Short Interest |
| 6445 | 4 | gdelt | neutral | neutral | Globus Medical ( NYSE : GMED ) Stock Price Down 5 . 4 % on Analyst Downgrade |
| 4541 | 4 | gdelt | neutral | neutral | TD Cowen Cuts Intuitive Surgical ( NASDAQ : ISRG ) Price Target to $520 . 00 |
| 19652 | 4 | gdelt | neutral | neutral | Wall Street Zen Upgrades Corning ( NYSE : GLW ) to  Buy   |
| 20414 | 4 | gdelt | neutral | neutral | e . l . f . Beauty ( NYSE : ELF ) Shares Up 5 . 6 % Following Analyst Upgrade |
| 21269 | 4 | gdelt | neutral | neutral | Viridian Therapeutics ( NASDAQ : VRDN ) Upgraded to Sell at Wall Street Zen |
| 5549 | 4 | gdelt | neutral | neutral | Short Interest in Innovator Equity Defined Protection ETF – 2 Yr to January 2027 ( BATS : TJAN ) Decreases By 55 . 7 % |
| 12140 | 5 | gdelt | neutral | neutral | Extra Space Storage ( NYSE : EXR ) Earns Outperform Rating from Analysts at Raymond James Financial |
| 20301 | 5 | gdelt | neutral | neutral | Western Union ( NYSE : WU ) Shares Down 4 . 3 % After Analyst Downgrade |
| 6127 | 5 | gdelt | neutral | neutral | Q3 EPS Estimates for Nurix Therapeutics Cut by HC Wainwright |
| 7751 | 5 | gdelt | neutral | positive | M & T Bank Corporation Announces Quarterly Preferred Stock Dividends |
| 15894 | 5 | gdelt | positive | positive | Baystreet . ca - Goldman Sachs Posts Record Earnings And Raises Dividend 11 %  |
| 20122 | 5 | gdelt | positive | neutral | Peapack - Gladstone Financial ( NASDAQ : PGC ) Stock Rating Upgraded by Wall Street Zen |
| 3789 | 5 | gdelt | neutral | neutral | CSX ( NASDAQ : CSX ) Reaches New 12 - Month High After Analyst Upgrade |
| 23704 | 5 | gdelt | neutral | neutral | HubSpot ( NYSE : HUBS ) Shares Down 4 . 6 % on Analyst Downgrade |
| 557 | 5 | gdelt | neutral | neutral | Phillips 66 ( NYSE : PSX ) Reaches New 1 - Year High After Analyst Upgrade |
| 14275 | 5 | gdelt | neutral | neutral | Vertiv ( NYSE : VRT ) Price Target Cut to $418 . 00 by Analysts at Royal Bank Of Canada |
| 20428 | 5 | gdelt | neutral | neutral | Brinker International ( NYSE : EAT ) Reaches New 1 - Year High Following Analyst Upgrade |
| 9307 | 6 | gdelt | neutral | neutral | Investors Buy Large Volume of Bank of New York Mellon Put Options ( NYSE : BNY )  |
| 3870 | 6 | gdelt | neutral | neutral | Morgan Stanley India Investment Fund , Inc . ( NYSE : IIF ) Sees Significant Decline in Short Interest |
| 17377 | 6 | gdelt | neutral | neutral | CAE ( TSE : CAE ) Cut to Underweight at Morgan Stanley |
| 14336 | 6 | gdelt | positive | neutral | Morgan Stanley ( NYSE : MS ) Price Target Raised to $250 . 00 |
| 2060 | 6 | gdelt | negative | negative | Oppenheimer Downgrades Bank of America ( BAC ) to Perform |
| 11848 | 6 | gdelt | neutral | neutral | Hikma Pharmaceutical ( HIK )  Buy  Rating Reiterated at Citigroup |
| 8270 | 6 | gdelt | neutral | neutral | Pearson downgraded by JP Morgan after six - year  overweight  call as shares slip 3 . 6 %  |
| 16061 | 6 | gdelt | neutral | neutral | Short Interest in Wells Fargo Advantage Funds �?Allspring Income Opportunities Fund ( NYSEAMERICAN : EAD ) Decreases By 37 . 7 %  |

## Disagreements (27 rows)

| id | slice | source | human | llama | title |
|---|---|---|---|---|---|
| 1857 | 1 | edgar | neutral | positive | SouthState Corporation 8-K |
| 9488 | 1 | gdelt | positive | neutral | JPMorgan invests $24M to resurrect rusting US shipyards |
| 1441 | 1 | edgar | neutral | positive | Capital One Financial Corporation 8-K |
| 1453 | 1 | edgar | neutral | positive | Fifth Third Bancorp 8-K |
| 10428 | 1 | edgar | neutral | positive | TriCo Bancshares 8-K |
| 1544 | 2 | edgar | neutral | positive | Wells Fargo & Company 8-K |
| 6648 | 2 | gdelt | positive | neutral | Morgan Stanley ( NYSE : MS ) Sets New 52 - Week High – Here Why |
| 19801 | 2 | gdelt | negative | neutral | Bank Stocks Underperform post Q1 : ICICI Bank gains while HDFC Bank , Axis Bank , Kotak Mahindra , Yes Bank decline |
| 1640 | 3 | edgar | neutral | positive | Customers Bancorp, Inc. 8-K |
| 3938 | 3 | gdelt | negative | neutral | 2026 - 07 - 13 | TCBK Stock Alert : Halper Sadeh LLC is Investigating Whether TriCo Bancshares is Obtaining a Fair Price for its Shareholders | NDAQ : TCBK |
| 15996 | 3 | gdelt | positive | neutral | Erste Group Bank Predicts Higher Earnings for UBS Group |
| 7143 | 4 | gdelt | positive | neutral | June CPI : Inflation eased following recent surge driven by Iran war |
| 17632 | 4 | gdelt | positive | negative | Citizens Bank to cut ties with CoreCivic and GEO after a fierce public pressure campaign |
| 1578 | 4 | edgar | neutral | positive | Axos Financial, Inc. 8-K |
| 1914 | 4 | edgar | neutral | positive | Valley National Bancorp 8-K |
| 15723 | 4 | edgar | neutral | positive | F.N.B. Corporation 8-K |
| 7751 | 5 | gdelt | neutral | positive | M & T Bank Corporation Announces Quarterly Preferred Stock Dividends |
| 1472 | 5 | edgar | neutral | positive | Huntington Bancshares Incorporated 8-K |
| 20122 | 5 | gdelt | positive | neutral | Peapack - Gladstone Financial ( NASDAQ : PGC ) Stock Rating Upgraded by Wall Street Zen |
| 1629 | 5 | edgar | neutral | positive | Cullen/Frost Bankers, Inc. 8-K |
| 22918 | 6 | gdelt | positive | neutral | FinancialContent - Western Alliance Bancorporation ( NYSE : WAL ) Surprises With Q2 CY2026 Sales |
| 1588 | 6 | edgar | positive | neutral | Banner Financial Corporation 8-K |
| 1457 | 6 | edgar | neutral | positive | Fifth Third Bancorp 8-K |
| 1674 | 6 | edgar | neutral | positive | First BanCorp. 8-K/A |
| 14336 | 6 | gdelt | positive | neutral | Morgan Stanley ( NYSE : MS ) Price Target Raised to $250 . 00 |
| 3580 | 6 | edgar | positive | neutral | TriCo Bancshares 8-K |
| 8881 | 6 | gdelt | positive | neutral | Goldman Sachs Q2 profit soars 78 %  on trading rally |
