# Quality gate — 2026-08-07

Human labels: `evals/items/gold_slice_*.csv` (300 rows). Llama labels: `labels_2026-07-22.csv` (`label_source='llama_kaggle'`, prompt v2).

## Headline (random sample, slices 1–5)

Overall agreement **221/250 = 88.4%** — placeholder threshold ≥85%: **PASS** on the letter of the threshold. **Read the next section before acting on that word** — the threshold sits below the do-nothing baseline, so passing it carries no information.

Slice 6 is a deliberate directional oversample (20 neg / 20 pos / 10 neu on Llama's label, own-bank titles only — see `export_new_gold_slice.py`), so it is excluded here; pooled over all 300 rows the number would read 253/300 = 84.3%, biased by sampling design, not labeler quality.

## Chance-corrected reading (the number that matters)

| | random (1–5) | pooled (1–6) |
|---|---|---|
| raw agreement | 0.884 | 0.843 |
| all-neutral baseline | 0.916 | 0.840 |
| **Cohen's kappa** | 0.473 | 0.564 |
| macro F1 | 0.648 | 0.700 |

**The gate threshold is set below the trivial baseline.** On the random sample, 91.6% of human labels are `neutral`, so a labeler that answers `neutral` to every row scores 91.6% — above both the ≥85% threshold and Llama's own 88.4%. Raw agreement cannot distinguish a working labeler from a broken one on this corpus; the team's threshold decision should be restated in kappa or macro-F1.

Kappa 0.473 (random) / 0.564 (pooled) is *moderate* agreement on the conventional reading — not a failure, not a pass.

**Methodology limit worth stating in the report.** The human side is a second-pass *review* of slices 1–5, not a blind independent re-label, so the labels are better than a single pass but no inter-annotator agreement statistic exists — and a review cannot produce one, because a reviewer who sees the first answer mostly agrees with it. The ceiling these kappa numbers are measured against is therefore unknown.

## Acceptance criteria for the next labeling run

Fixed before the run so they cannot be renegotiated afterwards. Every directional criterion is **paired**: precision alone is passed by a labeler that stops saying the class at all, recall alone by one that says it everywhere. Raw agreement is deliberately not a criterion.

| criterion | target | current (v2) | n | status |
|---|---|---|---|---|
| kappa (primary) | ≥0.60 | 0.564 | 300 | **below** |
| negative precision | ≥0.60 | 0.361 | 36 | **below** |
| negative recall — guards the above | ≥0.85 | 0.929 | 14 | meets |
| positive precision | ≥0.60 | 0.605 | 43 | meets |
| neutral recall — guards over-correction | ≥0.82 | 0.849 | 252 | meets |

Two of these demand improvement (kappa and negative precision — the defects v3 exists to fix); the rest sit at or just under today's value as **no-regression floors**, so a run that fixes negative by breaking positive or neutral fails. A floor already marked `meets` is not slack — it is the level that must survive.

Read these as the bar the *next* run has to clear, not as a verdict on v2 — v2 is the measurement that set them. Note the sample sizes: on the directional rows one row moves the number by several points, so these detect a large effect, not a small one.

## Per-class agreement (all rows, both directions)

Placeholder threshold ≥75% per class — **the denominator direction is undefined in DESIGN.md, so both are reported and the per-class verdict is deferred to the team's threshold decision.**

| class | human-side (recall) | llama-side (precision) |
|---|---|---|
| positive | 26/34 = 76.5% | 26/43 = 60.5% |
| negative | 13/14 = 92.9% | 13/36 = 36.1% |
| neutral | 214/252 = 84.9% | 214/221 = 96.8% |

**Key finding:** llama-side negative precision is 13/36 = 36.1% — when Llama says `negative`, humans usually see `neutral`. Llama over-calls negative; that is the systematic-bias pattern DESIGN.md asks this review to find.

This is the finding that gates Stage 2, not the headline number. Directional labels are the whole training signal, and the training set carries only a few hundred `negative` rows to begin with — if most of them are `neutral` in truth, the fine-tune learns the error.

### Negative over-call (21 rows: llama `negative`, human `neutral`)

11 of the 21 match the analyst/rating title heuristic — the bank is the *analyst* and another company is the subject. Those rows are the `keep for now` class in DESIGN.md's training-set hygiene table, so **the training-set filters do not remove them**: this is a labeling-prompt problem, not a filter problem. The remainder are ordinary news where the tone is negative but the bank's risk is not (`labeling_guide.md` trap tables).

| id | slice | source | human | llama | title |
|---|---|---|---|---|---|
| 15874 | 1 | gdelt | neutral | negative | Major financial corporation announces more layoffs in Illinois |
| 5426 | 1 | gdelt | neutral | negative | ClearShares Ultra - Short Maturity ETF ( NYSEARCA : OPER ) Sees Large Decline in Short Interest |
| 9474 | 1 | gdelt | neutral | negative | Comfort Systems united states ( NYSE : FIX ) Downgraded by Erste Group Bank to Hold |
| 19745 | 2 | gdelt | neutral | negative | JPMorgan Chase & Co . Cuts Redwood Trust ( NYSE : RWT ) Price Target to $6 . 00 |
| 6405 | 3 | gdelt | neutral | negative | Ardent Health ( NYSE : ARDT ) Shares Gap Down on Analyst Downgrade |
| 10026 | 3 | gdelt | neutral | negative | IBM Stock Sinks 22 % Pre - Market After Rare Q2 Revenue Warning |
| 15125 | 3 | gdelt | neutral | negative | Bankruptcy Court Holds That Receivership Order Divests Debtor Manager Of Authority To File Chapter 11 Petition - Insolvency / Bankruptcy |
| 23233 | 3 | gdelt | neutral | negative | Rekor Systems Q4 EPS Forecast Cut by Northland Securities |
| 20301 | 5 | gdelt | neutral | negative | Western Union ( NYSE : WU ) Shares Down 4 . 3 % After Analyst Downgrade |
| 21666 | 6 | gdelt | neutral | negative | Citizens Bank cuts ties with ICE , activists push for stronger commitment |
| 9307 | 6 | gdelt | neutral | negative | Investors Buy Large Volume of Bank of New York Mellon Put Options ( NYSE : BNY )  |
| 3870 | 6 | gdelt | neutral | negative | Morgan Stanley India Investment Fund , Inc . ( NYSE : IIF ) Sees Significant Decline in Short Interest |
| 17377 | 6 | gdelt | neutral | negative | CAE ( TSE : CAE ) Cut to Underweight at Morgan Stanley |
| 1927 | 6 | edgar | neutral | negative | WesBanco, Inc. 8-K |
| 21306 | 6 | gdelt | neutral | negative | Bigger crash ahead ? JPMorgan CEO Dimon says he wont buy stocks at current prices , says markets underestimating risks |
| 5631 | 6 | gdelt | neutral | negative | El Paso Wells Fargo Building Sign Is Up For Auction |
| 8270 | 6 | gdelt | neutral | negative | Pearson downgraded by JP Morgan after six - year  overweight  call as shares slip 3 . 6 %  |
| 3893 | 6 | gdelt | neutral | negative | Morgan Stanley Lowers PT on Hertz Global ( HTZ )  |
| 17234 | 6 | gdelt | neutral | negative | JPMorgan sexual harassment lawsuit dismissed , refiling expected |
| 19261 | 6 | gdelt | neutral | negative | Man arrested in connection with armed robbery of Springville PNC Bank |
| 16061 | 6 | gdelt | neutral | negative | Short Interest in Wells Fargo Advantage Funds �?Allspring Income Opportunities Fund ( NYSEAMERICAN : EAD ) Decreases By 37 . 7 %  |

## Confusion (llama → human)

| llama \ human | positive | negative | neutral |
|---|---|---|---|
| positive | 26 | 0 | 17 |
| negative | 2 | 13 | 21 |
| neutral | 6 | 1 | 214 |

## Dev vs holdout

`dev` is what a prompt may be tuned against; `holdout` is read once, at the end, and is the only human-truth evaluation the FinBERT training set is kept away from. A run that improves on dev while holdout drops is overfitted to the rows the prompt was written against.

| stratum | n | agreement | kappa | macro F1 |
|---|---|---|---|---|
| dev | 125 | 100/125 = 80.0% | 0.511 | 0.673 |
| holdout | 175 | 153/175 = 87.4% | 0.609 | 0.723 |

⚠️ The holdout is **not fully blind**: this report lists every disagreeing row, holdout rows included, so anyone who read an earlier revision has seen them. Prompt revisions must be written from the dev rows and `labeling_guide.md` only, and the final write-up should call this holdout semi-blind rather than blind.

## By slice

| slice | agreement |
|---|---|
| gold_slice_1 | 43/50 = 86.0% |
| gold_slice_2 | 46/50 = 92.0% |
| gold_slice_3 | 42/50 = 84.0% |
| gold_slice_4 | 43/50 = 86.0% |
| gold_slice_5 | 47/50 = 94.0% |
| gold_slice_6 | 32/50 = 64.0% (stratified — see headline note) |

## By source

| source | agreement |
|---|---|
| edgar | 64/75 = 85.3% |
| gdelt | 189/225 = 84.0% |

## Tone≠direction subgroup (title heuristic)

Agreement 33/48 = 68.8%. DESIGN.md requires reading the gate stratified on this subgroup — the heuristic flags candidate rows; eyeball them:

| id | slice | source | human | llama | title |
|---|---|---|---|---|---|
| 293 | 1 | gdelt | neutral | neutral | Ramsay Stattman Vela & Price Inc . Cuts Stake in Illinois Tool Works Inc . $ITW |
| 2501 | 1 | gdelt | neutral | neutral | Sumitomo Mitsui Trust Group Inc . Has $113 . 93 Million Stake in Zoetis Inc . $ZTS |
| 17441 | 1 | gdelt | neutral | neutral | CubeSmart ( NYSE : CUBE ) Receives $43 . 18 Average Price Target from Analysts |
| 4068 | 1 | gdelt | neutral | neutral | Labcorp ( NYSE : LH ) Price Target Raised to $320 . 00 |
| 5426 | 1 | gdelt | neutral | negative | ClearShares Ultra - Short Maturity ETF ( NYSEARCA : OPER ) Sees Large Decline in Short Interest |
| 14736 | 1 | gdelt | neutral | neutral | EPR Properties ( NYSE : EPR ) Hits New 1 - Year High After Dividend Announcement |
| 17754 | 1 | gdelt | neutral | positive | BMO Capital Markets Upgrades Amcor ( NYSE : AMCR ) to Hold |
| 10644 | 1 | gdelt | neutral | neutral | Apple , IBM downgraded : Wall Street top analyst calls |
| 9474 | 1 | gdelt | neutral | negative | Comfort Systems united states ( NYSE : FIX ) Downgraded by Erste Group Bank to Hold |
| 20381 | 2 | gdelt | neutral | neutral | Arcus Biosciences ( NYSE : RCUS ) Stock Price Up 6 . 6 % on Analyst Upgrade |
| 19745 | 2 | gdelt | neutral | negative | JPMorgan Chase & Co . Cuts Redwood Trust ( NYSE : RWT ) Price Target to $6 . 00 |
| 14622 | 2 | gdelt | neutral | neutral | PayPal ( NASDAQ : PYPL ) Trading Up 2 . 2 % After Analyst Upgrade |
| 21116 | 2 | gdelt | negative | negative | Commerce Bancshares ( NASDAQ : CBSH ) Downgraded by Wall Street Zen to Sell |
| 289 | 2 | gdelt | neutral | positive | Fidelity National Information Services ( NYSE : FIS ) Stock Rating Upgraded by Barclays |
| 16330 | 2 | gdelt | neutral | neutral | Short Interest in Global X SuperIncome Preferred ETF ( NYSEARCA : SPFF ) Declines By 38 . 0 % |
| 19801 | 2 | gdelt | negative | negative | Bank Stocks Underperform post Q1 : ICICI Bank gains while HDFC Bank , Axis Bank , Kotak Mahindra , Yes Bank decline |
| 6405 | 3 | gdelt | neutral | negative | Ardent Health ( NYSE : ARDT ) Shares Gap Down on Analyst Downgrade |
| 10072 | 3 | gdelt | neutral | neutral | Short Interest in Select Medical Holdings Corporation ( NYSE : SEM ) Declines By 52 . 1 %  |
| 5164 | 3 | gdelt | neutral | neutral | WisdomTree Continuous Commodity Index Fund ( NYSEARCA : GCC ) Sees Large Decline in Short Interest |
| 23233 | 3 | gdelt | neutral | negative | Rekor Systems Q4 EPS Forecast Cut by Northland Securities |
| 11808 | 3 | gdelt | neutral | neutral | Jim Cramer Didnt Hold Back With SanDisk Corporation ( SNDK ) Price Target Upgrade |
| 17047 | 4 | gdelt | neutral | neutral | Madison Square Garden ( NYSE : MSGS ) Upgraded at Morgan Stanley |
| 14605 | 4 | gdelt | neutral | neutral | AllianceBernstein Global High Income Fund , Inc . ( NYSE : AWF ) Sees Significant Decline in Short Interest |
| 6445 | 4 | gdelt | neutral | neutral | Globus Medical ( NYSE : GMED ) Stock Price Down 5 . 4 % on Analyst Downgrade |
| 4541 | 4 | gdelt | neutral | neutral | TD Cowen Cuts Intuitive Surgical ( NASDAQ : ISRG ) Price Target to $520 . 00 |
| 19652 | 4 | gdelt | neutral | neutral | Wall Street Zen Upgrades Corning ( NYSE : GLW ) to  Buy   |
| 20414 | 4 | gdelt | neutral | positive | e . l . f . Beauty ( NYSE : ELF ) Shares Up 5 . 6 % Following Analyst Upgrade |
| 21269 | 4 | gdelt | neutral | neutral | Viridian Therapeutics ( NASDAQ : VRDN ) Upgraded to Sell at Wall Street Zen |
| 5549 | 4 | gdelt | neutral | neutral | Short Interest in Innovator Equity Defined Protection ETF – 2 Yr to January 2027 ( BATS : TJAN ) Decreases By 55 . 7 % |
| 12140 | 5 | gdelt | neutral | neutral | Extra Space Storage ( NYSE : EXR ) Earns Outperform Rating from Analysts at Raymond James Financial |
| 20301 | 5 | gdelt | neutral | negative | Western Union ( NYSE : WU ) Shares Down 4 . 3 % After Analyst Downgrade |
| 6127 | 5 | gdelt | neutral | neutral | Q3 EPS Estimates for Nurix Therapeutics Cut by HC Wainwright |
| 7751 | 5 | gdelt | neutral | neutral | M & T Bank Corporation Announces Quarterly Preferred Stock Dividends |
| 15894 | 5 | gdelt | positive | positive | Baystreet . ca - Goldman Sachs Posts Record Earnings And Raises Dividend 11 %  |
| 20122 | 5 | gdelt | positive | positive | Peapack - Gladstone Financial ( NASDAQ : PGC ) Stock Rating Upgraded by Wall Street Zen |
| 3789 | 5 | gdelt | neutral | neutral | CSX ( NASDAQ : CSX ) Reaches New 12 - Month High After Analyst Upgrade |
| 23704 | 5 | gdelt | neutral | neutral | HubSpot ( NYSE : HUBS ) Shares Down 4 . 6 % on Analyst Downgrade |
| 557 | 5 | gdelt | neutral | neutral | Phillips 66 ( NYSE : PSX ) Reaches New 1 - Year High After Analyst Upgrade |
| 14275 | 5 | gdelt | neutral | neutral | Vertiv ( NYSE : VRT ) Price Target Cut to $418 . 00 by Analysts at Royal Bank Of Canada |
| 20428 | 5 | gdelt | neutral | neutral | Brinker International ( NYSE : EAT ) Reaches New 1 - Year High Following Analyst Upgrade |
| 9307 | 6 | gdelt | neutral | negative | Investors Buy Large Volume of Bank of New York Mellon Put Options ( NYSE : BNY )  |
| 3870 | 6 | gdelt | neutral | negative | Morgan Stanley India Investment Fund , Inc . ( NYSE : IIF ) Sees Significant Decline in Short Interest |
| 17377 | 6 | gdelt | neutral | negative | CAE ( TSE : CAE ) Cut to Underweight at Morgan Stanley |
| 14336 | 6 | gdelt | positive | positive | Morgan Stanley ( NYSE : MS ) Price Target Raised to $250 . 00 |
| 2060 | 6 | gdelt | negative | negative | Oppenheimer Downgrades Bank of America ( BAC ) to Perform |
| 11848 | 6 | gdelt | neutral | positive | Hikma Pharmaceutical ( HIK )  Buy  Rating Reiterated at Citigroup |
| 8270 | 6 | gdelt | neutral | negative | Pearson downgraded by JP Morgan after six - year  overweight  call as shares slip 3 . 6 %  |
| 16061 | 6 | gdelt | neutral | negative | Short Interest in Wells Fargo Advantage Funds �?Allspring Income Opportunities Fund ( NYSEAMERICAN : EAD ) Decreases By 37 . 7 %  |

## Disagreements (47 rows)

| id | slice | source | human | llama | title |
|---|---|---|---|---|---|
| 9488 | 1 | gdelt | positive | neutral | JPMorgan invests $24M to resurrect rusting US shipyards |
| 15874 | 1 | gdelt | neutral | negative | Major financial corporation announces more layoffs in Illinois |
| 5426 | 1 | gdelt | neutral | negative | ClearShares Ultra - Short Maturity ETF ( NYSEARCA : OPER ) Sees Large Decline in Short Interest |
| 17754 | 1 | gdelt | neutral | positive | BMO Capital Markets Upgrades Amcor ( NYSE : AMCR ) to Hold |
| 9474 | 1 | gdelt | neutral | negative | Comfort Systems united states ( NYSE : FIX ) Downgraded by Erste Group Bank to Hold |
| 1453 | 1 | edgar | neutral | positive | Fifth Third Bancorp 8-K |
| 10428 | 1 | edgar | neutral | positive | TriCo Bancshares 8-K |
| 18188 | 2 | gdelt | neutral | positive | Nebius raised $775 million by borrowing against its GPUs . It has $40 billion more contracts to securitise . |
| 19745 | 2 | gdelt | neutral | negative | JPMorgan Chase & Co . Cuts Redwood Trust ( NYSE : RWT ) Price Target to $6 . 00 |
| 289 | 2 | gdelt | neutral | positive | Fidelity National Information Services ( NYSE : FIS ) Stock Rating Upgraded by Barclays |
| 6648 | 2 | gdelt | positive | neutral | Morgan Stanley ( NYSE : MS ) Sets New 52 - Week High – Here Why |
| 6405 | 3 | gdelt | neutral | negative | Ardent Health ( NYSE : ARDT ) Shares Gap Down on Analyst Downgrade |
| 1671 | 3 | edgar | neutral | positive | Eagle Bancorp, Inc. 8-K |
| 3938 | 3 | gdelt | negative | neutral | 2026 - 07 - 13 | TCBK Stock Alert : Halper Sadeh LLC is Investigating Whether TriCo Bancshares is Obtaining a Fair Price for its Shareholders | NDAQ : TCBK |
| 10026 | 3 | gdelt | neutral | negative | IBM Stock Sinks 22 % Pre - Market After Rare Q2 Revenue Warning |
| 7157 | 3 | gdelt | neutral | positive | TD Issues Positive Forecast for Louisiana - Pacific ( NYSE : LPX ) Stock Price |
| 15125 | 3 | gdelt | neutral | negative | Bankruptcy Court Holds That Receivership Order Divests Debtor Manager Of Authority To File Chapter 11 Petition - Insolvency / Bankruptcy |
| 23233 | 3 | gdelt | neutral | negative | Rekor Systems Q4 EPS Forecast Cut by Northland Securities |
| 1406 | 3 | edgar | positive | neutral | Ally Financial Inc. 8-K |
| 7143 | 4 | gdelt | positive | neutral | June CPI : Inflation eased following recent surge driven by Iran war |
| 17632 | 4 | gdelt | positive | negative | Citizens Bank to cut ties with CoreCivic and GEO after a fierce public pressure campaign |
| 1578 | 4 | edgar | neutral | positive | Axos Financial, Inc. 8-K |
| 19966 | 4 | gdelt | neutral | positive | Reformation Launches IPO Of 14 . 06 Mln Shares ; Price To Be Between $15 |
| 20414 | 4 | gdelt | neutral | positive | e . l . f . Beauty ( NYSE : ELF ) Shares Up 5 . 6 % Following Analyst Upgrade |
| 23476 | 4 | gdelt | neutral | positive | Dyne Therapeutics Announces Pricing of Upsized $375 Million Public Offering of Common Stock |
| 15723 | 4 | edgar | neutral | positive | F.N.B. Corporation 8-K |
| 20301 | 5 | gdelt | neutral | negative | Western Union ( NYSE : WU ) Shares Down 4 . 3 % After Analyst Downgrade |
| 23306 | 5 | gdelt | neutral | positive | Chubb ( NYSE : CB ) Announces Earnings Results , Beats Estimates By $0 . 49 EPS |
| 1629 | 5 | edgar | neutral | positive | Cullen/Frost Bankers, Inc. 8-K |
| 22918 | 6 | gdelt | positive | neutral | FinancialContent - Western Alliance Bancorporation ( NYSE : WAL ) Surprises With Q2 CY2026 Sales |
| 21666 | 6 | gdelt | neutral | negative | Citizens Bank cuts ties with ICE , activists push for stronger commitment |
| 1408 | 6 | edgar | positive | neutral | Ally Financial Inc. 8-K |
| 1588 | 6 | edgar | positive | negative | Banner Financial Corporation 8-K |
| 9307 | 6 | gdelt | neutral | negative | Investors Buy Large Volume of Bank of New York Mellon Put Options ( NYSE : BNY )  |
| 3870 | 6 | gdelt | neutral | negative | Morgan Stanley India Investment Fund , Inc . ( NYSE : IIF ) Sees Significant Decline in Short Interest |
| 17377 | 6 | gdelt | neutral | negative | CAE ( TSE : CAE ) Cut to Underweight at Morgan Stanley |
| 1457 | 6 | edgar | neutral | positive | Fifth Third Bancorp 8-K |
| 17656 | 6 | gdelt | neutral | positive | Fifth Third Bancorp Increases Holdings in GATX Corporation $GATX |
| 1927 | 6 | edgar | neutral | negative | WesBanco, Inc. 8-K |
| 21306 | 6 | gdelt | neutral | negative | Bigger crash ahead ? JPMorgan CEO Dimon says he wont buy stocks at current prices , says markets underestimating risks |
| 11848 | 6 | gdelt | neutral | positive | Hikma Pharmaceutical ( HIK )  Buy  Rating Reiterated at Citigroup |
| 5631 | 6 | gdelt | neutral | negative | El Paso Wells Fargo Building Sign Is Up For Auction |
| 8270 | 6 | gdelt | neutral | negative | Pearson downgraded by JP Morgan after six - year  overweight  call as shares slip 3 . 6 %  |
| 3893 | 6 | gdelt | neutral | negative | Morgan Stanley Lowers PT on Hertz Global ( HTZ )  |
| 17234 | 6 | gdelt | neutral | negative | JPMorgan sexual harassment lawsuit dismissed , refiling expected |
| 19261 | 6 | gdelt | neutral | negative | Man arrested in connection with armed robbery of Springville PNC Bank |
| 16061 | 6 | gdelt | neutral | negative | Short Interest in Wells Fargo Advantage Funds �?Allspring Income Opportunities Fund ( NYSEAMERICAN : EAD ) Decreases By 37 . 7 %  |
