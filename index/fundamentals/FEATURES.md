# The 50 features

The fixed feature list behind `gp50_fixed_v1`. Ranked once by XGBoost gain on the first
fold's training window (report quarters 2017Q1–2018Q3, prediction time ≤ 2019-02-13,
40,034 rows / 4,197 positives), then frozen — every walk-forward fold and the production
model use this same list in this same order. `models.py` writes the full 529-name
ranking to `fixed_order.json`; this is its first 50.

**49 of the 50 are MDRM items, each divided by total assets (`RC_2170`) before entering
the model.** They are raw dollar amounts in $thousands as filed; the ratio is what
carries meaning across banks of different sizes. The fiftieth is `log_assets` —
`log(1 + total assets)` — which carries size itself. A collector therefore needs
**50 raw fields**: those 49 items plus `RC_2170`.

Codes are `{schedule}_{MDRM item}`. The FFIEC label column is the field name exactly as
filed, taken from row 2 of each schedule file in the CDR archive — the abbreviations are
theirs. The expanded reading is ours.

Labels are the **most recent** wording. FFIEC relabels items when the accounting
changes, and 83 of the 1,818 in `mdrm_names.json` have moved since 2017: ALLL became
ACL under CECL, troubled debt restructurings became loan modifications under
ASU 2022-02, capitalised leases became right-of-use assets under ASC 842. Six of the
50 below are affected. The MDRM item is the same series throughout — only its name
changed.

---

## By schedule

| Schedule | What it covers | Count |
|---|---|---|
| RC-R II | Risk-weighted asset categories | 12 |
| RC-N | Past due and nonaccrual loans | 8 |
| RC-C I | Loan portfolio composition | 7 |
| RC-O | Deposit insurance assessment base | 5 |
| RC-E | Deposit liabilities by type | 4 |
| RC | Balance sheet | 3 |
| RC-L | Off-balance-sheet commitments | 2 |
| RC-K | Quarterly averages | 2 |
| RC-F | Other assets | 2 |
| RC-B | Securities | 1 |
| RC-R I | Regulatory capital | 1 |
| RI | Income statement | 1 |
| RI-B I | Charge-offs and recoveries | 1 |
| — | Derived | 1 |

Credit quality dominates: RC-N, plus the nonaccrual and loan-modification lines inside
RC-R II and RC-C I, account for 14 of the 50. Deposit structure — RC-E and RC-O together —
accounts for 9, which is what the label's deposit leg would predict.

---

## The list

| # | Code | FFIEC label (as filed) | Reading |
|---|---|---|---|
| 1 | `RCL_3814` | UNUSED COMMITMENTS-REVOLVING, ETC. | Unused commitments on revolving, open-end lines secured by 1–4 family residential properties (home equity lines) |
| 2 | `RCCI_1797` | REVOLVING, OPEN-END LNS SECD BY 1-4F | Revolving, open-end loans secured by 1–4 family residential properties, extended under lines of credit |
| 3 | `RCO_F047` | AMT RETIREMT DEP ACCTS MOR THAN $250 | Amount in retirement deposit accounts of more than $250,000 |
| 4 | `RCN_1406` | TOTAL, PAST DUE 30-89 DAYS, ACCRUING | Total loans and leases past due 30–89 days and still accruing interest |
| 5 | `RCRII_S442` | LOANS AND LEASES WEIGHT CATEGORY | Loans and leases allocated to a risk-weight category |
| 6 | `log_assets` | (derived) | `log(1 + total assets)` — bank size |
| 7 | `RCRII_D957` | TOTAL CASH AND BALANCES DUE FROM DEP | Total cash and balances due from depository institutions |
| 8 | `RC_5369` | LOANS AND LEASES HELD FOR SALE | Loans and leases held for sale |
| 9 | `RCO_G472` | SUB NOTES & DBNTR MAT GT 5 YR | Subordinated notes and debentures with remaining maturity over 5 years |
| 10 | `RCRII_H300` | EF FUND CONTRBTNS CNTRL CNTRPRTIES | Default fund contributions to central counterparties |
| 11 | `RCRII_S449` | TOTAL LOANS AND R ON NONACCRUAL | Total loans and receivables on nonaccrual status |
| 12 | `RCO_G468` | UNSECURED OTHR BRW MAT GT 5 YR | Unsecured other borrowings with remaining maturity over 5 years |
| 13 | `RCN_F663` | LN MOD FIN DFCLTY SCD 1-4 RES NONACC | Loan modifications to borrowers in financial difficulty, secured by 1–4 family residential properties, on nonaccrual |
| 14 | `RCRII_S431` | TOTAL LOANS AND OTHER EXPOSURES | Total loans and other credit exposures |
| 15 | `RCN_5459` | ALL OTHER LOANS-PAST DU 30-89 DAYS | All other loans past due 30–89 days |
| 16 | `RCN_1407` | TOTAL, PAST DUE 90 OR MORE, ACCRUING | Total loans and leases past due 90 days or more and still accruing |
| 17 | `RI_4302` | APPLICABLE INCOME TAXES | Applicable income taxes on the income statement |
| 18 | `RCN_1403` | TOTAL, NONACCRUAL | Total loans and leases on nonaccrual status |
| 19 | `RCK_3368` | QTLY AVG OF TOTAL ASSETS | Quarterly average of total assets |
| 20 | `RCN_5461` | ALL OTHER LOANS-NONACCRUAL | All other loans on nonaccrual status |
| 21 | `RC_2200` | TOTAL DEPOSITS | Total deposits |
| 22 | `RCCI_2107` | OBLGS OF ST&POLITICAL SUBDVS IN U.S. | Obligations of states and political subdivisions in the U.S. |
| 23 | `RCCI_A568` | CLSD-END LNS SECD 1ST LIENS OVR 15 Y | Closed-end loans secured by first liens on 1–4 family residential properties, maturity over 15 years |
| 24 | `RCL_F164` | 1-4  FAM RES CNSTRCTN LN COMMITMNTS | Commitments to fund 1–4 family residential construction loans |
| 25 | `RCK_3465` | QUARTERLY AVG OF LNS SECD BY 1-4 FAM | Quarterly average of loans secured by 1–4 family residential properties |
| 26 | `RCN_B575` | PAST DUE(30-89DA): CREDIT CARD LOANS | Credit card loans past due 30–89 days |
| 27 | `RCRII_S585` | NOTIONAL PRINCIPONE YEAR OR LESS | Notional principal of derivative contracts with one year or less remaining maturity |
| 28 | `RIBI_C894` | OTHR CNSTRCTN LN & LAND DEV RECOVERI | Recoveries on other construction and land development loans |
| 29 | `RCE_B550` | NONTRANSACTION ACCTS: IPC DEPOSITS | Nontransaction accounts: deposits of individuals, partnerships and corporations |
| 30 | `RCCI_F158` | LN SECURED BY 1-4 FAM RES CONSTRUCTI | Loans secured by 1–4 family residential construction and land development |
| 31 | `RCF_B556` | ACCRUED INTEREST RECEIVABLE | Accrued interest receivable |
| 32 | `RCRI_P742` | COMMON EQUITY TIER 1 CAPITAL | Common equity tier 1 capital |
| 33 | `RCO_F045` | AMT RETIRE DEP ACCNT $250K OR LESS | Amount in retirement deposit accounts of $250,000 or less |
| 34 | `RCF_K270` | LIFE INS ASSET HYBRID ACCNT | Life insurance assets held in hybrid accounts |
| 35 | `RCCI_1590` | LOANS TO FINANCE AGRICULTURAL PROD | Loans to finance agricultural production and other loans to farmers |
| 36 | `RCCI_HK25` | TOTAL LN MOD FIN DFCLTY | Total loan modifications to borrowers experiencing financial difficulty (what troubled debt restructurings became under ASU 2022-02) |
| 37 | `RCRII_S413` | TOTAL LOANS AND TGAGE EXPOSURES | Total loans and mortgage exposures |
| 38 | `RCE_HK13` | MATURITY AND REPRICING DATA FOR TIME | Maturity and repricing data for time deposits |
| 39 | `RCRII_S547` | OVER-THE-COUNTER WEIGHT CATEGORY | Over-the-counter derivatives allocated to a risk-weight category |
| 40 | `RCRII_S529` | UNUSED COMMITMEN WEIGHT CATEGORY | Unused commitments allocated to a risk-weight category |
| 41 | `RCRII_D958` | CASH AND BALANCES DUE FROM DEPOSITOR | Cash and balances due from depository institutions |
| 42 | `RCE_2215` | TOTAL TRANSACTIONS ACCOUNTS | Total transaction accounts |
| 43 | `RCB_G323` | MBS OTHR OTHR RES MBS AFS FV | Other residential mortgage-backed securities, available-for-sale, at fair value |
| 44 | `RCN_K114` | LN MOD FIN DFCLTY SECD NONFARM OWN P | Loan modifications to borrowers in financial difficulty, secured by owner-occupied nonfarm nonresidential properties, past due |
| 45 | `RCRII_A222` | EXCESS AACL | Excess adjusted allowance for credit losses (the ALLL under CECL) |
| 46 | `RCRII_G607` | COMMERCIAL AND SIMILAR LETTERS OF CR | Commercial and similar letters of credit |
| 47 | `RCO_F051` | AMT OF DEP ACCNT MORE THAN $250K | Amount in deposit accounts of more than $250,000 |
| 48 | `RC_2145` | PREMISES&FIXED ASSETS(INCL ROU) | Premises and fixed assets, including right-of-use lease assets |
| 49 | `RCE_B552` | NONTRANSACT ACCT: CB'S & DI'S IN U.S | Nontransaction accounts: deposits of commercial banks and other depository institutions in the U.S. |
| 50 | `RCCI_K161` | LN MOD FIN DFCLTY LAND DEV NFARM NRE | Loan modifications to borrowers in financial difficulty, secured by land development and other nonfarm nonresidential property |

---

## Reading the ranking

The order is XGBoost gain, which measures how much each field improved the trees'
objective — not a coefficient and not a direction. A field at rank 3 is not "three times
more important" than one at rank 9, and nothing here says whether a higher value means
more or less risk. The GP that ships uses all 50 jointly and forms its estimate from
overall similarity to historically distressed banks, so no single field has a threshold
that can be read off it.

**Three of these share a source with the label.** `RC_2200` (total deposits, rank 21)
feeds the label's deposit leg; `RCN_1403` and `RCN_1407` (nonaccrual and 90+ days past
due, ranks 18 and 16) feed its NPL leg. They are not leakage — the label is measured
four quarters forward and these are the current quarter's values — but the control run
that drops all label-adjacent fields is in REPORT §3.1 and §8.1.

**Regenerating this list**: run `models.py` and read the first 50 names from
`fixed_order.json`. Same panel and same seed reproduce it exactly.
