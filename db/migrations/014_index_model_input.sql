-- 014_index_model_input.sql — fundamentals axis (Ming). Quarterly inputs for
-- the production scorer; the counterpart to 012, which holds its outputs.
--
-- One row per bank-quarter, holding the 50 raw Call Report fields the frozen
-- model consumes. The collector (pipeline/poll_ffiec_model_input.py) writes it
-- from the FFIEC CDR archives; index/fundamentals/score_quarter.py reads it,
-- refits the GP on index/fundamentals/train_sample.parquet, and writes
-- bank_index_score. Nothing else reads this table.
--
-- 50 fields, not the 529 the model was selected from. The full pool is only
-- needed to re-rank features, and extract.py can rebuild it from source when
-- that happens. The list here is exactly frozen_params.json["raw_fields"] —
-- 49 MDRM items plus RC_2170 — and changing it means re-freezing the model,
-- which is a new migration anyway.
--
-- Values are stored RAW, as filed: dollar amounts in thousands. The model
-- divides each by rc_2170 and takes log1p(rc_2170) for size, but that is a
-- modelling step, not a storage format. Storing raw keeps the table a faithful
-- copy of the filing and survives a change in how the ratio is formed.
--
-- Key is (rssd_id, report_date) because that is what FFIEC files under. 29
-- RSSDs map to more than one FDIC certificate across their history (mergers),
-- but (rssd_id, report_date) resolves to exactly one, so fdic_cert_number is
-- carried here as a resolved convenience for the join to 012 and to
-- fact_call_report. It is nullable: a filer with no certificate on record
-- should still be collected and scored, and simply not surface downstream.

CREATE TABLE index_model_input (
    rssd_id            integer NOT NULL,
    report_date        date    NOT NULL,
    fdic_cert_number   integer,           -- resolved at ingest; see header

    rcb_g323           numeric,           -- #43  MBS OTHR OTHR RES MBS AFS FV
    rcci_1590          numeric,           -- #35  LOANS TO FINANCE AGRICULTURAL PROD
    rcci_1797          numeric,           -- #2  REVOLVING, OPEN-END LNS SECD BY 1-4F
    rcci_2107          numeric,           -- #22  OBLGS OF ST&POLITICAL SUBDVS IN U.S.
    rcci_a568          numeric,           -- #23  CLSD-END LNS SECD 1ST LIENS OVR 15 Y
    rcci_f158          numeric,           -- #30  LN SECURED BY 1-4 FAM RES CONSTRUCTI
    rcci_hk25          numeric,           -- #36  TOTAL LOANS RESTRUCTURED IN TROUBLED
    rcci_k161          numeric,           -- #50  LN LAND DEV NFARM NRES SCD OWN OCC N
    rce_2215           numeric,           -- #42  TOTAL TRANSACTIONS ACCOUNTS
    rce_b550           numeric,           -- #29  NONTRANSACTION ACCTS: IPC DEPOSITS
    rce_b552           numeric,           -- #49  NONTRANSACT ACCT: CB'S & DI'S IN U.S
    rce_hk13           numeric,           -- #38  MATURITY AND REPRICING DATA FOR TIME
    rcf_b556           numeric,           -- #31  ACCRUED INTEREST RECEIVABLE
    rcf_k270           numeric,           -- #34  LIFE INS ASSET HYBRID ACCNT
    rck_3368           numeric,           -- #19  QTLY AVG OF TOTAL ASSETS
    rck_3465           numeric,           -- #25  QUARTERLY AVG OF LNS SECD BY 1-4 FAM
    rcl_3814           numeric,           -- #1  UNUSED COMMITMENTS-REVOLVING, ETC.
    rcl_f164           numeric,           -- #24  1-4  FAM RES CNSTRCTN LN COMMITMNTS
    rcn_1403           numeric,           -- #18  TOTAL, NONACCRUAL
    rcn_1406           numeric,           -- #4  TOTAL, PAST DUE 30-89 DAYS, ACCRUING
    rcn_1407           numeric,           -- #16  TOTAL, PAST DUE 90 OR MORE, ACCRUING
    rcn_5459           numeric,           -- #15  ALL OTHER LOANS-PAST DU 30-89 DAYS
    rcn_5461           numeric,           -- #20  ALL OTHER LOANS-NONACCRUAL
    rcn_b575           numeric,           -- #26  PAST DUE(30-89DA): CREDIT CARD LOANS
    rcn_f663           numeric,           -- #13  RSTRCTD LN SCD 1-4 RES NONACCRUAL
    rcn_k114           numeric,           -- #44  LN RSTR SECD NONFARM OWN PD 30 - 89
    rco_f045           numeric,           -- #33  AMT RETIRE DEP ACCNT $250K OR LESS
    rco_f047           numeric,           -- #3  AMT RETIREMT DEP ACCTS MOR THAN $250
    rco_f051           numeric,           -- #47  AMT OF DEP ACCNT MORE THAN $250K
    rco_g468           numeric,           -- #12  UNSECURED OTHR BRW MAT GT 5 YR
    rco_g472           numeric,           -- #9  SUB NOTES & DBNTR MAT GT 5 YR
    rcrii_a222         numeric,           -- #45  EXCESS ALLOWANCE FOR LN&LEASE LOSSES
    rcrii_d957         numeric,           -- #7  TOTAL CASH AND BALANCES DUE FROM DEP
    rcrii_d958         numeric,           -- #41  CASH AND BALANCES DUE FROM DEPOSITOR
    rcrii_g607         numeric,           -- #46  COMMERCIAL AND SIMILAR LETTERS OF CR
    rcrii_h300         numeric,           -- #10  EF FUND CONTRBTNS CNTRL CNTRPRTIES
    rcrii_s413         numeric,           -- #37  TOTAL LOANS AND TGAGE EXPOSURES
    rcrii_s431         numeric,           -- #14  TOTAL LOANS AND OTHER EXPOSURES
    rcrii_s442         numeric,           -- #5  LOANS AND LEASES WEIGHT CATEGORY
    rcrii_s449         numeric,           -- #11  TOTAL LOANS AND R ON NONACCRUAL
    rcrii_s529         numeric,           -- #40  UNUSED COMMITMEN WEIGHT CATEGORY
    rcrii_s547         numeric,           -- #39  OVER-THE-COUNTER WEIGHT CATEGORY
    rcrii_s585         numeric,           -- #27  NOTIONAL PRINCIPONE YEAR OR LESS
    rcri_p742          numeric,           -- #32  COMMON EQUITY TIER 1 CAPITAL
    rc_2145            numeric,           -- #48  PREMISES&FIXED ASSETS(INCL CAP LSES)
    rc_2170            numeric,           -- denominator for every ratio, and the source of log_assets
    rc_2200            numeric,           -- #21  TOTAL DEPOSITS
    rc_5369            numeric,           -- #8  LOANS AND LEASES HELD FOR SALE
    ribi_c894          numeric,           -- #28  OTHR CNSTRCTN LN & LAND DEV RECOVERI
    ri_4302            numeric,           -- #17  APPLICABLE INCOME TAXES

    collected_at       timestamptz NOT NULL DEFAULT now(),

    PRIMARY KEY (rssd_id, report_date)
);

-- The scorer reads one quarter at a time; the dashboard backfill walks history.
CREATE INDEX idx_index_model_input_quarter
    ON index_model_input (report_date DESC);

-- Joining out to 012 and fact_call_report goes through the certificate.
CREATE INDEX idx_index_model_input_cert
    ON index_model_input (fdic_cert_number, report_date DESC);
