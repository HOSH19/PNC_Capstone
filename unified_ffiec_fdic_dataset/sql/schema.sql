CREATE TABLE dim_bank (
    fdic_cert_number INTEGER PRIMARY KEY,
    rssd_id INTEGER,
    bank_name TEXT NOT NULL,
    city TEXT,
    state TEXT,
    active_status BOOLEAN,
    total_assets REAL,
    charter_type TEXT,
    established_date DATE,
    last_updated DATE
);

CREATE TABLE fact_call_report (
    fdic_cert_number INTEGER NOT NULL,
    rssd_id INTEGER,
    report_date DATE NOT NULL,
    total_assets REAL,
    total_deposits REAL,
    tier1_capital_ratio REAL,
    total_capital_ratio REAL,
    npl_ratio REAL,
    loan_loss_allowance_ratio REAL,
    liquidity_ratio REAL,
    securities_unrealized_loss REAL,
    cre_loans REAL,
    PRIMARY KEY (fdic_cert_number, report_date),
    FOREIGN KEY (fdic_cert_number) REFERENCES dim_bank(fdic_cert_number)
);

CREATE TABLE fact_distress_event (
    fdic_cert_number INTEGER NOT NULL,
    failure_date DATE NOT NULL,
    bank_name TEXT,
    city TEXT,
    state TEXT,
    acquiring_institution TEXT,
    event_type TEXT,
    distress_label INTEGER NOT NULL DEFAULT 1,
    PRIMARY KEY (fdic_cert_number, failure_date),
    FOREIGN KEY (fdic_cert_number) REFERENCES dim_bank(fdic_cert_number)
);

CREATE TABLE fact_bank_quarter (
    fdic_cert_number INTEGER NOT NULL,
    quarter_end_date DATE NOT NULL,
    total_assets REAL,
    total_deposits REAL,
    tier1_capital_ratio REAL,
    total_capital_ratio REAL,
    npl_ratio REAL,
    loan_loss_allowance_ratio REAL,
    liquidity_ratio REAL,
    securities_unrealized_loss REAL,
    cre_loans REAL,
    distress_within_4q INTEGER NOT NULL DEFAULT 0,
    distress_within_8q INTEGER NOT NULL DEFAULT 0,
    days_to_distress INTEGER,
    PRIMARY KEY (fdic_cert_number, quarter_end_date),
    FOREIGN KEY (fdic_cert_number) REFERENCES dim_bank(fdic_cert_number)
);
