-- Full loader table for CFPB Consumer Complaint Database (Ming).
-- pipeline/loaders/load_cfpb.py. bank_id is resolved app-side by matching
-- the source's free-text `Company` field against bank.aliases /
-- bank_legal_name / holding_name (messy source field -- see
-- DATA_SOURCES.md). Rows with no match are still stored with bank_id NULL
-- rather than dropped: an unmatched company may become trackable later
-- without needing to re-fetch history, and the row is otherwise valid data.

CREATE TABLE cfpb_complaint (
    complaint_id      bigint PRIMARY KEY,
    bank_id           text REFERENCES bank(bank_id),
    company           text NOT NULL,
    date_received     date NOT NULL,
    product           text,
    sub_product       text,
    issue             text,
    sub_issue         text,
    narrative         text,
    state             text,
    zip_code          text,
    submitted_via     text,
    company_response  text,
    timely_response   boolean,
    consumer_disputed text,
    collected_at      timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX idx_cfpb_complaint_bank_date ON cfpb_complaint (bank_id, date_received DESC);
