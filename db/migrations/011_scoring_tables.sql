-- 011_scoring_tables.sql — Phase-2 scoring contract (see scoring/DESIGN.md).
-- item_label: labeling-stage output, one row per (item, labeler) so
--   champion-challenger and human verdicts coexist as separate rows.
-- item_score: serving-stage output, latest score only, one row per item.
-- label_source is a CHECK (not an enum) so adding a labeler later is the
--   same one-line migration pattern as raw_item.source (002).

CREATE TABLE item_label (
    id           bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    raw_item_id  bigint NOT NULL REFERENCES raw_item(id),
    label        text NOT NULL CHECK (label IN ('positive', 'negative', 'neutral')),
    label_source text NOT NULL CHECK (label_source IN ('llama_kaggle', 'gemini', 'human')),
    model_meta   jsonb NOT NULL DEFAULT '{}',  -- model id, prompt version, quantization
    labeled_at   timestamptz NOT NULL DEFAULT now(),
    UNIQUE (raw_item_id, label_source)
);

CREATE TABLE item_score (
    raw_item_id   bigint PRIMARY KEY REFERENCES raw_item(id),
    label         text NOT NULL CHECK (label IN ('positive', 'negative', 'neutral')),
    probs         jsonb,                       -- 3-class probabilities
    keywords      text[],                      -- dashboard explainability, NULL until v2
    model_version text NOT NULL,
    scored_at     timestamptz NOT NULL DEFAULT now()
);

-- Daily batch queue scan: WHERE finbert_status = 'pending' on raw_item.
CREATE INDEX idx_raw_item_finbert_pending
    ON raw_item (id) WHERE finbert_status = 'pending';
