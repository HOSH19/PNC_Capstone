"""Roll item scores up to bank_sentiment_quarter (migration 016).

Full recompute every run, in one transaction: the table is ~104 banks x ~27
quarters, so rebuilding it is cheaper than tracking which quarters a batch of
new scores touched, and it makes threshold changes a re-run instead of a
migration.

Only rows that passed the attribution gate (raw_item.attributed, migration
015) contribute signal; ungated rows still count in n_items so the funnel
stays visible. Directional counts use p(class) >= threshold rather than the
argmax label — the model under-calls direction (scoring/DESIGN.md, "Training
on a champion that missed the criteria"), and the tunable threshold is the
mitigation that decision depends on. The default is argmax-equivalent-ish 0.5
until the backtest picks a better one; whatever was used is stored per row.

Scheduled in .github/workflows/score.yml after scoring. Safe to re-run any
time.

    python -m pipeline.aggregate_sentiment --threshold 0.5
"""

import argparse
import sys
import time

from pipeline import db

# ponytail: full DELETE + re-INSERT each run; go incremental only if the
# table ever outgrows a single cheap statement (it is bounded by banks x
# quarters, so it should not).
AGGREGATE_SQL = """
WITH item AS (
    SELECT r.bank_id,
           (date_trunc('quarter', r.published_at)
              + interval '3 months' - interval '1 day')::date AS quarter_end_date,
           r.attributed,
           s.raw_item_id IS NOT NULL                          AS scored,
           (s.probs ->> 'negative')::numeric                  AS p_neg,
           (s.probs ->> 'positive')::numeric                  AS p_pos,
           s.model_version
    FROM raw_item r
    LEFT JOIN item_score s ON s.raw_item_id = r.id
    WHERE r.published_at IS NOT NULL
)
INSERT INTO bank_sentiment_quarter
    (bank_id, quarter_end_date, fdic_cert_number,
     n_items, n_attributed, n_scored,
     n_negative, n_positive, mean_p_negative, mean_p_positive,
     threshold, model_version)
SELECT i.bank_id,
       i.quarter_end_date,
       b.fdic_cert,
       count(*),
       count(*) FILTER (WHERE i.attributed),
       count(*) FILTER (WHERE i.attributed AND i.scored),
       count(*) FILTER (WHERE i.attributed AND i.p_neg >= %(threshold)s),
       count(*) FILTER (WHERE i.attributed AND i.p_pos >= %(threshold)s),
       avg(i.p_neg) FILTER (WHERE i.attributed),
       avg(i.p_pos) FILTER (WHERE i.attributed),
       %(threshold)s,
       string_agg(DISTINCT i.model_version, ',')
FROM item i
JOIN bank b USING (bank_id)
GROUP BY i.bank_id, i.quarter_end_date, b.fdic_cert
"""


def recompute(conn, threshold: float) -> int:
    """Rebuild the whole table under `threshold`. Returns rows written."""
    with conn.cursor() as cur:
        cur.execute("DELETE FROM bank_sentiment_quarter")
        cur.execute(AGGREGATE_SQL, {"threshold": threshold})
        written = cur.rowcount
    conn.commit()
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="p(class) at or above this counts as directional; "
        "the backtest owns tuning it",
    )
    args = ap.parse_args()

    started = time.monotonic()
    conn = db.connect()
    try:
        written = recompute(conn, args.threshold)
        db.write_heartbeat(
            conn,
            "aggregate_sentiment",
            written,
            written,
            time.monotonic() - started,
            True,
        )
    except Exception:
        try:
            conn.rollback()
            db.write_heartbeat(
                conn, "aggregate_sentiment", 0, 0, time.monotonic() - started, False
            )
        except Exception:
            pass  # never mask the original failure
        raise
    finally:
        conn.close()
    print(f"{written} bank-quarters, threshold {args.threshold}", file=sys.stderr)


if __name__ == "__main__":
    main()
