"""Combine the fundamentals and sentiment axes into the four-tier ladder.

The rule is Rita's compute_status (dashboard/app.py, PR #15), which encodes
the mentor-agreed classification of 2026-07-12 — a rule ladder, not a
weighted average:

    score < 30 AND sentiment Negative  -> Imminent Disruption
    score <= 80 AND sentiment Negative -> Elevated Risk
    sentiment Negative OR score < 90   -> Watch
    otherwise                          -> Stable

Banks with no sentiment read fall back to fundamentals-only and cap at
Elevated Risk, exactly as the dashboard does. All cutoffs are CLI flags
because none of them are team-agreed yet — the dashboard comment says so —
and picking them IS the backtest's job (run it across a sweep and compare).

Output is the backtest harness's format (evals/backtest.py):
fdic_cert_number, quarter_end_date, risk_score, model_version. risk_score
is tier + distress_prob — the ladder is the semantics, but four tiers over
~500 test quarters is nearly all ties, so the GP's distress probability
orders banks within a tier. A `status` column rides along for reading.

Fundamentals come from bank_index_score in the database (Ming's table holds
2019Q4-2026Q1); sentiment from aggregate_sentiment_files.py's CSV.

    python -m pipeline.combine_axes \\
        --sentiment corpus/sentiment_quarter.csv --out corpus/combined.csv
"""

import argparse
import csv
import sys

from pipeline import db

TIERS = ("Stable", "Watch", "Elevated Risk", "Imminent Disruption")

CSV_COLUMNS = [
    "fdic_cert_number",
    "quarter_end_date",
    "risk_score",
    "model_version",
    "status",
]


def ladder(
    score: float | None,
    neg_share: float | None,
    imminent_score: float = 30.0,
    elevated_score: float = 80.0,
    watch_score: float = 90.0,
    neg_share_cut: float = 0.1,
) -> str:
    """Rita's compute_status, with the sentiment label derived here:
    Negative when the quarter's share of p>=threshold negative items
    reaches neg_share_cut. neg_share None means no sentiment read."""
    if neg_share is None:
        if score is None or score >= watch_score:
            return "Stable"
        if score <= elevated_score:
            return "Elevated Risk"
        return "Watch"
    negative = neg_share >= neg_share_cut
    if score is not None and score < imminent_score and negative:
        return "Imminent Disruption"
    # Fundamentals floor — one deliberate deviation from compute_status. As
    # written there, score 20 with CALM news lands on Watch while the same
    # score with NO news lands on Elevated Risk: adding sentiment data
    # improves the rating. Distress quarters are often quiet in the news
    # (the labeler under-calls direction on top), so without this floor the
    # ladder rates a pre-collapse bank better than ignorance would.
    if score is not None and score <= elevated_score:
        return "Elevated Risk"
    if negative or (score is not None and score < watch_score):
        return "Watch"
    return "Stable"


def combine_row(fund: dict, neg_share: float | None, cuts: dict) -> dict:
    status = ladder(float(fund["score"]), neg_share, **cuts)
    return {
        "fdic_cert_number": fund["fdic_cert_number"],
        "quarter_end_date": str(fund["quarter_end_date"]),
        # Tier orders the ladder; distress_prob (in [0,1]) breaks ties
        # within a tier without ever crossing tier boundaries.
        "risk_score": TIERS.index(status) + float(fund["distress_prob"]),
        "model_version": "ladder_v1",
        "status": status,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sentiment", required=True, help="aggregate_sentiment_files CSV")
    ap.add_argument("--out", required=True)
    ap.add_argument("--imminent-score", type=float, default=30.0)
    ap.add_argument("--elevated-score", type=float, default=80.0)
    ap.add_argument("--watch-score", type=float, default=90.0)
    ap.add_argument("--neg-share-cut", type=float, default=0.1)
    args = ap.parse_args()
    cuts = {
        "imminent_score": args.imminent_score,
        "elevated_score": args.elevated_score,
        "watch_score": args.watch_score,
        "neg_share_cut": args.neg_share_cut,
    }

    with open(args.sentiment, newline="") as f:
        neg_shares = {
            (int(r["fdic_cert_number"]), r["quarter_end_date"]): float(r["neg_share"])
            for r in csv.DictReader(f)
            if r["fdic_cert_number"]
        }

    conn = db.connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT fdic_cert_number, quarter_end_date, score, distress_prob
                   FROM bank_index_score"""
            )
            fundamentals = cur.fetchall()
    finally:
        conn.close()

    out = [
        combine_row(
            f,
            neg_shares.get((f["fdic_cert_number"], str(f["quarter_end_date"]))),
            cuts,
        )
        for f in fundamentals
    ]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        w.writerows(out)

    with_sent = sum(1 for r in out if r["status"] != "")
    tiers = {t: sum(1 for r in out if r["status"] == t) for t in TIERS}
    print(
        f"{len(out)} bank-quarters ({len(neg_shares)} with sentiment) "
        f"-> {args.out}\n  {tiers}",
        file=sys.stderr,
    )
    assert with_sent == len(out)


if __name__ == "__main__":
    main()
