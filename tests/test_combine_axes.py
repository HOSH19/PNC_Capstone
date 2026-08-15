from datetime import UTC, datetime

from pipeline.aggregate_sentiment_files import aggregate, quarter_end
from pipeline.combine_axes import TIERS, combine_row, ladder


def _item(bank="jpm", month=2, p_neg=0.1, p_pos=0.1, attributed=True):
    return {
        "bank_id": bank,
        "published_at": datetime(2020, month, 15, tzinfo=UTC),
        "attributed": attributed,
        "p_negative": p_neg,
        "p_positive": p_pos,
        "model_version": "finbert-ft-2026-08-09",
    }


def test_quarter_end_matches_the_sql_bucketing():
    assert quarter_end(datetime(2020, 2, 15)) == "2020-03-31"
    assert quarter_end(datetime(2024, 12, 31)) == "2024-12-31"


def test_unattributed_items_carry_no_signal():
    """The gate is load-bearing for the backtest; the parquet keeps ungated
    rows only so the gate stays tunable without re-scoring."""
    rows = aggregate([_item(), _item(attributed=False)], threshold=0.5)
    assert rows[0]["n_scored"] == 1


def test_directional_counts_are_threshold_counts_not_argmax():
    rows = aggregate([_item(p_neg=0.6), _item(p_neg=0.4)], threshold=0.5)
    assert rows[0]["n_negative"] == 1 and rows[0]["neg_share"] == 0.5


def test_ladder_matches_the_dashboard_rule():
    # (score, neg_share) -> tier, per compute_status in dashboard/app.py
    assert ladder(20, 0.5) == "Imminent Disruption"
    # The fundamentals floor: calm news must never rate a bank better than
    # having no news at all would (compute_status as written does exactly
    # that — score 20 + calm news -> Watch, score 20 + no news -> Elevated).
    assert ladder(20, 0.0) == "Elevated Risk"
    assert ladder(70, 0.5) == "Elevated Risk"
    assert ladder(85, 0.5) == "Watch"
    assert ladder(95, 0.5) == "Watch"  # negative alone is at least Watch
    assert ladder(95, 0.0) == "Stable"


def test_no_sentiment_caps_at_elevated_risk():
    """The dashboard's None branch: without a sentiment read the 'both axes'
    tier is unreachable."""
    assert ladder(10, None) == "Elevated Risk"
    assert ladder(85, None) == "Watch"
    assert ladder(95, None) == "Stable"


def test_risk_score_orders_within_tier_without_crossing_tiers():
    fund_risky = {
        "fdic_cert_number": 1,
        "quarter_end_date": "2020-03-31",
        "score": 20,
        "distress_prob": 0.9,
    }
    fund_calm = {**fund_risky, "fdic_cert_number": 2, "distress_prob": 0.2}
    hi = combine_row(fund_risky, 0.5, {})
    lo = combine_row(fund_calm, 0.5, {})
    assert hi["status"] == lo["status"] == "Imminent Disruption"
    assert hi["risk_score"] > lo["risk_score"]
    assert lo["risk_score"] > TIERS.index("Elevated Risk") + 1.0 - 0.001
