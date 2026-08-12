import pytest

from pipeline.export_training_set import build_training_set, is_holdings_spam


def _label(id, label="neutral"):
    return {"raw_item_id": id, "label": label, "model_meta": "{}"}


def _batch(id, title="Bank news", source="gdelt", excerpt="", published="2026-07-01"):
    return {
        "raw_item_id": id,
        "source": source,
        "bank_id": "pnc",
        "published_at": f"{published} 12:00:00+00:00",
        "title": title,
        "text_excerpt": excerpt,
    }


def test_hygiene_filters_and_funnel():
    labels = [_label("1"), _label("2"), _label("3"), _label("4")]
    batch = [
        _batch("1"),
        _batch("2", source="edgar", title="Popular, Inc. 10-Q", excerpt=""),
        _batch("3", title="Ibex Wealth Advisors Grows Stake in Micron Technology"),
        _batch("4", title="Commerce Bancshares Downgraded by Wall Street Zen"),
    ]
    train, val, _hold, funnel = build_training_set(labels, batch, {}, holdout_days=0)
    ids = {r["raw_item_id"] for r in train + val}
    assert ids == {"1", "4"}  # contentless edgar + spam dropped, rating row kept
    assert funnel["skipped"] == {"contentless_edgar": 1, "holdings_spam": 1}


def test_dup_title_folded_case_insensitive():
    labels = [_label("1"), _label("2")]
    batch = [_batch("1", title="Same Story"), _batch("2", title="same story")]
    train, val, _hold, funnel = build_training_set(labels, batch, {}, holdout_days=0)
    assert len(train) + len(val) == 1
    assert funnel["skipped"]["dup_title"] == 1


def test_human_override_wins_and_is_counted():
    labels = [_label("1", "neutral"), _label("2", "neutral")]
    batch = [_batch("1"), _batch("2", title="Other news")]
    train, val, _hold, funnel = build_training_set(
        labels, batch, {"1": "negative", "2": "neutral"}, holdout_days=0
    )
    by_id = {r["raw_item_id"]: r["label"] for r in train + val}
    assert by_id["1"] == "negative"
    assert funnel["human_overrides"] == 1  # id 2 agreed, not an override


def test_time_split_last_weeks_go_to_val():
    labels = [_label("1"), _label("2"), _label("3")]
    batch = [
        _batch("1", title="Old", published="2026-05-01"),
        _batch("2", title="Mid", published="2026-06-01"),
        _batch("3", title="New", published="2026-07-01"),
    ]
    train, val, _hold, _f = build_training_set(labels, batch, {}, holdout_days=20)
    assert {r["raw_item_id"] for r in train} == {"1", "2"}
    assert {r["raw_item_id"] for r in val} == {"3"}


def test_edgar_text_includes_excerpt():
    labels = [_label("1")]
    batch = [_batch("1", source="edgar", title="Bank 8-K", excerpt="Material event.")]
    train, val, _hold, _f = build_training_set(labels, batch, {}, holdout_days=0)
    assert (train + val)[0]["text"] == "Bank 8-K\nMaterial event."


def test_missing_batch_row_raises():
    with pytest.raises(ValueError, match="no batch row"):
        build_training_set([_label("9")], [], {}, holdout_days=0)


def test_holdout_rows_leave_train_and_val_but_are_returned():
    """A holdout row kept in val would still be a row the model trained
    against once val is used for model selection — and val's other labels
    are Llama's, so it is the only human-truth evaluation there is."""
    labels = [_label("1"), _label("2"), _label("3")]
    batch = [_batch("1"), _batch("2", title="Second"), _batch("3", title="Third")]
    train, val, hold, funnel = build_training_set(
        labels, batch, {"2": "negative"}, holdout_days=0, holdout_ids=frozenset({"2"})
    )
    assert "2" not in {r["raw_item_id"] for r in train + val}
    assert [r["raw_item_id"] for r in hold] == ["2"]
    assert hold[0]["label"] == "negative"  # human label, not Llama's
    assert funnel["skipped"]["holdout"] == 1
    assert funnel["holdout"]["n"] == 1


def test_holdout_defaults_to_empty():
    labels = [_label("1")]
    train, val, hold, funnel = build_training_set(
        labels, [_batch("1")], {}, holdout_days=0
    )
    assert hold == []
    assert len(train + val) == 1


def test_stratum_split_is_stable_and_halves_slice_6():
    from pipeline.quality_gate import stratum_for

    assert stratum_for("gold_slice_1", 0) == "dev"
    assert stratum_for("gold_slice_2", 0) == "holdout"
    assert stratum_for("gold_slice_6", 24) == "dev"
    assert stratum_for("gold_slice_6", 25) == "holdout"


def test_holdings_spam_patterns():
    assert is_holdings_spam("Acme Wealth LLC Purchases 1,200 Shares of PNC")
    assert is_holdings_spam("L3Harris Technologies $LHX Shares Purchased by Acme LLC")
    assert is_holdings_spam("Sumitomo Has $113.93 Million Stake in Zoetis Inc.")
    assert is_holdings_spam("SkyOak Wealth LLC Invests $204,000 in Intel Corporation")
    assert not is_holdings_spam("Commerce Bancshares Downgraded by Wall Street Zen")
    assert not is_holdings_spam("JPMorgan Cuts Redwood Trust Price Target to $6.00")
    assert not is_holdings_spam("Bank explores strategic alternatives")
    assert not is_holdings_spam(None)
