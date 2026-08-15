import pytest

from pipeline.merge_labels import merge_labels


def _row(id, label):
    return {"raw_item_id": id, "label": label, "model_meta": "{}"}


def test_only_the_named_class_comes_from_the_override():
    base = [_row("1", "neutral"), _row("2", "positive"), _row("3", "neutral")]
    override = {"1": "negative", "2": "negative", "3": "positive"}
    merged = merge_labels(base, override, "negative")
    out = {r["raw_item_id"]: r["label"] for r in merged}
    assert out["1"] == "negative"  # override says negative -> taken
    assert out["2"] == "negative"  # taken even over a directional base label
    assert out["3"] == "neutral"  # override's positive is ignored, base wins


def test_base_is_kept_where_override_agrees_on_nothing():
    base = [_row("1", "positive")]
    out = merge_labels(base, {"1": "neutral"}, "negative")
    assert out[0]["label"] == "positive"


def test_missing_override_row_raises():
    """A half-merged champion would be worse than a failed merge: the two
    runs would have seen different corpora and nothing would say so."""
    with pytest.raises(ValueError, match="no override label"):
        merge_labels([_row("9", "neutral")], {}, "negative")


def test_invalid_only_label_raises():
    with pytest.raises(ValueError):
        merge_labels([_row("1", "neutral")], {"1": "neutral"}, "bad")
