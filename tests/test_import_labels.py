import pytest

from pipeline.import_labels import parse_labels


def test_parse_valid_row():
    rows = [
        {
            "raw_item_id": "42",
            "label": "negative",
            "model_meta": '{"model": "llama", "prompt_version": "v2"}',
        }
    ]
    assert parse_labels(rows) == [
        {
            "raw_item_id": 42,
            "label": "negative",
            "model_meta": {"model": "llama", "prompt_version": "v2"},
        }
    ]


def test_parse_normalizes_label():
    rows = [{"raw_item_id": "1", "label": "Positive\n", "model_meta": "{}"}]
    assert parse_labels(rows)[0]["label"] == "positive"


def test_parse_empty_meta_becomes_dict():
    rows = [{"raw_item_id": "1", "label": "neutral", "model_meta": ""}]
    assert parse_labels(rows)[0]["model_meta"] == {}


def test_parse_rejects_invalid_label():
    rows = [{"raw_item_id": "1", "label": "maybe", "model_meta": "{}"}]
    with pytest.raises(ValueError):
        parse_labels(rows)


def test_parse_all_or_nothing():
    # one bad row aborts the whole file (no partial import)
    rows = [
        {"raw_item_id": "1", "label": "positive", "model_meta": "{}"},
        {"raw_item_id": "2", "label": "??bad", "model_meta": "{}"},
    ]
    with pytest.raises(ValueError):
        parse_labels(rows)
