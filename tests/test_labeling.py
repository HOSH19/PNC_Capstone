import pytest

from pipeline.labeling import (
    LABELS,
    build_model_meta,
    render_prompt,
    validate_label,
)


def test_render_prompt_inserts_article():
    out = render_prompt("Bank X fails", "before {{ARTICLE}} after")
    assert out == "before Bank X fails after"


def test_render_prompt_missing_placeholder_raises():
    with pytest.raises(ValueError):
        render_prompt("x", "no placeholder here")


def test_validate_label_normalizes():
    assert validate_label("Positive\n") == "positive"


def test_validate_label_rejects_unknown():
    with pytest.raises(ValueError):
        validate_label("maybe")


def test_labels_are_the_three_classes():
    assert LABELS == ("positive", "negative", "neutral")


def test_build_model_meta_shape():
    meta = build_model_meta(
        "meta-llama/Llama-3.1-8B-Instruct", "awq-4bit", "v1", "2026-07-21"
    )
    assert meta == {
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "quantization": "awq-4bit",
        "prompt_version": "v1",
        "run_date": "2026-07-21",
    }
