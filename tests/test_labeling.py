from pathlib import Path

import pytest

from pipeline.labeling import (
    LABELS,
    build_model_meta,
    parse_prompt_version,
    render_prompt,
    validate_label,
)

PROMPTS = Path(__file__).resolve().parent.parent / "evals" / "prompts"


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


def test_parse_prompt_version_reads_the_marker():
    assert parse_prompt_version("<!-- prompt_version: v3 -->\nlabel this") == "v3"
    assert parse_prompt_version("<!--prompt_version:v2-->") == "v2"


def test_parse_prompt_version_missing_marker_raises():
    """Silently defaulting would write a false prompt_version into
    model_meta — the provenance is the whole point of the field."""
    with pytest.raises(ValueError, match="prompt_version"):
        parse_prompt_version("You label news articles.")


def test_every_shipped_prompt_declares_its_version():
    """Guards the round-trip: kaggle_llama_labeling reads the version from
    whichever prompt file it is pointed at."""
    for path in sorted(PROMPTS.glob("*.md")):
        assert parse_prompt_version(path.read_text(encoding="utf-8"))


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
