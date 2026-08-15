import json

import pytest

from pipeline.score_finbert import to_score_row, triage


def _gdelt(id, title="Bank posts a loss", language="English"):
    return {
        "id": id,
        "source": "gdelt",
        "title": title,
        "text_excerpt": None,
        "meta": {"language": language},
    }


def _edgar(id, title="Bank 8-K", excerpt="Material event."):
    return {
        "id": id,
        "source": "edgar",
        "title": title,
        "text_excerpt": excerpt,
        "meta": {},
    }


def test_triage_uses_the_same_filter_training_did():
    """Serving a different distribution than the model was trained on is the
    skew the shared eligibility module exists to prevent."""
    scorable, skipped = triage([_gdelt(1), _gdelt(2, language="Spanish"), _edgar(3)])
    assert [r["id"] for r in scorable] == [1, 3]
    assert [(r["id"], r["reason"]) for r in skipped] == [(2, "non_english")]


def test_scorable_rows_carry_the_text_the_model_will_see():
    scorable, _ = triage([_edgar(1)])
    assert scorable[0]["text"] == "Bank 8-K\nMaterial event."


def test_empty_text_is_skipped_not_scored():
    _, skipped = triage([_gdelt(1, title=None)])
    assert skipped[0]["reason"] == "empty_text"


def test_unknown_source_raises_instead_of_skipping_everything():
    """A missing adapter is a deploy mistake; silently skipping every row of
    that source would look like the source simply had no eligible rows."""
    with pytest.raises(RuntimeError, match="no eligibility adapter"):
        triage([{"id": 9, "source": "newsapi", "title": "x", "meta": {}}])


def test_probs_are_serialised_for_jsonb_not_dropped():
    """The aggregation layer is expected to lower the directional threshold
    rather than take argmax, which needs the distribution to survive."""
    row = to_score_row(
        7,
        "negative",
        {"negative": 0.61, "neutral": 0.3, "positive": 0.09},
        "finbert-ft-2026-08-09",
    )
    assert row["raw_item_id"] == 7 and row["label"] == "negative"
    assert json.loads(row["probs"])["negative"] == 0.61
    assert row["model_version"] == "finbert-ft-2026-08-09"
