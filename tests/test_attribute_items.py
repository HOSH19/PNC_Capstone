import pytest

from pipeline.attribute_items import verdict_for
from pipeline.attribution import build_patterns

PATTERNS = build_patterns(
    [{"bank_id": "jpm", "bank_legal_name": "JPMorgan Chase Bank", "aliases": []}]
)


def test_gdelt_rows_pass_only_when_the_title_names_the_bank():
    row = {
        "source": "gdelt",
        "bank_id": "jpm",
        "title": "JPMorgan Chase Bank posts loss",
    }
    off_topic = {**row, "title": "Netflix tops most-watched list"}
    assert verdict_for(row, PATTERNS) is True
    assert verdict_for(off_topic, PATTERNS) is False


def test_ingest_matched_sources_pass_regardless_of_title():
    """An 8-K's title rarely names the filer; the CIK already did."""
    row = {"source": "edgar", "bank_id": "jpm", "title": "Form 8-K: Material event"}
    assert verdict_for(row, PATTERNS) is True


def test_unknown_source_raises_instead_of_defaulting():
    """True would let a future query-fetched source bypass the gate silently;
    False would zero a legitimate one out of the rollup."""
    with pytest.raises(RuntimeError, match="no attribution rule"):
        verdict_for({"source": "newsapi", "bank_id": "jpm", "title": "x"}, PATTERNS)
