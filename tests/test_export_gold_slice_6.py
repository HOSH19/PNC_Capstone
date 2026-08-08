from pipeline.export_new_gold_slice import (
    build_alias_patterns,
    matches_own_bank,
    select_candidates,
    stratified_sample,
)


def _row(id, bank_id, title, label="neutral", source="gdelt"):
    return {
        "id": id,
        "source": source,
        "bank_id": bank_id,
        "title": title,
        "text_excerpt": None,
        "label": label,
    }


def test_word_boundary_does_not_match_substring():
    # "citi" is a substring of "citizens" -- must not match.
    patterns = build_alias_patterns(
        [
            {"bank_id": "citi", "aliases": ["Citi", "Citigroup"]},
            {"bank_id": "cfg", "aliases": ["Citizens Bank", "Citizens Financial Group"]},
        ]
    )
    assert not matches_own_bank("Citizens Bank posts earnings", "citi", patterns)
    assert matches_own_bank("Citi posts earnings", "citi", patterns)
    assert matches_own_bank("Citizens Bank posts earnings", "cfg", patterns)


def test_matches_own_bank_only_checks_that_banks_patterns():
    patterns = build_alias_patterns(
        [
            {"bank_id": "citi", "aliases": ["Citi"]},
            {"bank_id": "cfg", "aliases": ["Citizens Bank"]},
        ]
    )
    # title genuinely mentions cfg, but row is tagged citi -- should not match.
    assert not matches_own_bank("Citizens Bank posts earnings", "citi", patterns)


def test_select_candidates_excludes_seen_ids_and_unmatched_titles():
    patterns = build_alias_patterns([{"bank_id": "citi", "aliases": ["Citi"]}])
    rows = [
        _row(1, "citi", "Citi posts record profit"),  # keep
        _row(2, "citi", "Citi posts record profit"),  # excluded id
        _row(3, "citi", "Dow falls on Iran conflict"),  # no alias match
    ]
    selected = select_candidates(rows, patterns, excluded_ids={2})
    assert [r["id"] for r in selected] == [1]


def test_stratified_sample_respects_counts_and_is_deterministic():
    candidates = (
        [_row(i, "citi", "Citi t", label="negative") for i in range(5)]
        + [_row(100 + i, "citi", "Citi t", label="positive") for i in range(5)]
        + [_row(200 + i, "citi", "Citi t", label="neutral") for i in range(5)]
    )
    counts = {"negative": 2, "positive": 2, "neutral": 1}
    sampled = stratified_sample(candidates, counts, seed=6)
    by_label = {}
    for r in sampled:
        by_label.setdefault(r["label"], 0)
        by_label[r["label"]] += 1
    assert by_label == {"negative": 2, "positive": 2, "neutral": 1}
    # deterministic for a fixed seed
    assert sampled == stratified_sample(candidates, counts, seed=6)


def test_stratified_sample_raises_on_shortfall():
    candidates = [_row(1, "citi", "Citi t", label="negative")]
    try:
        stratified_sample(candidates, {"negative": 5}, seed=1)
        assert False, "expected ValueError"
    except ValueError as e:
        assert "negative: have 1, need 5" in str(e)
