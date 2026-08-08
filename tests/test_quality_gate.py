import pytest

from pipeline.quality_gate import (
    chance_corrected,
    compute_gate,
    evaluate_criteria,
    flag_tone_direction,
)


def _gold(id, label, slice="gold_slice_1", source="gdelt", title="Bank news"):
    return {"id": id, "label": label, "slice": slice, "source": source, "title": title}


def test_overall_confusion_and_per_class_both_directions():
    gold = [
        _gold("1", "neutral"),
        _gold("2", "neutral"),
        _gold("3", "positive"),
        _gold("4", "negative"),
    ]
    llama = {"1": "neutral", "2": "positive", "3": "positive", "4": "neutral"}
    gate = compute_gate(gold, llama)
    assert gate["overall"] == {"n": 4, "agree": 2}
    assert gate["confusion"][("positive", "neutral")] == 1
    assert gate["confusion"][("neutral", "negative")] == 1
    pos = gate["per_class"]["positive"]
    assert (pos["diag"], pos["human_n"], pos["llama_n"]) == (1, 1, 2)
    neu = gate["per_class"]["neutral"]
    assert (neu["diag"], neu["human_n"], neu["llama_n"]) == (1, 2, 2)


def test_headline_excludes_stratified_slice_but_strata_keep_it():
    gold = [
        _gold("1", "neutral"),
        _gold("2", "negative", slice="gold_slice_6"),
    ]
    llama = {"1": "neutral", "2": "neutral"}
    gate = compute_gate(gold, llama)
    assert gate["headline"] == {"n": 1, "agree": 1}
    assert gate["overall"] == {"n": 2, "agree": 1}
    assert gate["by_slice"]["gold_slice_6"] == {"n": 1, "agree": 0}


def test_strata_by_source_and_tone_direction_rows():
    gold = [
        _gold("1", "negative", title="Bank explores strategic alternatives"),
        _gold("2", "neutral", source="edgar", title="Holding 8-K"),
    ]
    llama = {"1": "neutral", "2": "neutral"}
    gate = compute_gate(gold, llama)
    assert gate["by_source"]["edgar"] == {"n": 1, "agree": 1}
    assert gate["tone_direction"]["n"] == 1
    assert gate["tone_direction"]["rows"][0]["id"] == "1"
    assert [r["id"] for r in gate["disagreements"]] == ["1"]


def test_missing_llama_label_raises():
    with pytest.raises(ValueError, match="no llama label"):
        compute_gate([_gold("9", "neutral")], {})


def test_chance_corrected_majority_can_beat_agreement():
    """The reason the report exists: high agreement on a neutral-heavy
    sample can still lose to answering `neutral` every time."""
    rows = [
        {"label": "neutral", "llama": "neutral", "agree": True},
        {"label": "neutral", "llama": "neutral", "agree": True},
        {"label": "neutral", "llama": "neutral", "agree": True},
        {"label": "neutral", "llama": "negative", "agree": False},
        {"label": "positive", "llama": "positive", "agree": True},
    ]
    ch = chance_corrected(rows)
    assert ch["agreement"] == 0.8
    assert ch["majority"] == 0.8  # 4 of 5 humans said neutral
    assert 0 < ch["kappa"] < 1
    assert ch["macro_f1"] < ch["agreement"]  # rare classes drag it down


def test_chance_corrected_perfect_and_empty():
    rows = [
        {"label": "neutral", "llama": "neutral", "agree": True},
        {"label": "positive", "llama": "positive", "agree": True},
    ]
    assert chance_corrected(rows)["kappa"] == 1.0
    assert chance_corrected([]) == {"n": 0}


def test_negative_over_call_isolated_with_rating_count():
    gold = [
        _gold("1", "neutral", title="CAE Cut to Underweight at Morgan Stanley"),
        _gold("2", "neutral", title="Man arrested in robbery of a PNC branch"),
        _gold("3", "negative", title="Bank posts loss"),
    ]
    llama = {"1": "negative", "2": "negative", "3": "negative"}
    oc = compute_gate(gold, llama)["negative_over_call"]
    assert [r["id"] for r in oc["rows"]] == ["1", "2"]  # the correct one excluded
    assert oc["rating_pattern"] == 1


def test_criteria_catch_the_degenerate_never_say_negative_labeler():
    """The reason the criteria are paired: a labeler that says `negative`
    once, correctly, scores perfect precision while missing 4 of 5 real
    negatives. Precision alone would pass it; recall must fail it."""
    gold = [_gold(str(i), "negative") for i in range(5)] + [
        _gold(str(5 + i), "neutral") for i in range(20)
    ]
    llama = {str(i): ("negative" if i == 0 else "neutral") for i in range(25)}
    crit = {c["name"]: c for c in evaluate_criteria(compute_gate(gold, llama))}
    assert crit["negative precision"]["actual"] == 1.0
    assert crit["negative precision"]["passed"]
    guard = crit["negative recall — guards the above"]
    assert guard["actual"] == 0.2
    assert not guard["passed"]  # the pair is what rejects this labeler


def test_criteria_report_sample_size_with_each_number():
    gold = [_gold("1", "negative"), _gold("2", "neutral"), _gold("3", "neutral")]
    llama = {"1": "negative", "2": "neutral", "3": "negative"}
    crit = {c["name"]: c for c in evaluate_criteria(compute_gate(gold, llama))}
    assert crit["negative precision"]["n"] == 2  # llama said negative twice
    assert crit["negative recall — guards the above"]["n"] == 1  # human said it once


def test_flag_tone_direction_patterns():
    assert flag_tone_direction("Regulator lifts consent order on bank")
    assert flag_tone_direction("JPMorgan cuts Redwood Trust's price target")
    assert flag_tone_direction("XYZ Capital Purchases 1,200 Shares of PNC")
    assert flag_tone_direction("Bank dividend held steady")
    assert not flag_tone_direction("Bank opens new branch in Ohio")
    assert not flag_tone_direction(None)
