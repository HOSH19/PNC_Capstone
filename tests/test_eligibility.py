import pytest

from pipeline.eligibility import Result, UnknownSource, check, is_syndication_noise


def _gdelt(title="Bank X under investigation", language="English", **kw):
    row = {
        "source": "gdelt",
        "title": title,
        "text_excerpt": None,
        "meta": {"language": language},
    }
    row.update(kw)
    return row


def _edgar(title="Holding Co 8-K", excerpt="Material event disclosed."):
    return {"source": "edgar", "title": title, "text_excerpt": excerpt, "meta": {}}


def test_gdelt_english_eligible():
    assert check(_gdelt()) == Result(True, text="Bank X under investigation")


def test_gdelt_non_english_skipped():
    r = check(_gdelt(language="Spanish"))
    assert r.eligible is False
    assert r.reason == "non_english"


def test_edgar_eligible_joins_title_and_excerpt():
    r = check(_edgar())
    assert r.eligible is True
    assert r.text == "Holding Co 8-K\nMaterial event disclosed."


def test_gdelt_empty_title_skipped():
    r = check(_gdelt(title=None))
    assert r.eligible is False
    assert r.reason == "empty_text"


def test_unknown_source_raises():
    with pytest.raises(UnknownSource):
        check({"source": "foo", "title": "x", "meta": {}})


def test_noise_hook_present_but_deferred():
    # v1: 훅은 존재하되 항상 False (predicate는 export dry-run 후 확정)
    assert is_syndication_noise(_gdelt()) is False


def test_text_for_gdelt_returns_title():
    from pipeline.eligibility import text_for

    assert text_for(_gdelt(title="Deposit run at X")) == "Deposit run at X"


def test_text_for_edgar_joins_title_and_excerpt():
    from pipeline.eligibility import text_for

    assert text_for(_edgar()) == "Holding Co 8-K\nMaterial event disclosed."


def test_text_for_unknown_source_raises():
    from pipeline.eligibility import UnknownSource, text_for

    with pytest.raises(UnknownSource):
        text_for({"source": "foo", "title": "x", "meta": {}})
