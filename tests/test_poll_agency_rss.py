"""Contract tests for the agency RSS poller: bank matching and row shaping."""

import pytest

from pipeline import poll_agency_rss as rss


def bank(bank_id="pnc", legal="PNC Bank, National Association",
         holding="PNC Financial Services Group, Inc.",
         aliases=("PNC", "PNC Bank"), notes=None):
    return {"bank_id": bank_id, "bank_legal_name": legal, "holding_name": holding,
            "aliases": list(aliases), "notes": notes}


def match(text, banks):
    return rss.match_banks(text, rss.build_alias_index(banks))


def test_alias_must_appear_as_whole_word():
    banks = [bank("bk", legal="The Bank of New York Mellon",
                  holding="The Bank of New York Mellon Corporation", aliases=["BNY"])]
    assert match("Reserves held at FRBNY were unchanged", banks) == set()
    assert match("BNY reported quarterly results", banks) == {"bk"}


def test_matching_is_case_insensitive():
    assert match("consent order against pnc bank", [bank()]) == {"pnc"}


def test_generic_flagged_bank_matches_holding_name_only():
    banks = [bank("ffbc", legal="First Financial Bank",
                  holding="First Financial Bancorp",
                  aliases=["First Financial"],
                  notes="generic legal name, collides with ffin")]
    assert match("First Financial Bank fined", banks) == set()
    assert match("First Financial Bancorp fined", banks) == {"ffbc"}


def test_holding_name_ending_in_punctuation_still_matches():
    """Regression: \\b cannot anchor a name ending in punctuation ('Inc.',
    'Corp.', 'Bancorp.'), and 16 of the 22 generic-flagged banks' holding_name
    (their ONLY pattern) ends that way -- they could never match. Lookarounds
    + punctuation stripping also let sources that omit the period match."""
    banks = [bank("ffbc", legal="First Financial Bank",
                  holding="First Financial Bancorp.",
                  aliases=["First Financial"], notes="generic")]
    assert match("First Financial Bancorp. fined", banks) == {"ffbc"}
    assert match("FIRST FINANCIAL BANCORP settles claim", banks) == {"ffbc"}
    assert match("First Financial Bancorporation", banks) == set()


def test_bank_with_no_usable_names_is_dropped_not_matched_everywhere():
    banks = [bank("ghost", legal=None, holding=None, aliases=(),
                  notes="generic")]
    assert rss.build_alias_index(banks) == []


def test_title_hash_ignores_case_and_punctuation():
    assert (rss.normalize_title_hash("Fed Announces: New Policy!")
            == rss.normalize_title_hash("fed   announces new policy"))


def entry(title="Statement on PNC Bank", summary="details",
          id="https://fed.gov/pr/1", link="https://fed.gov/pr/1",
          published_parsed=(2026, 7, 1, 12, 0, 0, 1, 182, 0)):
    e = {"title": title, "summary": summary, "link": link,
         "published_parsed": published_parsed}
    if id:
        e["id"] = id
    return e


def test_item_matching_two_banks_emits_one_row_per_bank():
    banks = [bank(), bank("jpm", legal="JPMorgan Chase Bank, National Association",
                          holding="JPMorgan Chase & Co.", aliases=["JPMorgan"])]
    rows = rss.to_rows("fed_all_releases", [entry(title="PNC and JPMorgan respond")],
                       rss.build_alias_index(banks))
    assert {r["bank_id"] for r in rows} == {"pnc", "jpm"}
    for r in rows:
        assert r["source"] == "agency_rss"
        assert r["external_id"] == "https://fed.gov/pr/1"
        assert r["published_at"].year == 2026
        assert r["meta"] == {"feed": "fed_all_releases"}


def test_unmatched_and_idless_items_are_skipped():
    index = rss.build_alias_index([bank()])
    entries = [entry(title="H.4.1 statistical release"),      # no tracked bank
               entry(title="PNC notice", id=None, link=None)]  # nothing to key on
    assert rss.to_rows("fed_h41", entries, index) == []


def test_link_stands_in_for_missing_entry_id_and_excerpt_is_capped():
    index = rss.build_alias_index([bank()])
    [row] = rss.to_rows("fed_h41", [entry(id=None, summary="x" * 5000)], index)
    assert row["external_id"] == "https://fed.gov/pr/1"
    assert len(row["text_excerpt"]) == 4000


def test_bad_feed_is_isolated_and_flags_heartbeat(monkeypatch, fake_db):
    fake_db.banks = [bank()]

    def fake_fetch(feed_key, url):
        if feed_key == "fed_all_releases":
            raise RuntimeError("boom")
        return [entry(id=f"https://fed.gov/{feed_key}")]

    monkeypatch.setattr(rss, "fetch_feed", fake_fetch)

    with pytest.raises(SystemExit, match="fed_all_releases"):
        rss.main()

    assert {r["external_id"] for r in fake_db.rows} == {
        "https://fed.gov/fed_bank_reg_policy", "https://fed.gov/fed_h41"}
    job, seen, inserted, ok = fake_db.heartbeats[-1]
    assert (job, inserted, ok) == ("poll_agency_rss", 2, False)
    assert fake_db.rollbacks == 1
