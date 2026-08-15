from datetime import UTC, datetime

from pipeline.attribution import build_patterns
from pipeline.backfill_gkg import (
    CSV_COLUMNS,
    alias_pattern,
    csv_rows,
    parse_date,
    to_rows,
)
from pipeline.poll_agency_rss import build_alias_index

BANKS = [
    {
        "bank_id": "pnc",
        "bank_legal_name": "PNC Bank, National Association",
        "holding_name": "PNC Financial Services Group, Inc.",
        "aliases": ["PNC", "PNC Bank"],
        "notes": None,
    },
    {
        "bank_id": "wfc",
        "bank_legal_name": "Wells Fargo Bank, N.A.",
        "holding_name": "Wells Fargo & Company",
        "aliases": ["Wells Fargo"],
        "notes": None,
    },
]
INDEX = build_alias_index(BANKS)


def _gkg(title, url="https://x.test/1", date="20230115123000"):
    return {
        "DATE": date,
        "domain": "x.test",
        "url": url,
        "title": title,
        "V2Tone": "-1.5,1,2,3,4,5,6",
    }


def test_titles_are_unescaped_because_they_are_the_model_input():
    """A GDELT row's title is the entire text the model scores, so an
    entity-escaped one would be scored as literal '&#xF3;' noise."""
    rows, _ = to_rows([_gkg("Wells Fargo cierra su divisi&#xF3;n")], INDEX)
    assert rows[0]["title"] == "Wells Fargo cierra su división"


def test_one_row_per_matched_bank():
    rows, funnel = to_rows([_gkg("Wells Fargo and PNC Bank both report losses")], INDEX)
    assert sorted(r["bank_id"] for r in rows) == ["pnc", "wfc"]
    assert funnel["rows"] == 2
    assert {r["external_id"] for r in rows} == {"https://x.test/1"}


def test_rows_without_a_bank_are_dropped_not_attributed():
    """The BigQuery pre-filter is deliberately broader than the per-bank
    matcher, so unmatched rows arrive and must not be guessed at."""
    rows, funnel = to_rows([_gkg("Some unrelated company reports losses")], INDEX)
    assert rows == []
    assert funnel["skipped"]["no_bank"] == 1


def test_missing_title_or_date_is_skipped():
    rows, funnel = to_rows(
        [_gkg("", url="https://x.test/2"), _gkg("Wells Fargo news", date="bad")],
        INDEX,
    )
    assert rows == []
    assert funnel["skipped"]["no_title"] == 1
    assert funnel["skipped"]["no_date"] == 1


def test_shape_matches_what_the_gdelt_poller_writes():
    """Backfilled rows join the same table as live ones; a divergent shape
    would silently change what eligibility and the model see."""
    rows, _ = to_rows([_gkg("Wells Fargo posts a loss")], INDEX)
    r = rows[0]
    assert r["source"] == "gdelt"
    assert r["text_excerpt"] is None  # GDELT has never carried a body
    assert r["meta"]["language"] == "English"  # eligibility filters on this
    assert r["published_at"] == datetime(2023, 1, 15, 12, 30, tzinfo=UTC)
    assert r["title_hash"]


def test_parse_date_handles_gkg_format_and_junk():
    assert parse_date("20200401093000") == datetime(2020, 4, 1, 9, 30, tzinfo=UTC)
    assert parse_date("") is None
    assert parse_date("not-a-date") is None


def test_alias_pattern_drops_short_aliases_only():
    pattern = alias_pattern(BANKS)
    assert "Wells\\ Fargo" in pattern or "Wells Fargo" in pattern
    assert "PNC Bank" in pattern.replace("\\", "")
    # "PNC" alone is too short to pre-filter on; the per-bank matcher keeps it
    assert not any(part == "PNC" for part in pattern.split("|"))


def test_generic_word_aliases_are_dropped_from_matching():
    """ "Popular" pulled 49% of the 2020 corpus -- Netflix rankings, popular
    shops -- because it is an English word, not a distinctive brand token."""
    from pipeline.backfill_gkg import safe_banks, usable_aliases

    bpop = {
        "bank_id": "bpop",
        "bank_legal_name": "Banco Popular de Puerto Rico",
        "holding_name": "Popular, Inc.",
        "aliases": ["Popular", "Banco Popular"],
        "notes": None,
    }
    assert usable_aliases(bpop) == ["Banco Popular"]

    index = build_alias_index(safe_banks([bpop]))
    rows, funnel = to_rows([_gkg("The most popular TV show on Netflix")], index)
    assert rows == [] and funnel["skipped"]["no_bank"] == 1
    rows, _ = to_rows([_gkg("Banco Popular reports a quarterly loss")], index)
    assert rows[0]["bank_id"] == "bpop"


def test_csv_rows_carry_the_attribution_verdict_lowercase():
    """The file-based corpus has no attribute_items pass to stamp the verdict
    later, so it rides along per row -- lowercase, because csv.writer would
    spell Python bools 'True'/'False'."""
    rows, _ = to_rows([_gkg("Wells Fargo posts a loss")], INDEX)
    out = csv_rows(rows, build_patterns(BANKS))
    assert list(out[0]) == CSV_COLUMNS
    assert out[0]["attributed"] == "true"
    assert out[0]["published_at"] == "2023-01-15T12:30:00+00:00"
    assert out[0]["language"] == "English"


def test_csv_rows_attributed_false_when_title_names_a_different_bank():
    row = {
        "bank_id": "pnc",
        "published_at": datetime(2023, 1, 15, 12, 30, tzinfo=UTC),
        "title": "Wells Fargo posts a loss",
        "url": "https://x.test/1",
        "meta": {"language": "English"},
    }
    out = csv_rows([row], build_patterns(BANKS))
    assert out[0]["attributed"] == "false"


def test_csv_rows_dedup_on_bank_and_url():
    """The DB path leaned on raw_item's ON CONFLICT for this; a CSV has no
    such net, so the dedup moves into memory."""
    rows, _ = to_rows([_gkg("Wells Fargo and PNC Bank both report losses")], INDEX)
    out = csv_rows(rows + rows, build_patterns(BANKS))
    # same url under two banks is two rows; the same (bank_id, url) again is not
    assert sorted(r["bank_id"] for r in out) == ["pnc", "wfc"]
