from pipeline.attribution import (
    build_patterns,
    counts_for_bank,
    gate_rows,
    usable_aliases,
)

BANKS = [
    {
        "bank_id": "citi",
        "bank_legal_name": "Citibank, N.A.",
        "holding_name": "Citigroup Inc.",
        # As the real seed spells them — "Citigroup" is listed, which is
        # what lets the derived single token survive the stripping rule.
        "aliases": ["Citigroup", "Citibank", "Citi"],
        "notes": None,
    },
    {
        "bank_id": "cfg",
        "bank_legal_name": "Citizens Bank, N.A.",
        "holding_name": "Citizens Financial Group, Inc.",
        "aliases": ["Citizens Bank"],
        "notes": None,
    },
    {
        "bank_id": "bpop",
        "bank_legal_name": "Banco Popular de Puerto Rico",
        "holding_name": "Popular, Inc.",
        "aliases": ["Popular", "Banco Popular"],
        "notes": None,
    },
    {
        "bank_id": "cbu",
        "bank_legal_name": "Community Bank, N.A.",
        "holding_name": "Community Bank System, Inc.",
        "aliases": ["Community Bank"],
        "notes": "generic — collides",
    },
]
P = build_patterns(BANKS)


def test_citi_does_not_match_citizens():
    """The case DESIGN names: a bare substring match makes every Citizens
    story count for Citigroup as well."""
    assert counts_for_bank("Citizens Bank cuts ties with ICE", "citi", P) is False
    assert counts_for_bank("Citizens Bank cuts ties with ICE", "cfg", P) is True
    assert counts_for_bank("Citi raises its dividend", "citi", P) is True


def test_generic_english_word_alias_cannot_attribute():
    """ "Popular" swallowed 49% of a backfill year before it was denylisted."""
    assert usable_aliases(BANKS[2]) == ["Banco Popular"]
    assert counts_for_bank("The most popular show on Netflix", "bpop", P) is False
    assert counts_for_bank("Banco Popular posts a loss", "bpop", P) is True


def test_generic_flagged_bank_uses_holding_name_only():
    """Its legal name and aliases collide with industry phrases, so a
    "community bank leverage ratio" headline must not attribute to cbu."""
    assert counts_for_bank("New community bank leverage ratio rules", "cbu", P) is False
    assert counts_for_bank("Community Bank System reports Q2", "cbu", P) is True


def test_a_bank_with_no_usable_pattern_attributes_nothing():
    """Failing open would hand that bank every row filed under it — the exact
    contamination the gate exists to stop."""
    patterns = build_patterns([{"bank_id": "x", "aliases": [], "notes": None}])
    assert counts_for_bank("anything at all", "x", patterns) is False
    assert counts_for_bank("anything", "unknown_bank", P) is False


def test_missing_title_does_not_attribute():
    assert counts_for_bank(None, "citi", P) is False
    assert counts_for_bank("", "citi", P) is False


def test_gate_rows_keeps_only_self_naming_and_counts_both_sides():
    rows = [
        {"bank_id": "citi", "title": "Citigroup beats estimates"},
        {"bank_id": "citi", "title": "SK Hynix raises $26.5bn in US debut"},
        {"bank_id": "cfg", "title": "Citizens Bank names new CFO"},
    ]
    kept, funnel = gate_rows(rows, P)
    assert [r["bank_id"] for r in kept] == ["citi", "cfg"]
    assert funnel == {"total": 3, "attributed": 2, "unattributed": 1}


def test_legal_suffixes_are_stripped_because_headlines_drop_them():
    """The seed says "Community Bank System, Inc."; a headline says
    "Community Bank System reports Q2". Matching only the seeded form
    attributes almost nothing."""
    from pipeline.attribution import name_forms

    assert name_forms("Citizens Bank, N.A.") == ["Citizens Bank, N.A", "Citizens Bank"]
    # A derived single token needs the seed to vouch for it, or "Glacier
    # Bancorp" would hand gbci every melting-glacier headline.
    assert name_forms("Citigroup Inc.") == ["Citigroup Inc"]
    assert name_forms("Citigroup Inc.", frozenset({"citigroup"})) == [
        "Citigroup Inc",
        "Citigroup",
    ]


def test_stripping_cannot_resurrect_a_denylisted_word():
    """ "Popular, Inc." reduces to "Popular" — the exact word the denylist
    exists to keep out, so the check has to run after stripping too."""
    patterns = build_patterns([BANKS[2]])
    assert counts_for_bank("a popular new restaurant", "bpop", patterns) is False
    # The suffixed form is specific enough to keep: only the bare word goes.
    assert counts_for_bank("Popular, Inc. cuts its dividend", "bpop", patterns) is True
    assert counts_for_bank("Banco Popular cuts its dividend", "bpop", patterns) is True


def test_stripping_cannot_manufacture_a_bare_english_word():
    """Reviewed 2026-08-14: "Bancorp" is a suffix, so stripping turned
    "Glacier Bancorp" into "Glacier", "Hope Bancorp" into "Hope" and
    "U.S. Bancorp" into "U.S." — the last making the gate a no-op for usb,
    since nearly every US banking headline says "U.S."."""
    seed = [
        {
            "bank_id": "gbci",
            "bank_legal_name": "Glacier Bank",
            "holding_name": "Glacier Bancorp, Inc.",
            "aliases": ["Glacier Bancorp", "Glacier Bank"],
            "notes": None,
        },
        {
            "bank_id": "usb",
            "bank_legal_name": "U.S. Bank, N.A.",
            "holding_name": "U.S. Bancorp",
            "aliases": ["U.S. Bancorp", "U.S. Bank"],
            "notes": "generic",
        },
    ]
    p = build_patterns(seed)
    assert counts_for_bank("Glacier melt accelerates in Montana", "gbci", p) is False
    assert counts_for_bank("Glacier Bancorp reports Q2", "gbci", p) is True
    assert counts_for_bank("U.S. banks brace for CRE losses", "usb", p) is False
    assert counts_for_bank("U.S. Bancorp cuts its dividend", "usb", p) is True


def test_a_form_inside_another_banks_name_is_dropped():
    """ffbc "First Financial Bancorp." strips to "First Financial", which is
    contained in ffin "First Financial Bankshares" — the pair the seed's
    generic flag exists for, which stripping was quietly undoing."""
    seed = [
        {
            "bank_id": "ffbc",
            "bank_legal_name": "First Financial Bank",
            "holding_name": "First Financial Bancorp.",
            "aliases": ["First Financial Bancorp."],
            "notes": "generic",
        },
        {
            "bank_id": "ffin",
            "bank_legal_name": "First Financial Bank",
            "holding_name": "First Financial Bankshares, Inc.",
            "aliases": ["First Financial Bankshares"],
            "notes": "generic",
        },
    ]
    p = build_patterns(seed)
    assert counts_for_bank("First Financial Bankshares beats", "ffbc", p) is False


def test_funnel_keys_are_total_even_when_nothing_attributes():
    """main() divides by them; a Counter that omits `attributed` turns an
    all-unattributed window into a KeyError instead of a 0%."""
    rows = [{"bank_id": "citi", "title": "SK Hynix raises $26.5bn"}]
    _, funnel = gate_rows(rows, P)
    assert funnel == {"total": 1, "attributed": 0, "unattributed": 1}
    assert gate_rows([], P)[1] == {"total": 0, "attributed": 0, "unattributed": 0}
