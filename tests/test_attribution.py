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
        "aliases": ["Citi", "Citibank"],
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
    assert "Citigroup" in name_forms("Citigroup Inc.")


def test_stripping_cannot_resurrect_a_denylisted_word():
    """ "Popular, Inc." reduces to "Popular" — the exact word the denylist
    exists to keep out, so the check has to run after stripping too."""
    patterns = build_patterns([BANKS[2]])
    assert counts_for_bank("a popular new restaurant", "bpop", patterns) is False
    # The suffixed form is specific enough to keep: only the bare word goes.
    assert counts_for_bank("Popular, Inc. cuts its dividend", "bpop", patterns) is True
    assert counts_for_bank("Banco Popular cuts its dividend", "bpop", patterns) is True
