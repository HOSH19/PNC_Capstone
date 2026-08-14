"""Does a scored row count toward the bank it was filed under? (Stage 3.)

Every eligible item is labeled and scored regardless; this decides only
whether that score joins a given bank's rollup. See scoring/DESIGN.md
"Bank attribution" — the check is: **is the bank named in its own title?**

It exists because `raw_item.bank_id` records the query a row arrived under,
not what the row is about. Measured on the 2026-07-22 batch, only 726 of
7,756 GDELT titles (9.4%) name their own bank, and 21 of 30 signal-bearing
gold rows sit under the wrong one. Gating at ingest would remove the false
signal but throw away ~90% of the training corpus, which is *valid* text ->
direction material; gating at the rollup keeps both.

⚠️ The backtest depends on this. The 2020-2024 backfill comes from GKG
matched on the title naming the bank, so those rows are ~100% self-naming
while live rows are 6.5% — measured, 398,841 of 401,061 against 1,310 of
20,286 over a healthy fortnight. Score the backfill without applying this
gate to the live path too and the backtest reads better than production for
reasons that have nothing to do with the model.

Lives in pipeline/ because tests/test_structure.py allows executable code
only in the dirs that own it, and index/ is not one of them — same reason
quality_gate.py and export_training_set.py are here rather than in scoring/.
"""

import argparse
import re
import sys
from collections import Counter

from pipeline import db

# Aliases that are ordinary English words rather than brand tokens. Matching
# them as free text lets one bank swallow the corpus: "Popular" pulled 58,384
# of 119,488 backfill rows for 2020 -- Netflix rankings, popular shops -- at a
# 2% financial-vocabulary rate where real banks run 30-45%.
#
# The DOC API never exposed this because GDELT treats gdelt_query's "Popular"
# as a quoted phrase against its own index; the moment we match aliases
# ourselves, it bites. It is the only bare English word the seed itself
# spells out — the other single-word aliases are Amex, BofA, Citi, Schwab,
# Truist and similar. Words that *stripping* would manufacture ("Glacier"
# from "Glacier Bancorp") are a separate problem, handled in name_forms
# rather than by extending this list. Fix properly by flagging bpop "generic"
# in db/seed/banks.csv, which poll_agency_rss.build_alias_index already keys
# off; until a re-seed lands, this denylist is what protects the gate.
GENERIC_ALIASES = frozenset({"popular"})


def usable_aliases(bank: dict) -> list[str]:
    """A bank's aliases minus the ones that are just English words."""
    return [
        a
        for a in (bank.get("aliases") or [])
        if a and a.strip().lower() not in GENERIC_ALIASES
    ]


def safe_banks(banks: list[dict]) -> list[dict]:
    """Banks with generic aliases stripped, for matching against free text."""
    return [{**b, "aliases": usable_aliases(b)} for b in banks]


# Legal suffixes headlines drop. "Community Bank System, Inc." is how the
# seed spells it and "Community Bank System reports Q2" is how a headline
# does, so matching only the seeded form attributes almost nothing.
CORP_SUFFIX = re.compile(
    r"[\s,]+(inc|corp|corporation|company|co|n\.?a|"
    r"ltd|llc|plc|group|holdings|bancorp|bancorporation)\.?$",
    re.IGNORECASE,
)


def name_forms(name: str, seeded: frozenset[str] = frozenset()) -> list[str]:
    """A name and its headline form, stripped of trailing legal suffixes.

    Applied repeatedly, because the seed stacks them: "Citizens Bank, N.A."
    and "…Financial Group, Inc." both need two passes to reach the form a
    headline actually uses.

    A stripped form collapsing to ONE token is dropped unless the seed spells
    it out. "Bancorp" is a suffix, so stripping turns "Glacier Bancorp" into
    "Glacier", "Hope Bancorp" into "Hope" and "U.S. Bancorp" into "U.S." —
    words that match weather reports, soft-landing commentary and virtually
    every US banking headline respectively. `seeded` is what saves the real
    ones: "Citi" survives because the seed lists it as an alias, and nothing
    else on that list is a bare English word.
    """
    forms: list[str] = []
    current = re.sub(r"^\W+|\W+$", "", name.strip())
    while current and current not in forms:
        if forms and " " not in current and current.lower() not in seeded:
            break  # a derived single token: too broad unless seeded
        forms.append(current)
        current = re.sub(r"^\W+|\W+$", "", CORP_SUFFIX.sub("", current))
    return forms


def seeded_names(bank: dict) -> frozenset[str]:
    """Every name the seed actually spells out for this bank, normalized."""
    values = [bank.get("bank_legal_name"), bank.get("holding_name")]
    values += list(bank.get("aliases") or [])
    return frozenset(re.sub(r"^\W+|\W+$", "", v.strip()).lower() for v in values if v)


def _collides(core: str, bank_id: str, others: dict[str, frozenset[str]]) -> bool:
    """Does `core` appear as a whole word inside a different bank's name?"""
    pattern = re.compile(r"(?<!\w)" + re.escape(core) + r"(?!\w)")
    return any(
        bid != bank_id and any(pattern.search(n) for n in names)
        for bid, names in others.items()
    )


def build_patterns(banks: list[dict]) -> dict[str, list[re.Pattern]]:
    """bank_id -> whole-word regexes over its names.

    Lookarounds rather than \\b: \\b needs a word character on both sides, so
    it misbehaves next to punctuation-adjacent names. `(?<!\\w)`/`(?!\\w)`
    requires the name to stand alone either way — which is the whole point,
    since bare "Citi" otherwise matches "Citizens" (DESIGN, Bank attribution).
    Same convention as poll_agency_rss.build_alias_index.
    """
    cleaned = safe_banks(banks)
    others = {b["bank_id"]: seeded_names(b) for b in cleaned}
    out: dict[str, list[re.Pattern]] = {}
    for b in cleaned:
        # "generic"-flagged banks use the holding name only: their legal names
        # and aliases collide with unrelated banks and industry phrases
        # ("Community Bank" matched "community bank leverage ratio").
        generic = bool(b.get("notes") and "generic" in b["notes"].lower())
        if generic:
            names = [b.get("holding_name")]
        else:
            names = [b.get("bank_legal_name"), b.get("holding_name")]
            names += list(b.get("aliases") or [])
        seeded = seeded_names(b)
        cores = {form.lower() for n in names if n for form in name_forms(n, seeded)}
        # Re-check after stripping: "Popular, Inc." reduces to "Popular", the
        # very word the denylist exists to keep out.
        cores -= GENERIC_ALIASES
        # And drop any form that sits inside another bank's name. Stripping
        # otherwise undoes the generic flag: ffbc's "First Financial Bancorp."
        # becomes "First Financial", which is contained in ffin's "First
        # Financial Bankshares" — the very pair the flag was introduced for.
        # Checked against the seeded names rather than a token count, because
        # the question is whether a real collision exists, not how long the
        # string is: "Community Bank System" is three tokens and collides with
        # nothing, "First Financial" is two and collides.
        cores = {c for c in cores if not _collides(c, b["bank_id"], others)}
        patterns = [
            re.compile(r"(?<!\w)" + re.escape(c) + r"(?!\w)") for c in sorted(cores)
        ]
        if patterns:
            out[b["bank_id"]] = patterns
    return out


def counts_for_bank(title: str | None, bank_id: str, patterns: dict) -> bool:
    """Does `title` name `bank_id` by one of its own names?

    A bank with no usable pattern never attributes, rather than attributing
    everything: silently counting every row for a bank whose aliases were all
    filtered out is the failure this gate exists to prevent.
    """
    ps = patterns.get(bank_id)
    if not ps or not title:
        return False
    lowered = title.lower()
    return any(p.search(lowered) for p in ps)


def gate_rows(rows: list[dict], patterns: dict) -> tuple[list[dict], dict]:
    """Split scored rows into attributed and not. Pure, no I/O."""
    kept = []
    # Seeded so the keys are total: a subset where nothing attributes is
    # precisely when a caller reads `attributed`, and a Counter would omit it.
    funnel = Counter({"total": 0, "attributed": 0, "unattributed": 0})
    for r in rows:
        funnel["total"] += 1
        if counts_for_bank(r.get("title"), r.get("bank_id"), patterns):
            kept.append(r)
            funnel["attributed"] += 1
        else:
            funnel["unattributed"] += 1
    return kept, dict(funnel)


def main() -> None:
    """Report what the gate would do to the rows already collected.

    The aggregation layer does not exist yet, so this is how the gate's
    effect gets measured before anything depends on it.
    """
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--since", default="2020-01-01")
    ap.add_argument("--until", default="2100-01-01")  # no silent truncation
    args = ap.parse_args()

    conn = db.connect()
    try:
        patterns = build_patterns(db.get_live_banks(conn))
        with conn.cursor() as cur:
            cur.execute(
                """SELECT bank_id, title,
                          -- backfill_gkg stamps this; the DOC API backfill
                          -- (backfill_gdelt) wrote the same source over the
                          -- same years, so a date cutoff would count its rows
                          -- as GKG and inflate the headline share.
                          (meta ->> 'via' = 'gkg_backfill') AS is_backfill
                   FROM raw_item
                   WHERE source = 'gdelt'
                     AND published_at >= %(since)s AND published_at < %(until)s""",
                {"since": args.since, "until": args.until},
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    for label, subset in (
        ("backfill (GKG)", [r for r in rows if r["is_backfill"]]),
        ("everything else (DOC API)", [r for r in rows if not r["is_backfill"]]),
    ):
        if not subset:
            continue
        _, funnel = gate_rows(subset, patterns)
        share = funnel["attributed"] / funnel["total"]
        print(
            f"{label}: {funnel['attributed']:,} of {funnel['total']:,} "
            f"attributed ({share:.1%})",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
