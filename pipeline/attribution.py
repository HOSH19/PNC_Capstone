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
while live rows are 6.6% — measured, 402,316 of 407,032 against 1,346 of
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
# ourselves, it bites. Of 21 single-word aliases across the seed this is the
# only true English word — the rest are Amex, BofA, Citi, Schwab, Truist and
# similar. Fix properly by flagging bpop "generic" in db/seed/banks.csv, which
# poll_agency_rss.build_alias_index already keys off; until a re-seed lands,
# this denylist is what protects the gate.
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


def name_forms(name: str) -> list[str]:
    """A name and its headline form, stripped of trailing legal suffixes.

    Applied repeatedly, because the seed stacks them: "Citizens Bank, N.A."
    and "…Financial Group, Inc." both need two passes to reach the form a
    headline actually uses.
    """
    forms = []
    current = re.sub(r"^\W+|\W+$", "", name.strip())
    while current and current not in forms:
        forms.append(current)
        current = re.sub(r"^\W+|\W+$", "", CORP_SUFFIX.sub("", current))
    return forms


def build_patterns(banks: list[dict]) -> dict[str, list[re.Pattern]]:
    """bank_id -> whole-word regexes over its names.

    Lookarounds rather than \\b: \\b needs a word character on both sides, so
    it misbehaves next to punctuation-adjacent names. `(?<!\\w)`/`(?!\\w)`
    requires the name to stand alone either way — which is the whole point,
    since bare "Citi" otherwise matches "Citizens" (DESIGN, Bank attribution).
    Same convention as poll_agency_rss.build_alias_index.
    """
    out: dict[str, list[re.Pattern]] = {}
    for b in safe_banks(banks):
        # "generic"-flagged banks use the holding name only: their legal names
        # and aliases collide with unrelated banks and industry phrases
        # ("Community Bank" matched "community bank leverage ratio").
        if b.get("notes") and "generic" in b["notes"].lower():
            names = [b.get("holding_name")]
        else:
            names = [b.get("bank_legal_name"), b.get("holding_name")]
            names += list(b.get("aliases") or [])
        cores = {form.lower() for n in names if n for form in name_forms(n)}
        # Re-check after stripping: "Popular, Inc." reduces to "Popular", the
        # very word the denylist exists to keep out.
        cores -= GENERIC_ALIASES
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
    kept, funnel = [], Counter()
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
    ap.add_argument("--until", default="2027-01-01")
    args = ap.parse_args()

    conn = db.connect()
    try:
        patterns = build_patterns(db.get_live_banks(conn))
        with conn.cursor() as cur:
            cur.execute(
                """SELECT bank_id, title,
                          (published_at < '2026-01-01') AS is_backfill
                   FROM raw_item
                   WHERE source = 'gdelt'
                     AND published_at >= %(since)s AND published_at < %(until)s""",
                {"since": args.since, "until": args.until},
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    for label, subset in (
        ("backfill (GKG, 2020-2024)", [r for r in rows if r["is_backfill"]]),
        ("live (DOC API, 2026+)", [r for r in rows if not r["is_backfill"]]),
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
