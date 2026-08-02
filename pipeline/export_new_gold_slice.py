"""Export 50 more gold-slice rows for human verification (Stage 1 quality gate).

Draws from item_label rows Llama has already labeled (label_source=
'llama_kaggle'), excluding every raw_item_id already spent in
evals/items/gold_slice_1..5.csv, and keeps only rows whose title names the
row's own assigned bank as a whole word via bank.aliases -- a plain
substring check would match "Citi" inside "Citizens". Samples 20 negative /
20 positive / 10 neutral (Llama's label, used only to stratify) and writes
evals/items/gold_slice_6.csv with the label column blank, same columns as
the existing slices, for blind human labeling.

Run: python -m pipeline.export_gold_slice_6
"""

import argparse
import csv
import glob
import random
import re
import sys
from collections import defaultdict

from pipeline import db

DEFAULT_COUNTS = {"negative": 20, "positive": 20, "neutral": 10}
OUT_COLUMNS = ("id", "source", "title", "text_excerpt", "label", "comment")


def load_excluded_ids(items_dir: str) -> set[int]:
    """raw_item_ids already used by any existing evals/items/gold_slice_*.csv."""
    excluded: set[int] = set()
    for path in sorted(glob.glob(f"{items_dir}/gold_slice_*.csv")):
        with open(path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                excluded.add(int(row["id"]))
    return excluded


def build_alias_patterns(banks: list[dict]) -> dict[str, list[re.Pattern]]:
    """bank_id -> word-boundary regexes over bank.aliases.

    Lookarounds, not \\b: \\b needs a word char on both sides, so it can
    misbehave around punctuation-adjacent aliases; (?<!\\w)/(?!\\w) requires
    the alias to stand alone as a whole word either way, same convention as
    poll_agency_rss.py's build_alias_index. This is what stops a short alias
    like "Citi" from matching inside "Citizens".
    """
    out: dict[str, list[re.Pattern]] = {}
    for b in banks:
        cores = (
            re.sub(r"^\W+|\W+$", "", a.lower()) for a in (b.get("aliases") or [])
        )
        patterns = [re.compile(r"(?<!\w)" + re.escape(c) + r"(?!\w)") for c in cores if c]
        if patterns:
            out[b["bank_id"]] = patterns
    return out


def matches_own_bank(title: str | None, bank_id: str, patterns_by_bank: dict) -> bool:
    """Does `title` name `bank_id` by one of its own aliases (whole word)?"""
    patterns = patterns_by_bank.get(bank_id)
    if not patterns or not title:
        return False
    title_lower = title.lower()
    return any(p.search(title_lower) for p in patterns)


def select_candidates(
    rows: list[dict], patterns_by_bank: dict, excluded_ids: set[int]
) -> list[dict]:
    """Pure filter: unseen ids + title names the row's own bank by alias."""
    return [
        r
        for r in rows
        if r["id"] not in excluded_ids
        and matches_own_bank(r["title"], r["bank_id"], patterns_by_bank)
    ]


def stratified_sample(candidates: list[dict], counts: dict, seed: int) -> list[dict]:
    """Sample counts[label] rows per Llama label, then shuffle the combined
    set so file order doesn't leak which rows are which class (blind
    labeling). Raises if any class is short, rather than silently sampling
    fewer.
    """
    rng = random.Random(seed)
    by_label: dict[str, list[dict]] = defaultdict(list)
    for r in candidates:
        by_label[r["label"]].append(r)

    picked: list[dict] = []
    shortfalls: dict[str, tuple[int, int]] = {}
    for label, n in counts.items():
        pool = by_label.get(label, [])
        if len(pool) < n:
            shortfalls[label] = (len(pool), n)
        picked.extend(rng.sample(pool, min(n, len(pool))))

    if shortfalls:
        detail = ", ".join(
            f"{label}: have {have}, need {need}"
            for label, (have, need) in shortfalls.items()
        )
        raise ValueError(f"not enough candidates ({detail})")

    rng.shuffle(picked)
    return picked


def fetch_banks(conn) -> list[dict]:
    with conn.cursor() as cur:
        cur.execute("SELECT bank_id, aliases FROM bank")
        return cur.fetchall()


def fetch_candidate_rows(conn) -> list[dict]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT il.raw_item_id AS id, ri.source, ri.bank_id, ri.title, "
            "       ri.text_excerpt, il.label "
            "FROM item_label il "
            "JOIN raw_item ri ON ri.id = il.raw_item_id "
            "WHERE il.label_source = 'llama_kaggle' "
            "  AND ri.source IN ('gdelt', 'edgar')"
        )
        return cur.fetchall()


def write_csv(path: str, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=OUT_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    "id": r["id"],
                    "source": r["source"],
                    "title": r["title"],
                    "text_excerpt": r["text_excerpt"] or "",
                    "label": "",
                    "comment": "",
                }
            )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="evals/items/gold_slice_6.csv")
    ap.add_argument("--items-dir", default="evals/items")
    ap.add_argument("--negative", type=int, default=DEFAULT_COUNTS["negative"])
    ap.add_argument("--positive", type=int, default=DEFAULT_COUNTS["positive"])
    ap.add_argument("--neutral", type=int, default=DEFAULT_COUNTS["neutral"])
    ap.add_argument("--seed", type=int, default=6, help="reproducible sample draw")
    args = ap.parse_args()
    counts = {
        "negative": args.negative,
        "positive": args.positive,
        "neutral": args.neutral,
    }

    excluded_ids = load_excluded_ids(args.items_dir)

    conn = db.connect()
    try:
        banks = fetch_banks(conn)
        rows = fetch_candidate_rows(conn)
    finally:
        conn.close()

    patterns_by_bank = build_alias_patterns(banks)
    candidates = select_candidates(rows, patterns_by_bank, excluded_ids)
    sampled = stratified_sample(candidates, counts, args.seed)
    write_csv(args.out, sampled)
    print(f"wrote {len(sampled)} rows to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
