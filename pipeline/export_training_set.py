"""Assemble the FinBERT fine-tune train/val CSVs (Stage 2, scoring/DESIGN.md).

Joins labels_<date>.csv to labeling_batch_<date>.csv on raw_item_id, applies
the training-set hygiene filters, overrides Llama's label with the human gold
labels where they exist, and writes a time-based train/validation split
(the last --holdout-days before the newest row go to validation — random
splits leak syndication near-duplicates). Days, not DESIGN.md's "e.g. last
N weeks": GDELT polling went live 2026-07-09, so 94% of the batch sits in
its final 13 days and a week-sized holdout would swallow the corpus.

Text assembly reuses pipeline.eligibility.text_for — the single definition of
"what text represents this item", shared with labeling and serving.

Run:
  python -m pipeline.export_training_set --labels labels_2026-07-22.csv \
    --batch labeling_batch_2026-07-22.csv --gold-dir evals/items \
    --output-prefix finbert_2026-08-07
"""

import argparse
import csv
import re
import sys
from collections import Counter
from datetime import datetime, timedelta

from pipeline.eligibility import text_for
from pipeline.quality_gate import load_gold_rows

# Training-set hygiene (DESIGN.md): holdings/13F wire-spam titles whose
# direction belongs to the *other* entity ("X Purchases N Shares of Y").
# Structural two-entity templates only — rating/price-target titles
# ("Commerce Bancshares Downgraded by WSZ") deliberately do NOT match; that
# is the keep-for-now class. DESIGN's table row (723/51) came from a
# measurement whose predicate was not kept; per its own "settle in the
# export script" note, this predicate is the settled one — it flags 983
# rows (43 directional) on the 2026-07-22 batch, disjoint from the rating
# class (overlap measured 0).
HOLDINGS_SPAM = re.compile(
    r"(purchase|buy|acquire|sell|boost|raise|increase|decrease|cut|trim"
    r"|lower|reduce|grow|take|add)s? [^.]*?\b(stock )?"
    r"(position|shares|stake|holdings) (of|in)\b"
    r"|shares (sold|purchased|bought|acquired) by"
    r"|has \$[\d,. ]+(million|billion)? ?(stock )?(position|stake|holdings) in"
    r"|invests? \$[\d,. ]+(million|billion)? ?in",
    re.I,
)


def is_holdings_spam(title: str | None) -> bool:
    return bool(title) and bool(HOLDINGS_SPAM.search(title))


def build_training_set(
    labels_rows: list[dict],
    batch_rows: list[dict],
    gold_by_id: dict[str, str],
    holdout_days: int,
    holdout_ids: frozenset[str] = frozenset(),
) -> tuple[list[dict], list[dict], list[dict], dict]:
    """Hygiene filters + human override + time split. Pure, no I/O.

    Returns (train, val, holdout, funnel); each output row is
    {raw_item_id, published_at, text, label}.

    holdout_ids are human-labeled rows pulled out of BOTH train and val (see
    quality_gate.stratum_for). Without this the gold rows scatter across the
    split by publication date and the only human-truth rows left to evaluate
    on are ones the model trained against; val's labels are otherwise Llama's,
    so scoring against it measures agreement with the labeler, not accuracy.
    """
    by_id = {r["raw_item_id"]: r for r in batch_rows}
    funnel: dict = {"total": len(labels_rows)}
    skipped: Counter = Counter()
    seen_titles: set[str] = set()
    kept: list[dict] = []
    holdout: list[dict] = []
    overrides = 0

    for lab in labels_rows:
        item = by_id.get(lab["raw_item_id"])
        if item is None:
            raise ValueError(f"no batch row for raw_item_id {lab['raw_item_id']}")
        title = (item.get("title") or "").strip()
        if item["source"] == "edgar" and not (item.get("text_excerpt") or "").strip():
            skipped["contentless_edgar"] += 1
            continue
        if is_holdings_spam(title):
            skipped["holdings_spam"] += 1
            continue
        norm = title.lower()
        if norm in seen_titles:
            skipped["dup_title"] += 1
            continue
        seen_titles.add(norm)
        label = gold_by_id.get(lab["raw_item_id"])
        if label is not None and label != lab["label"]:
            overrides += 1
        row = {
            "raw_item_id": lab["raw_item_id"],
            "published_at": item["published_at"],
            "text": text_for(item),
            "label": label if label is not None else lab["label"],
        }
        # Holdout rows are evaluation-only: out of train AND val, both here
        # and — unlike the dedup/hygiene skips — kept as their own file.
        if lab["raw_item_id"] in holdout_ids:
            skipped["holdout"] += 1
            holdout.append(row)
            continue
        kept.append(row)

    cutoff = max(datetime.fromisoformat(r["published_at"]) for r in kept) - timedelta(
        days=holdout_days
    )
    train = [r for r in kept if datetime.fromisoformat(r["published_at"]) < cutoff]
    val = [r for r in kept if datetime.fromisoformat(r["published_at"]) >= cutoff]

    funnel["skipped"] = dict(skipped)
    funnel["human_overrides"] = overrides
    funnel["cutoff"] = cutoff.isoformat()
    for name, rows in (("train", train), ("val", val), ("holdout", holdout)):
        funnel[name] = {"n": len(rows), **Counter(r["label"] for r in rows)}
    return train, val, holdout, funnel


def read_rows(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: str, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["raw_item_id", "text", "label"])
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in ("raw_item_id", "text", "label")})


def print_funnel(funnel: dict) -> None:
    print("training-set funnel:", file=sys.stderr)
    print(f"  total labeled      {funnel['total']}", file=sys.stderr)
    for reason, n in sorted(funnel["skipped"].items()):
        print(f"    skipped[{reason}]  {n}", file=sys.stderr)
    print(f"  human overrides    {funnel['human_overrides']}", file=sys.stderr)
    print(f"  time cutoff        {funnel['cutoff']}", file=sys.stderr)
    for split in ("train", "val", "holdout"):
        print(f"  {split:7}            {funnel[split]}", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True, help="labels_<date>.csv")
    ap.add_argument("--batch", required=True, help="labeling_batch_<date>.csv")
    ap.add_argument("--gold-dir", default="evals/items")
    ap.add_argument("--holdout-days", type=int, default=3)
    ap.add_argument(
        "--no-holdout",
        action="store_true",
        help="keep the human holdout rows in train/val (loses the clean eval set)",
    )
    ap.add_argument("--output-prefix", required=True, help="e.g. finbert_<date>")
    args = ap.parse_args()

    gold = load_gold_rows(args.gold_dir)
    gold_by_id = {r["id"]: r["label"] for r in gold}
    holdout_ids = frozenset(
        r["id"] for r in gold if r["stratum"] == "holdout" and not args.no_holdout
    )
    train, val, holdout, funnel = build_training_set(
        read_rows(args.labels),
        read_rows(args.batch),
        gold_by_id,
        args.holdout_days,
        holdout_ids,
    )
    print_funnel(funnel)
    write_csv(f"{args.output_prefix}_train.csv", train)
    write_csv(f"{args.output_prefix}_val.csv", val)
    if holdout:
        write_csv(f"{args.output_prefix}_holdout.csv", holdout)
    print(
        f"wrote {len(train)} train / {len(val)} val / {len(holdout)} holdout rows "
        f"to {args.output_prefix}_{{train,val,holdout}}.csv",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
