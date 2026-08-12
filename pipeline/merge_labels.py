"""Build the champion label set from two prompt runs (Stage 1).

The champion is a per-class ensemble, not a single prompt: `negative` comes
from prompt v4, every other label from v3 (decision log 2026-08-09). Each
prompt is better on a different axis and the axes turned out to be separable,
so the merge is one rule — take the override file's label wherever it says
`--only-label`, otherwise keep the base file's.

This exists so the champion labels can be regenerated rather than depending on
a one-off command someone ran once. Run:

  python -m pipeline.merge_labels --base labels_v3_full.csv \
    --override labels_v4_full.csv --only-label negative \
    --output labels_ensemble_full.csv --run-date 2026-08-09
"""

import argparse
import csv
import json

from pipeline.labeling import LABELS, validate_label


def merge_labels(
    base: list[dict], override: dict[str, str], only_label: str
) -> list[dict]:
    """Base rows with `only_label` taken from override. Pure, no I/O.

    All-or-nothing on the join, as import_labels is: a base row with no
    override counterpart means the two runs saw different corpora, which
    would silently produce a half-merged champion.
    """
    validate_label(only_label)
    out = []
    for row in base:
        rid = row["raw_item_id"]
        if rid not in override:
            raise ValueError(f"no override label for raw_item_id {rid}")
        label = only_label if override[rid] == only_label else row["label"]
        out.append({**row, "label": validate_label(label)})
    return out


def read_rows(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="labels file for every other class")
    ap.add_argument("--override", required=True, help="labels file for --only-label")
    ap.add_argument("--only-label", default="negative", choices=LABELS)
    ap.add_argument("--output", required=True)
    ap.add_argument("--run-date", required=True, help="YYYY-MM-DD")
    args = ap.parse_args()

    base = read_rows(args.base)
    override = {r["raw_item_id"]: r["label"] for r in read_rows(args.override)}
    merged = merge_labels(base, override, args.only_label)

    def version(rows: list[dict]) -> str:
        return json.loads(rows[0]["model_meta"]).get("prompt_version", "?")

    over_version = version(read_rows(args.override))
    meta = json.loads(base[0]["model_meta"])
    meta["prompt_version"] = f"{version(base)}+{over_version}-{args.only_label[:3]}"
    meta["run_date"] = args.run_date
    meta["note"] = f"{args.only_label} from the override run, others from the base run"
    blob = json.dumps(meta)

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["raw_item_id", "label", "model_meta"])
        w.writeheader()
        for r in merged:
            w.writerow(
                {
                    "raw_item_id": r["raw_item_id"],
                    "label": r["label"],
                    "model_meta": blob,
                }
            )
    changed = sum(
        1 for a, b in zip(base, merged, strict=True) if a["label"] != b["label"]
    )
    print(f"wrote {len(merged)} labels to {args.output} ({changed} from override)")


if __name__ == "__main__":
    main()
