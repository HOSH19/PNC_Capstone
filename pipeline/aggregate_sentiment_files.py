"""Roll the backtest parquet up to bank-quarter, as a CSV.

The file counterpart of aggregate_sentiment.py: that module rolls live
item_score rows up inside the database, this one rolls the Kaggle-scored
2020-2024 parquet up locally. Same semantics — only attributed rows carry
signal, directional counts are p(class) >= threshold with the threshold
recorded per row, never argmax.

fdic_cert_number comes from db/seed/banks.csv, not the database: the whole
point of the file path is that it runs without one. pyarrow is imported
inside main like torch in score_finbert — CI tests the pure aggregation on
plain dicts and never reads a parquet.

    python -m pipeline.aggregate_sentiment_files \\
        --parquet scores_gkg_2020_2024.parquet --out corpus/sentiment_quarter.csv
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

CSV_COLUMNS = [
    "bank_id",
    "quarter_end_date",
    "fdic_cert_number",
    "n_scored",
    "n_negative",
    "n_positive",
    "neg_share",
    "pos_share",
    "mean_p_negative",
    "mean_p_positive",
    "threshold",
    "model_version",
]


def quarter_end(d) -> str:
    """date/datetime -> quarter-end ISO date, matching date_trunc('quarter')
    + 3 months - 1 day in aggregate_sentiment's SQL."""
    q_month = (d.month - 1) // 3 * 3 + 3
    last = {3: 31, 6: 30, 9: 30, 12: 31}[q_month]
    return f"{d.year}-{q_month:02d}-{last}"


def aggregate(items: list[dict], threshold: float) -> list[dict]:
    """Attributed item rows -> bank-quarter rows. Pure.

    `items` need bank_id, published_at (datetime), attributed (bool),
    p_negative, p_positive, model_version. Unattributed rows are dropped
    here — the gate is load-bearing for the backtest (DESIGN, 2026-08-14) —
    which is why the parquet keeps them: re-running this with a changed gate
    or threshold is free, re-scoring is not.
    """
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in items:
        if r["attributed"]:
            groups[(r["bank_id"], quarter_end(r["published_at"]))].append(r)
    out = []
    for (bank_id, q), rows in sorted(groups.items()):
        n = len(rows)
        n_neg = sum(1 for r in rows if r["p_negative"] >= threshold)
        n_pos = sum(1 for r in rows if r["p_positive"] >= threshold)
        out.append(
            {
                "bank_id": bank_id,
                "quarter_end_date": q,
                "fdic_cert_number": "",  # joined in main from the seed
                "n_scored": n,
                "n_negative": n_neg,
                "n_positive": n_pos,
                "neg_share": round(n_neg / n, 4),
                "pos_share": round(n_pos / n, 4),
                "mean_p_negative": round(sum(r["p_negative"] for r in rows) / n, 4),
                "mean_p_positive": round(sum(r["p_positive"] for r in rows) / n, 4),
                "threshold": threshold,
                "model_version": ",".join(sorted({r["model_version"] for r in rows})),
            }
        )
    return out


def seed_certs(seed_path: str) -> dict[str, str]:
    with open(seed_path, newline="") as f:
        return {r["bank_id"]: r["fdic_cert"] for r in csv.DictReader(f)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument(
        "--seed", default=str(Path(__file__).parents[1] / "db/seed/banks.csv")
    )
    args = ap.parse_args()

    import pyarrow.parquet as pq  # local-only dependency, like torch

    table = pq.read_table(
        args.parquet,
        columns=[
            "bank_id",
            "published_at",
            "attributed",
            "p_negative",
            "p_positive",
            "model_version",
        ],
    )
    items = table.to_pylist()
    rows = aggregate(items, args.threshold)
    certs = seed_certs(args.seed)
    for r in rows:
        r["fdic_cert_number"] = certs.get(r["bank_id"], "")

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    attributed = sum(r["n_scored"] for r in rows)
    print(
        f"{len(items)} items -> {attributed} attributed -> "
        f"{len(rows)} bank-quarters -> {args.out}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
