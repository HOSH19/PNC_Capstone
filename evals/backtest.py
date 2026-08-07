"""Distress backtest harness (Shu Han).

Grades bank×quarter risk scores against
evals/items/distress_bank_quarter.csv per evals/backtest_protocol.md.

Smoke (run A — naive −tier1 + random baseline):
  python3 evals/backtest.py --smoke

Score CSV:
  python3 evals/backtest.py --scores path/to/scores.csv
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LABELS = ROOT / "evals" / "items" / "distress_bank_quarter.csv"
CALL = ROOT / "unified_ffiec_fdic_dataset" / "tables" / "fact_call_report.csv"
REPORTS = ROOT / "evals" / "reports"

SPLIT_DATE = date(2021, 12, 31)
TEST_END = date(2024, 12, 31)
BUDGET = 10
PRECISION_AT = 50
RANDOM_SEED = 0


def parse_date(s: str) -> date:
    y, m, d = s[:10].split("-")
    return date(int(y), int(m), int(d))


def parse_float(s: str | None) -> float | None:
    if s is None or s == "":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def average_precision(y_true: list[int], y_score: list[float]) -> float:
    """PR-AUC / average precision. Uses sklearn if present, else stdlib."""
    if not y_true or sum(y_true) == 0:
        return float("nan")
    try:
        from sklearn.metrics import average_precision_score

        return float(average_precision_score(y_true, y_score))
    except ImportError:
        # Rank by score descending; AP = mean precision at each positive hit.
        order = sorted(range(len(y_score)), key=lambda i: y_score[i], reverse=True)
        hit = 0
        total_pos = sum(y_true)
        ap_sum = 0.0
        for rank, i in enumerate(order, start=1):
            if y_true[i] == 1:
                hit += 1
                ap_sum += hit / rank
        return ap_sum / total_pos


def precision_at_k(y_true: list[int], y_score: list[float], k: int) -> float:
    if not y_true or k <= 0:
        return float("nan")
    order = sorted(range(len(y_score)), key=lambda i: y_score[i], reverse=True)
    top = order[: min(k, len(order))]
    return sum(y_true[i] for i in top) / len(top)


def recall_at_budget(
    rows: list[dict], budget: int
) -> tuple[float, int, int, int]:
    """Micro-averaged recall@budget over test quarters with ≥1 positive.

    Returns (recall, alerted_positives, total_positives, n_quarters_used).
    """
    by_q: dict[date, list[dict]] = defaultdict(list)
    for r in rows:
        by_q[r["quarter_end_date"]].append(r)

    alerted = 0
    total_pos = 0
    n_q = 0
    for _q, group in sorted(by_q.items()):
        pos_in_q = sum(1 for r in group if r["y"] == 1)
        if pos_in_q == 0:
            continue
        n_q += 1
        total_pos += pos_in_q
        ranked = sorted(group, key=lambda r: r["risk_score"], reverse=True)
        top = ranked[: min(budget, len(ranked))]
        alerted += sum(1 for r in top if r["y"] == 1)

    if total_pos == 0:
        return float("nan"), 0, 0, 0
    return alerted / total_pos, alerted, total_pos, n_q


def load_labels(path: Path) -> dict[tuple[int, date], dict]:
    out: dict[tuple[int, date], dict] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            cert = int(row["fdic_cert_number"])
            qd = parse_date(row["quarter_end_date"])
            out[(cert, qd)] = {
                "fdic_cert_number": cert,
                "quarter_end_date": qd,
                "bank_id": row.get("bank_id", ""),
                "y": int(row["distress_within_4q"]),
                "is_event_quarter": int(row.get("is_event_quarter") or 0),
            }
    return out


def load_scores_csv(path: Path) -> list[dict]:
    """Load score rows; keep first row per (fdic_cert_number, quarter_end_date).

    Ming's gp50 file has a few duplicate keys (cert 32541); without dedupe the
    harness would double-count those test rows.
    """
    rows: list[dict] = []
    seen: set[tuple[int, date]] = set()
    n_dup = 0
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            score = parse_float(row.get("risk_score"))
            if score is None:
                continue
            cert = int(row["fdic_cert_number"])
            qd = parse_date(row["quarter_end_date"])
            key = (cert, qd)
            if key in seen:
                n_dup += 1
                continue
            seen.add(key)
            rows.append(
                {
                    "fdic_cert_number": cert,
                    "quarter_end_date": qd,
                    "risk_score": score,
                    "model_version": row.get("model_version") or path.stem,
                }
            )
    if n_dup:
        print(
            f"NOTE: dropped {n_dup} duplicate score key(s) in {path.name}",
            file=sys.stderr,
        )
    return rows


def build_naive_tier1_scores(labels: dict[tuple[int, date], dict]) -> list[dict]:
    """risk = -tier1_capital_ratio for labeled seed-bank quarters."""
    wanted = set(labels)
    scores: list[dict] = []
    with CALL.open(newline="") as f:
        for row in csv.DictReader(f):
            cert = int(row["fdic_cert_number"])
            qd = parse_date(row["report_date"])
            if (cert, qd) not in wanted:
                continue
            tier1 = parse_float(row.get("tier1_capital_ratio"))
            if tier1 is None:
                continue
            scores.append(
                {
                    "fdic_cert_number": cert,
                    "quarter_end_date": qd,
                    "risk_score": -tier1,
                    "model_version": "naive_neg_tier1",
                }
            )
    return scores


def build_random_scores(
    template: list[dict], seed: int = RANDOM_SEED
) -> list[dict]:
    rng = random.Random(seed)
    out = []
    for r in template:
        out.append(
            {
                "fdic_cert_number": r["fdic_cert_number"],
                "quarter_end_date": r["quarter_end_date"],
                "risk_score": rng.random(),
                "model_version": f"random_seed{seed}",
            }
        )
    return out


def join_test(
    scores: list[dict],
    labels: dict[tuple[int, date], dict],
    split_date: date,
    test_end: date,
) -> list[dict]:
    joined: list[dict] = []
    for s in scores:
        qd = s["quarter_end_date"]
        if qd <= split_date or qd > test_end:
            continue
        lab = labels.get((s["fdic_cert_number"], qd))
        if lab is None:
            continue
        joined.append(
            {
                "fdic_cert_number": s["fdic_cert_number"],
                "quarter_end_date": qd,
                "bank_id": lab["bank_id"],
                "risk_score": s["risk_score"],
                "y": lab["y"],
                "model_version": s["model_version"],
            }
        )
    return joined


def evaluate(
    rows: list[dict], budget: int, precision_k: int
) -> dict:
    y_true = [r["y"] for r in rows]
    y_score = [r["risk_score"] for r in rows]
    pr_auc = average_precision(y_true, y_score)
    p_at = precision_at_k(y_true, y_score, precision_k)
    recall, alerted, total_pos, n_q = recall_at_budget(rows, budget)
    return {
        "n_rows": len(rows),
        "n_pos": sum(y_true),
        "n_banks": len({r["fdic_cert_number"] for r in rows}),
        "pr_auc": pr_auc,
        "precision_at_k": p_at,
        "precision_k": precision_k,
        "recall_at_budget": recall,
        "budget": budget,
        "recall_alerted": alerted,
        "recall_total_pos": total_pos,
        "recall_quarters": n_q,
        "model_version": rows[0]["model_version"] if rows else "",
    }


def fmt(x: float, digits: int = 4) -> str:
    if x != x:  # NaN
        return "—"
    return f"{x:.{digits}f}"


def render_report(
    results: list[dict],
    *,
    split_date: date,
    test_end: date,
    labels_path: Path,
) -> str:
    lines = [
        "# Backtest report — harness smoke (run A)\n",
        f"Labels: `{labels_path.relative_to(ROOT)}`  ",
        f"Split: train ≤ `{split_date.isoformat()}`, "
        f"test `{split_date.isoformat()}` < date ≤ `{test_end.isoformat()}`  ",
        "Target: `distress_within_4q`. Score: higher = riskier.\n",
        "## Results\n",
        "| model_version | n_test | n_pos | PR-AUC | "
        f"precision@{results[0]['precision_k'] if results else PRECISION_AT} | "
        f"recall@budget={results[0]['budget'] if results else BUDGET} "
        "(micro) | recall detail |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for r in results:
        detail = (
            f"{r['recall_alerted']}/{r['recall_total_pos']} pos "
            f"over {r['recall_quarters']} qtrs"
        )
        lines.append(
            f"| `{r['model_version']}` | {r['n_rows']} | {r['n_pos']} | "
            f"{fmt(r['pr_auc'])} | {fmt(r['precision_at_k'])} | "
            f"{fmt(r['recall_at_budget'])} | {detail} |"
        )
    lines.append("")
    lines.append("## Checklist\n")
    lines.append("- [x] model_version and score definition stated")
    lines.append("- [x] train/test cuts and positive counts stated")
    lines.append("- [x] PR-AUC, precision@k, recall@budget reported")
    lines.append("- [x] random + naive −tier1 both present")
    lines.append("- [ ] combined run — blocked until sentiment scores exist\n")
    lines.append("## Notes\n")
    lines.append(
        "Naive score is `risk = −tier1_capital_ratio` from "
        "`fact_call_report`. If naive is not clearly above random on "
        "PR-AUC / recall@budget, treat the harness as suspect before "
        "grading Ming's GP.\n"
    )
    return "\n".join(lines)


def run_smoke(args: argparse.Namespace) -> int:
    labels = load_labels(Path(args.labels))
    naive = build_naive_tier1_scores(labels)
    random_scores = build_random_scores(naive, seed=args.seed)

    results = []
    for scores in (naive, random_scores):
        test = join_test(
            scores, labels, parse_date(args.split_date), parse_date(args.test_end)
        )
        if not test:
            print("ERROR: no test rows after join/filter", file=sys.stderr)
            return 1
        results.append(
            evaluate(test, budget=args.budget, precision_k=args.precision_at)
        )

    report = render_report(
        results,
        split_date=parse_date(args.split_date),
        test_end=parse_date(args.test_end),
        labels_path=Path(args.labels),
    )
    print(report)

    if args.write_report:
        REPORTS.mkdir(parents=True, exist_ok=True)
        out = REPORTS / "2026-07-31_backtest_naive_tier1.md"
        out.write_text(report)
        print(f"Wrote {out.relative_to(ROOT)}", file=sys.stderr)
    return 0


def run_scores(args: argparse.Namespace) -> int:
    labels_path = Path(args.labels).expanduser().resolve()
    scores_path = Path(args.scores).expanduser().resolve()
    labels = load_labels(labels_path)
    scores = load_scores_csv(scores_path)
    test = join_test(
        scores, labels, parse_date(args.split_date), parse_date(args.test_end)
    )
    if not test:
        print("ERROR: no test rows after join/filter", file=sys.stderr)
        return 1
    result = evaluate(test, budget=args.budget, precision_k=args.precision_at)
    report = render_report(
        [result],
        split_date=parse_date(args.split_date),
        test_end=parse_date(args.test_end),
        labels_path=labels_path,
    )
    # Single-model report still uses the multi-model table shape.
    print(report)
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Run A: naive −tier1 vs random on the locked test split",
    )
    p.add_argument("--scores", type=str, help="Score CSV (fdic_cert_number, "
                   "quarter_end_date, risk_score, model_version)")
    p.add_argument("--labels", type=str, default=str(DEFAULT_LABELS))
    p.add_argument("--split-date", type=str, default=SPLIT_DATE.isoformat())
    p.add_argument("--test-end", type=str, default=TEST_END.isoformat())
    p.add_argument("--budget", type=int, default=BUDGET)
    p.add_argument("--precision-at", type=int, default=PRECISION_AT)
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    p.add_argument(
        "--write-report",
        action="store_true",
        help="With --smoke, also write evals/reports/2026-07-31_backtest_naive_tier1.md",
    )
    args = p.parse_args()

    if args.smoke:
        return run_smoke(args)
    if args.scores:
        return run_scores(args)
    p.error("provide --smoke or --scores PATH")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
