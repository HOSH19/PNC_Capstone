"""Build full-filer distress labels using the v1 rule (all Call Report banks).

Same event / lookahead logic as evals/build_distress_labels.py and
evals/distress_definition.md, but without the 104-seed filter. Source is
unified_ffiec_fdic_dataset/tables/fact_call_report.csv (~8.6k certs).

Run:
  python3 evals/build_distress_labels_full.py
  python3 evals/build_distress_labels_full.py --verify-seed
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))

# Reuse the locked v1 rule helpers.
from build_distress_labels import (  # noqa: E402
    DEPOSIT_THR,
    HEADER,
    LOOKAHEAD_QUARTERS,
    NPL_LEVEL_THR,
    NPL_MULTIPLE_THR,
    START,
    label_bank,
    load_seed,
    parse_date,
    parse_float,
)
CALL = ROOT / "unified_ffiec_fdic_dataset" / "tables" / "fact_call_report.csv"
SEED_LABELS = ROOT / "evals" / "items" / "distress_bank_quarter.csv"
OUT = ROOT / "evals" / "items" / "distress_bank_quarter_full.csv"

# distress_within_4q on quarter T requires observing events through T+4.
# Drop each bank's last LOOKAHEAD_QUARTERS rows — covers both the global
# dataset end and banks that exit the panel early (merger, closure, etc.).
DROP_UNCLOSED = True


def load_all_quarters(seed_ids: dict[int, str]) -> list[dict]:
    rows: list[dict] = []
    with CALL.open(newline="") as f:
        for row in csv.DictReader(f):
            cert = int(row["fdic_cert_number"])
            rd = parse_date(row["report_date"])
            if rd < START:
                continue
            rows.append(
                {
                    "fdic_cert_number": cert,
                    "bank_id": seed_ids.get(cert, ""),
                    "quarter_end_date": rd,
                    "total_deposits": parse_float(row.get("total_deposits")),
                    "npl_ratio": parse_float(row.get("npl_ratio")),
                }
            )
    rows.sort(key=lambda r: (r["fdic_cert_number"], r["quarter_end_date"]))
    return rows


def drop_unclosed_per_bank(by_bank: dict[int, list[dict]]) -> tuple[list[dict], int]:
    """Remove each bank's last LOOKAHEAD_QUARTERS rows (unobservable future)."""
    if not DROP_UNCLOSED:
        flat = [r for cert in sorted(by_bank) for r in by_bank[cert]]
        return flat, 0

    kept_by_bank: dict[int, list[dict]] = {}
    dropped_rows = 0
    for cert in sorted(by_bank):
        bank_rows = by_bank[cert]
        if len(bank_rows) <= LOOKAHEAD_QUARTERS:
            dropped_rows += len(bank_rows)
            continue
        kept_by_bank[cert] = bank_rows[:-LOOKAHEAD_QUARTERS]
        dropped_rows += LOOKAHEAD_QUARTERS

    flat = [r for cert in sorted(kept_by_bank) for r in kept_by_bank[cert]]
    print(
        f"  dropped {dropped_rows:,} unclosed bank-quarters "
        f"({len(by_bank):,} certs -> {len(kept_by_bank):,} with closed labels): "
        f"{sum(len(r) for r in by_bank.values()):,} -> {len(flat):,} rows"
    )
    return flat, dropped_rows


def build_rows(seed_ids: dict[int, str]) -> list[dict]:
    rows = load_all_quarters(seed_ids)
    by_bank: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        by_bank[r["fdic_cert_number"]].append(r)
    for bank_rows in by_bank.values():
        label_bank(bank_rows)
    flat, _ = drop_unclosed_per_bank(by_bank)
    return flat


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HEADER, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    "fdic_cert_number": r["fdic_cert_number"],
                    "quarter_end_date": r["quarter_end_date"].isoformat(),
                    "bank_id": r["bank_id"],
                    "is_event_quarter": r["is_event_quarter"],
                    "distress_within_4q": r["distress_within_4q"],
                    "event_reason": r["event_reason"],
                }
            )


def summarize(rows: list[dict]) -> None:
    events = [r for r in rows if r["is_event_quarter"] == 1]
    positives = [r for r in rows if r["distress_within_4q"] == 1]
    print(f"Wrote {OUT.relative_to(ROOT)}")
    print(f"  rows:                 {len(rows)}")
    print(f"  banks:                {len({r['fdic_cert_number'] for r in rows})}")
    print(f"  is_event_quarter=1:   {len(events)}")
    print(f"  distress_within_4q=1: {len(positives)}")
    print(
        f"  rule constants:       dep={DEPOSIT_THR} npl_mult={NPL_MULTIPLE_THR} "
        f"npl_lvl={NPL_LEVEL_THR} lookahead={LOOKAHEAD_QUARTERS}"
    )
    by_year = Counter(r["quarter_end_date"].year for r in events)
    print(
        "  events/year:          "
        + ", ".join(f"{y}:{by_year[y]}" for y in sorted(by_year))
    )


def verify_seed(full_rows: list[dict]) -> int:
    """Compare full-build rows restricted to seed certs against the locked CSV.

    Only keys present in the full (closed-label) build are required to match.
    Seed CSV rows in unclosed tail quarters are skipped.
    """
    seed = load_seed()
    seed_certs = set(seed)
    full_by_key = {
        (r["fdic_cert_number"], r["quarter_end_date"].isoformat()): r
        for r in full_rows
        if r["fdic_cert_number"] in seed_certs
    }

    if not SEED_LABELS.exists():
        print(f"FAIL: missing {SEED_LABELS}", file=sys.stderr)
        return 1

    mismatches = 0
    compared = 0
    skipped_unclosed = 0
    seed_keys: set[tuple[int, str]] = set()
    with SEED_LABELS.open(newline="") as f:
        for row in csv.DictReader(f):
            key = (int(row["fdic_cert_number"]), row["quarter_end_date"][:10])
            seed_keys.add(key)
            got = full_by_key.get(key)
            if got is None:
                skipped_unclosed += 1
                continue
            compared += 1
            for col in ("is_event_quarter", "distress_within_4q", "event_reason"):
                exp = row[col]
                act = str(got[col])
                if exp != act:
                    print(f"MISMATCH {key} {col}: seed={exp!r} full={act!r}")
                    mismatches += 1

    extra = set(full_by_key) - seed_keys
    if extra:
        print(f"EXTRA seed rows in full build not in CSV: {len(extra)}")
        mismatches += len(extra)

    if mismatches == 0:
        print(
            f"VERIFY OK: agreed on {compared} seed bank-quarters "
            f"(skipped {skipped_unclosed} seed rows outside closed window)"
        )
        return 0
    print(f"VERIFY FAIL: {mismatches} disagreements", file=sys.stderr)
    return 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--verify-seed",
        action="store_true",
        help="After writing, require row-level agreement with the 104-bank CSV",
    )
    args = p.parse_args()

    seed_ids = load_seed()
    rows = build_rows(seed_ids)
    write_csv(rows, OUT)
    summarize(rows)
    if args.verify_seed:
        return verify_seed(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
