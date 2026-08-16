"""Build the v1 distress answer-key CSV for Ming / backtest.

Implements evals/distress_definition.md:
  event = deposit QoQ <= -10% OR (NPL multiple >= 1.5 AND NPL > 2%)
  distress_within_4q = 1 on the four prior quarters before each event

Run:
  python3 evals/build_distress_labels.py
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEED = ROOT / "db" / "seed" / "banks.csv"
CALL = ROOT / "unified_ffiec_fdic_dataset" / "tables" / "fact_call_report.csv"
OUT = ROOT / "evals" / "items" / "distress_bank_quarter.csv"

START = date(2017, 1, 1)
DEPOSIT_THR = -0.10
NPL_MULTIPLE_THR = 1.5
NPL_LEVEL_THR = 2.0
LOOKAHEAD_QUARTERS = 4

HEADER = [
    "fdic_cert_number",
    "quarter_end_date",
    "bank_id",
    "is_event_quarter",
    "distress_within_4q",
    "event_reason",
]


def parse_float(s: str | None) -> float | None:
    if s is None or s == "":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def parse_date(s: str) -> date:
    y, m, d = s[:10].split("-")
    return date(int(y), int(m), int(d))


def load_seed() -> dict[int, str]:
    out: dict[int, str] = {}
    with SEED.open(newline="") as f:
        for row in csv.DictReader(f):
            cert = row.get("fdic_cert", "").strip()
            if cert:
                out[int(cert)] = row["bank_id"]
    return out


def load_quarters(seed: dict[int, str]) -> list[dict]:
    rows: list[dict] = []
    with CALL.open(newline="") as f:
        for row in csv.DictReader(f):
            cert = int(row["fdic_cert_number"])
            if cert not in seed:
                continue
            rd = parse_date(row["report_date"])
            if rd < START:
                continue
            rows.append(
                {
                    "fdic_cert_number": cert,
                    "bank_id": seed[cert],
                    "quarter_end_date": rd,
                    "total_deposits": parse_float(row.get("total_deposits")),
                    "npl_ratio": parse_float(row.get("npl_ratio")),
                }
            )
    rows.sort(key=lambda r: (r["fdic_cert_number"], r["quarter_end_date"]))
    return rows


def event_legs(row: dict, prev: dict | None) -> tuple[bool, bool]:
    """Return (deposit_outflow, npl_spike). Missing inputs => False."""
    deposit = False
    npl_spike = False
    if prev is not None:
        dep0, dep1 = prev["total_deposits"], row["total_deposits"]
        if dep0 is not None and dep1 is not None and dep0 > 0:
            if (dep1 - dep0) / dep0 <= DEPOSIT_THR:
                deposit = True
        npl0, npl1 = prev["npl_ratio"], row["npl_ratio"]
        if npl0 is not None and npl1 is not None and npl0 > 0:
            if npl1 / npl0 >= NPL_MULTIPLE_THR and npl1 > NPL_LEVEL_THR:
                npl_spike = True
    return deposit, npl_spike


def reason_str(deposit: bool, npl_spike: bool) -> str:
    parts = []
    if deposit:
        parts.append("deposit_outflow")
    if npl_spike:
        parts.append("npl_spike")
    return "|".join(parts)


def label_bank(bank_rows: list[dict]) -> None:
    """Set is_event_quarter, event_reason, distress_within_4q in place."""
    n = len(bank_rows)
    for i, row in enumerate(bank_rows):
        prev = bank_rows[i - 1] if i > 0 else None
        dep, npl = event_legs(row, prev)
        row["is_event_quarter"] = 1 if (dep or npl) else 0
        row["event_reason"] = reason_str(dep, npl) if (dep or npl) else ""
        row["distress_within_4q"] = 0

    for i, row in enumerate(bank_rows):
        if row["is_event_quarter"] != 1:
            continue
        # Four prior rows in the bank's quarterly panel (Q-1 … Q-4).
        for back in range(1, LOOKAHEAD_QUARTERS + 1):
            j = i - back
            if j < 0:
                break
            bank_rows[j]["distress_within_4q"] = 1


def drop_unclosed_per_bank(
    by_bank: dict[int, list[dict]],
    *,
    lookahead: int = LOOKAHEAD_QUARTERS,
) -> tuple[list[dict], int]:
    """Remove each bank's last `lookahead` rows (unobservable future)."""
    kept_by_bank: dict[int, list[dict]] = {}
    dropped_rows = 0
    for cert in sorted(by_bank):
        bank_rows = by_bank[cert]
        if len(bank_rows) <= lookahead:
            dropped_rows += len(bank_rows)
            continue
        kept_by_bank[cert] = bank_rows[:-lookahead]
        dropped_rows += lookahead

    flat = [r for cert in sorted(kept_by_bank) for r in kept_by_bank[cert]]
    return flat, dropped_rows


def build_rows(seed: dict[int, str]) -> list[dict]:
    rows = load_quarters(seed)
    by_bank: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        by_bank[r["fdic_cert_number"]].append(r)
    for bank_rows in by_bank.values():
        label_bank(bank_rows)
    flat, dropped = drop_unclosed_per_bank(by_bank)
    if dropped:
        print(f"  dropped {dropped} unclosed seed bank-quarters")
    return flat


def write_csv(rows: list[dict]) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as f:
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
    event_banks = {r["fdic_cert_number"] for r in events}
    pos_banks = {r["fdic_cert_number"] for r in positives}
    by_year = Counter(r["quarter_end_date"].year for r in events)
    by_reason = Counter(r["event_reason"] for r in events)
    # Event quarters should NOT also be training positives (by definition).
    both = sum(
        1 for r in rows if r["is_event_quarter"] == 1 and r["distress_within_4q"] == 1
    )

    print(f"Wrote {OUT.relative_to(ROOT)}")
    print(f"  rows:                 {len(rows)}")
    print(f"  is_event_quarter=1:   {len(events)}  (expect 33)")
    print(f"  distinct event banks: {len(event_banks)}  (expect 24)")
    print(f"  distress_within_4q=1: {len(positives)}  banks={len(pos_banks)}")
    print(
        f"  event∩within_4q:      {both}  "
        "(ok if >0: earlier event precedes a later one within 4q)"
    )
    print(
        "  events/year:          "
        + ", ".join(f"{y}:{by_year[y]}" for y in sorted(by_year))
    )
    print(
        "  event reasons:        "
        + ", ".join(f"{k}={v}" for k, v in sorted(by_reason.items()))
    )


def main() -> None:
    seed = load_seed()
    rows = build_rows(seed)
    write_csv(rows)
    summarize(rows)


if __name__ == "__main__":
    main()
