"""Descriptive analysis for distress-event threshold selection (Shu Han Unit 1).

Reads local Call Report CSVs for the 104 seed banks (2017+), prints a markdown
report to stdout with distributions, candidate threshold counts, concentration
checks, and recommended thresholds. No label table is written.

Run:
  python3 eda/distress_threshold_eda.py > eda/reports/$(date +%F)_distress_threshold_eda.md
"""

from __future__ import annotations

import csv
import statistics
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEED = ROOT / "db" / "seed" / "banks.csv"
CALL = ROOT / "unified_ffiec_fdic_dataset" / "tables" / "fact_call_report.csv"

START = date(2017, 1, 1)
PERCENTILES = (1, 5, 10, 50, 90, 95, 99)


def parse_float(s: str | None) -> float | None:
    if s is None or s == "":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def parse_date(s: str) -> date:
    # report_date is YYYY-MM-DD
    y, m, d = s[:10].split("-")
    return date(int(y), int(m), int(d))


def pctile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    k = (len(xs) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    if f == c:
        return xs[f]
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def fmt(v: float | None, digits: int = 2) -> str:
    if v is None:
        return "—"
    return f"{v:.{digits}f}"


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "---|" * len(headers),
    ]
    for r in rows:
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines)


def load_seed_certs() -> dict[int, str]:
    """fdic_cert_number -> bank_id."""
    out: dict[int, str] = {}
    with SEED.open(newline="") as f:
        for row in csv.DictReader(f):
            cert = row.get("fdic_cert", "").strip()
            if not cert:
                continue
            out[int(cert)] = row["bank_id"]
    return out


def load_seed_quarters(seed: dict[int, str]) -> list[dict]:
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
                    "report_date": rd,
                    "total_assets": parse_float(row.get("total_assets")),
                    "total_deposits": parse_float(row.get("total_deposits")),
                    "npl_ratio": parse_float(row.get("npl_ratio")),
                    "tier1_capital_ratio": parse_float(row.get("tier1_capital_ratio")),
                    "liquidity_ratio": parse_float(row.get("liquidity_ratio")),
                    "securities_unrealized_loss": parse_float(
                        row.get("securities_unrealized_loss")
                    ),
                }
            )
    rows.sort(key=lambda r: (r["fdic_cert_number"], r["report_date"]))
    return rows


def add_derived(rows: list[dict]) -> None:
    by_bank: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        by_bank[r["fdic_cert_number"]].append(r)

    for bank_rows in by_bank.values():
        prev = None
        for r in bank_rows:
            r["deposit_qoq"] = None
            r["npl_multiple"] = None
            # Loss as % of assets (negative = unrealized loss). Dollars are
            # thousands of USD, same units as total_assets.
            loss, assets = r["securities_unrealized_loss"], r.get("total_assets")
            if loss is not None and assets is not None and assets > 0:
                r["unrealized_loss_pct_assets"] = 100.0 * loss / assets
            else:
                r["unrealized_loss_pct_assets"] = None
            if prev is not None:
                dep0, dep1 = prev["total_deposits"], r["total_deposits"]
                if dep0 is not None and dep1 is not None and dep0 > 0:
                    r["deposit_qoq"] = (dep1 - dep0) / dep0
                npl0, npl1 = prev["npl_ratio"], r["npl_ratio"]
                if npl0 is not None and npl1 is not None and npl0 > 0:
                    r["npl_multiple"] = npl1 / npl0
            prev = r


def col_values(rows: list[dict], key: str) -> list[float]:
    return [r[key] for r in rows if r.get(key) is not None]


def distribution_block(name: str, values: list[float], unit: str = "") -> list[str]:
    lines = [f"### {name}\n"]
    lines.append(f"n = {len(values)}" + (f" ({unit})" if unit else "") + "\n")
    if not values:
        lines.append("No non-null values.\n")
        return lines
    headers = ["stat"] + [f"p{p}" for p in PERCENTILES] + ["min", "max", "mean"]
    row = (
        ["value"]
        + [fmt(pctile(values, p), 4 if abs(pctile(values, 50) or 0) < 1 else 2)
           for p in PERCENTILES]
        + [
            fmt(min(values), 4 if abs(min(values)) < 1 else 2),
            fmt(max(values), 4 if abs(max(values)) < 1 else 2),
            fmt(statistics.fmean(values), 4 if abs(statistics.fmean(values)) < 1 else 2),
        ]
    )
    # Recompute with consistent digits for ratios near zero
    digits = 4 if max(abs(v) for v in values) < 2 else 2
    row = (
        ["value"]
        + [fmt(pctile(values, p), digits) for p in PERCENTILES]
        + [
            fmt(min(values), digits),
            fmt(max(values), digits),
            fmt(statistics.fmean(values), digits),
        ]
    )
    lines.append(md_table(headers, [row]))
    lines.append("")
    return lines


def event_stats(
    rows: list[dict], predicate, label: str
) -> tuple[str, int, int, dict[int, int], dict[int, int]]:
    """Return markdown summary bits for events matching predicate(row)->bool."""
    hits = [r for r in rows if predicate(r)]
    banks = {r["fdic_cert_number"] for r in hits}
    by_year: dict[int, int] = Counter(r["report_date"].year for r in hits)
    by_bank: dict[int, int] = Counter(r["fdic_cert_number"] for r in hits)
    return label, len(hits), len(banks), dict(by_year), dict(by_bank)


def concentration_note(by_bank: dict[int, int], seed: dict[int, str], n_events: int) -> str:
    if n_events == 0:
        return "no events"
    top_cert, top_n = max(by_bank.items(), key=lambda x: x[1])
    pct = 100.0 * top_n / n_events
    flag = " **CONCENTRATED**" if pct >= 25 else ""
    return f"top bank `{seed.get(top_cert, top_cert)}` = {top_n}/{n_events} ({pct:.0f}%){flag}"


def year_row(by_year: dict[int, int]) -> str:
    if not by_year:
        return "—"
    return ", ".join(f"{y}:{by_year[y]}" for y in sorted(by_year))


def main() -> None:
    p = print
    seed = load_seed_certs()
    rows = load_seed_quarters(seed)
    add_derived(rows)

    certs_present = {r["fdic_cert_number"] for r in rows}
    missing = sorted(set(seed) - certs_present)

    unrealized = col_values(rows, "securities_unrealized_loss")
    unrealized_pct = col_values(rows, "unrealized_loss_pct_assets")
    unrealized_nonempty = len(unrealized)
    unrealized_total = len(rows)

    p("# Distress threshold EDA — Unit 1\n")
    p("Generated by `eda/distress_threshold_eda.py`. Local CSVs only "
      "(seed banks × Call Reports, `report_date >= 2017-01-01`).\n")
    p("**Locked for later units:** when a rule fires in quarter Q, the 4 prior "
      "quarters get `distress_within_4q = 1`. This report only proposes what "
      "counts as an *event* in Q.\n")

    p("## 0. Coverage\n")
    p(md_table(
        ["metric", "value"],
        [
            ["seed banks in `banks.csv`", str(len(seed))],
            ["seed banks with ≥1 Call Report row 2017+", str(len(certs_present))],
            ["seed banks missing 2017+ Call Report", str(len(missing))],
            ["bank × quarter rows", str(len(rows))],
            [
                "securities_unrealized_loss non-null",
                f"{unrealized_nonempty}/{unrealized_total} "
                f"({100 * unrealized_nonempty / max(unrealized_total, 1):.1f}%)",
            ],
        ],
    ))
    p()
    if missing:
        p("Missing certs / bank_ids: " +
          ", ".join(f"{c} (`{seed[c]}`)" for c in missing) + "\n")
    else:
        p("All 104 seed banks have at least one 2017+ Call Report row.\n")
    if unrealized_nonempty == 0:
        p("`securities_unrealized_loss` is empty in this build — skipped as a "
          "candidate signal.\n")
    else:
        p("`securities_unrealized_loss` is populated; analyzed as "
          "**% of total assets** (same thousands-USD units). Negative = mark-to-market loss.\n")

    deposit_qoq = col_values(rows, "deposit_qoq")
    npl = col_values(rows, "npl_ratio")
    npl_mult = col_values(rows, "npl_multiple")
    tier1 = col_values(rows, "tier1_capital_ratio")
    liq = col_values(rows, "liquidity_ratio")

    p("## 1. Distributions\n")
    for block in (
        distribution_block("Deposit QoQ change", deposit_qoq, "fraction, e.g. -0.10 = −10%"),
        distribution_block("NPL ratio", npl, "percent"),
        distribution_block("NPL QoQ multiple (npl_t / npl_{t-1}, prior > 0)", npl_mult),
        distribution_block("Tier-1 capital ratio", tier1, "percent"),
        distribution_block("Liquidity ratio", liq, "percent"),
        distribution_block(
            "Unrealized securities P&L / assets",
            unrealized_pct,
            "percent of assets; negative = loss",
        ),
    ):
        for line in block:
            p(line)

    # --- candidate threshold counts ---
    candidates = []

    def add_cand(label, pred):
        candidates.append(event_stats(rows, pred, label))

    add_cand("deposits ≤ −5%", lambda r: r["deposit_qoq"] is not None and r["deposit_qoq"] <= -0.05)
    add_cand("deposits ≤ −10% (placeholder)", lambda r: r["deposit_qoq"] is not None and r["deposit_qoq"] <= -0.10)
    add_cand("deposits ≤ −15%", lambda r: r["deposit_qoq"] is not None and r["deposit_qoq"] <= -0.15)
    add_cand("deposits ≤ −20%", lambda r: r["deposit_qoq"] is not None and r["deposit_qoq"] <= -0.20)

    add_cand("NPL > 1%", lambda r: r["npl_ratio"] is not None and r["npl_ratio"] > 1.0)
    add_cand("NPL > 2%", lambda r: r["npl_ratio"] is not None and r["npl_ratio"] > 2.0)
    add_cand("NPL > 3%", lambda r: r["npl_ratio"] is not None and r["npl_ratio"] > 3.0)

    add_cand(
        "NPL ≥1.5× and NPL > 2% (placeholder)",
        lambda r: (
            r["npl_multiple"] is not None
            and r["npl_ratio"] is not None
            and r["npl_multiple"] >= 1.5
            and r["npl_ratio"] > 2.0
        ),
    )
    add_cand(
        "NPL ≥2× and NPL > 2%",
        lambda r: (
            r["npl_multiple"] is not None
            and r["npl_ratio"] is not None
            and r["npl_multiple"] >= 2.0
            and r["npl_ratio"] > 2.0
        ),
    )

    add_cand("tier1 < 8%", lambda r: r["tier1_capital_ratio"] is not None and r["tier1_capital_ratio"] < 8.0)
    add_cand("tier1 < 6%", lambda r: r["tier1_capital_ratio"] is not None and r["tier1_capital_ratio"] < 6.0)
    add_cand("tier1 < 4%", lambda r: r["tier1_capital_ratio"] is not None and r["tier1_capital_ratio"] < 4.0)

    # liquidity low-tail from distribution: use p5/p1 as candidates once computed
    liq_p5 = pctile(liq, 5)
    liq_p1 = pctile(liq, 1)
    if liq_p5 is not None:
        add_cand(
            f"liquidity < p5 ({fmt(liq_p5)})",
            lambda r, thr=liq_p5: r["liquidity_ratio"] is not None and r["liquidity_ratio"] < thr,
        )
    if liq_p1 is not None:
        add_cand(
            f"liquidity < p1 ({fmt(liq_p1)})",
            lambda r, thr=liq_p1: r["liquidity_ratio"] is not None and r["liquidity_ratio"] < thr,
        )

    add_cand(
        "unrealized loss ≤ −2% of assets",
        lambda r: (
            r["unrealized_loss_pct_assets"] is not None
            and r["unrealized_loss_pct_assets"] <= -2.0
        ),
    )
    add_cand(
        "unrealized loss ≤ −5% of assets",
        lambda r: (
            r["unrealized_loss_pct_assets"] is not None
            and r["unrealized_loss_pct_assets"] <= -5.0
        ),
    )
    add_cand(
        "unrealized loss ≤ −8% of assets",
        lambda r: (
            r["unrealized_loss_pct_assets"] is not None
            and r["unrealized_loss_pct_assets"] <= -8.0
        ),
    )

    # Combined placeholder from role doc
    def placeholder_or(r):
        dep = r["deposit_qoq"] is not None and r["deposit_qoq"] <= -0.10
        npl_spike = (
            r["npl_multiple"] is not None
            and r["npl_ratio"] is not None
            and r["npl_multiple"] >= 1.5
            and r["npl_ratio"] > 2.0
        )
        return dep or npl_spike

    add_cand("OR placeholder (dep≤−10% OR NPL≥1.5×&>2%)", placeholder_or)

    p("## 2. Candidate threshold counts\n")
    p("An *event* here is a bank × quarter where the rule fires (not yet "
      "expanded to a 4-quarter lookahead).\n")
    count_rows = []
    for label, n_ev, n_banks, by_year, by_bank in candidates:
        count_rows.append([
            label,
            str(n_ev),
            str(n_banks),
            year_row(by_year),
            concentration_note(by_bank, seed, n_ev),
        ])
    p(md_table(
        ["rule", "events", "distinct banks", "events/year", "concentration"],
        count_rows,
    ))
    p()

    # Detail for placeholder OR and recommended OR
    p("## 3. Placeholder probe (sanity check)\n")
    _, n_ev, n_banks, by_year, by_bank = event_stats(
        rows, placeholder_or, "placeholder OR"
    )
    p(f"Role-doc probe expected ~33 events / 24 banks. "
      f"**This run: {n_ev} events / {n_banks} banks.**\n")
    p("Events by year: " + year_row(by_year) + "\n")
    top = sorted(by_bank.items(), key=lambda x: -x[1])[:10]
    if top:
        p(md_table(
            ["bank_id", "fdic_cert", "events"],
            [[seed[c], str(c), str(n)] for c, n in top],
        ))
        p()

    # --- recommendations from distributions ---
    # Aim: tens of events, many banks, not one-bank dominated.
    dep_p5 = pctile(deposit_qoq, 5)
    dep_p1 = pctile(deposit_qoq, 1)
    npl_p95 = pctile(npl, 95)
    npl_p99 = pctile(npl, 99)
    t1_p5 = pctile(tier1, 5)
    t1_p1 = pctile(tier1, 1)
    u_p1 = pctile(unrealized_pct, 1)
    u_p5 = pctile(unrealized_pct, 5)

    _, t1_ev, t1_banks, _, _ = event_stats(
        rows,
        lambda r: r["tier1_capital_ratio"] is not None and r["tier1_capital_ratio"] < 8.0,
        "t1",
    )
    include_t1 = t1_ev > 0

    _, u5_ev, u5_banks, _, u5_by = event_stats(
        rows,
        lambda r: (
            r["unrealized_loss_pct_assets"] is not None
            and r["unrealized_loss_pct_assets"] <= -5.0
        ),
        "u5",
    )
    u5_conc = (max(u5_by.values()) / u5_ev) if u5_ev else 0
    # Include unrealized as an OR leg only if it adds acute events without
    # flooding the label and without one-bank domination.
    include_unrealized = 5 <= u5_ev <= 80 and u5_banks >= 5 and u5_conc < 0.25

    _, ph_ev, ph_banks, _, ph_by_bank = event_stats(rows, placeholder_or, "ph")
    ph_conc = (max(ph_by_bank.values()) / ph_ev) if ph_ev else 0

    use_dep_15 = False
    if ph_ev > 80 or ph_conc >= 0.25:
        def tight_or(r):
            dep = r["deposit_qoq"] is not None and r["deposit_qoq"] <= -0.15
            npl_spike = (
                r["npl_multiple"] is not None
                and r["npl_ratio"] is not None
                and r["npl_multiple"] >= 1.5
                and r["npl_ratio"] > 2.0
            )
            return dep or npl_spike

        _, t_ev, t_banks, _, t_by = event_stats(rows, tight_or, "tight")
        t_conc = (max(t_by.values()) / t_ev) if t_ev else 0
        if 10 <= t_ev <= 80 and t_banks >= 8 and t_conc < ph_conc:
            use_dep_15 = True

    dep_thr = -0.15 if use_dep_15 else -0.10
    dep_thr_pct = int(abs(dep_thr) * 100)

    def final_rule(r):
        dep = r["deposit_qoq"] is not None and r["deposit_qoq"] <= dep_thr
        npl_spike = (
            r["npl_multiple"] is not None
            and r["npl_ratio"] is not None
            and r["npl_multiple"] >= 1.5
            and r["npl_ratio"] > 2.0
        )
        hit = dep or npl_spike
        if include_t1:
            hit = hit or (
                r["tier1_capital_ratio"] is not None and r["tier1_capital_ratio"] < 8.0
            )
        if include_unrealized:
            hit = hit or (
                r["unrealized_loss_pct_assets"] is not None
                and r["unrealized_loss_pct_assets"] <= -5.0
            )
        return hit

    _, rec_ev, rec_banks, rec_year, rec_by_bank = event_stats(rows, final_rule, "rec")

    p("## 4. Recommended thresholds (proposal — not locked)\n")
    p("Chosen to land in the **tens of events / many banks** range, with "
      "concentration flagged if one bank owns ≥25% of events. "
      "Unit 2 will lock these into a definition doc after review.\n")

    if include_unrealized:
        unrealized_rule = "≤ −5% of assets"
        unrealized_why = (
            f"p5={fmt(u_p5)}, p1={fmt(u_p1)}. ≤−5% of assets is a severe HTM/AFS mark "
            f"({u5_ev} quarters / {u5_banks} banks) and is included as an OR leg."
        )
    else:
        unrealized_rule = (
            f"not used as hard event ({u5_ev} quarters / {u5_banks} banks at ≤−5%; "
            f"top-bank share {u5_conc:.0%})"
        )
        unrealized_why = (
            f"p5={fmt(u_p5)}, p1={fmt(u_p1)}. ≤−5% hits {u5_ev} quarters / {u5_banks} banks "
            f"(top-bank share {u5_conc:.0%}). Useful as a **GP feature**, but as a binary "
            f"event it either floods the label or clusters on a few securities-heavy banks. "
            f"Keep out of the answer-key OR for v1; revisit in Unit 2 if desired."
        )

    p(md_table(
        ["signal", "proposed rule", "rationale"],
        [
            [
                "Deposit outflow",
                f"QoQ ≤ −{dep_thr_pct}%",
                f"p5 of deposit QoQ is {fmt(dep_p5, 4)}; p1 is {fmt(dep_p1, 4)}. "
                f"−{dep_thr_pct}% sits in the left tail without requiring a near-zero "
                f"event count. "
                + ("Tightened from −10% because the looser OR was too large/concentrated. "
                   if use_dep_15 else
                   "Keeps the role-doc placeholder after checking the distribution."),
            ],
            [
                "NPL spike",
                "multiple ≥ 1.5× **and** NPL > 2%",
                f"NPL p95={fmt(npl_p95)}, p99={fmt(npl_p99)}. Level floor avoids "
                f"noise from tiny NPL bases; 1.5× matches a sharp deterioration.",
            ],
            [
                "Tier-1 capital",
                "< 8%" if include_t1 else "not used (no seed-bank breaches 2017+)",
                f"p5={fmt(t1_p5)}, p1={fmt(t1_p1)}. "
                + (f"{t1_ev} quarters / {t1_banks} banks below 8% — include as OR leg."
                   if include_t1 else
                   "Large seed banks stay well above regulatory stress on this series; "
                   "including a threshold that never fires adds nothing."),
            ],
            [
                "Liquidity",
                "not used as a hard event rule",
                f"p5={fmt(liq_p5)}, p1={fmt(liq_p1)}. Low liquidity is informative as a "
                f"**feature** for Ming's GP, but as a binary event it mostly tags "
                f"structurally low-liquidity business models rather than acute distress. "
                f"Defer to features, not the answer key.",
            ],
            [
                "Unrealized securities loss",
                unrealized_rule,
                unrealized_why,
            ],
        ],
    ))
    p()

    parts = [f"deposits ≤ −{dep_thr_pct}%", "NPL ≥1.5× and >2%"]
    if include_t1:
        parts.append("tier1 < 8%")
    if include_unrealized:
        parts.append("unrealized ≤ −5% of assets")
    p(f"**Proposed event rule (OR):** {' OR '.join(parts)}\n")

    p(md_table(
        ["metric", "value"],
        [
            ["event quarters (rule fires)", str(rec_ev)],
            ["distinct banks", str(rec_banks)],
            ["events/year", year_row(rec_year)],
            ["concentration", concentration_note(rec_by_bank, seed, rec_ev)],
        ],
    ))
    p()
    top = sorted(rec_by_bank.items(), key=lambda x: -x[1])[:10]
    if top:
        p("Top banks under proposed rule:\n")
        p(md_table(
            ["bank_id", "fdic_cert", "events"],
            [[seed[c], str(c), str(n)] for c, n in top],
        ))
        p()

    p("## 5. Notes for Unit 2\n")
    p("- Confirm or edit the proposed OR rule above before writing the definition doc.")
    p("- Label contract (locked): `fdic_cert_number`, `quarter_end_date`, "
      "`is_event_quarter`, `distress_within_4q` (1 on the 4 quarters before each event).")
    p("- Enforcement / OCC still deferred — fundamentals-only for this proposal.")
    p("- Do not train Ming's GP on these numbers until Unit 3 emits the table.\n")


if __name__ == "__main__":
    main()
