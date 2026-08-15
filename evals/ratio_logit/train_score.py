"""Interpretable Call Report ratio + logistic rival (v2, fair training).

Fixes vs v1: 366-day train embargo; labels with unclosed tails dropped upstream.

Run:
  python3 evals/ratio_logit/train_score.py
"""

from __future__ import annotations

import csv
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
CALL = ROOT / "unified_ffiec_fdic_dataset" / "tables" / "fact_call_report.csv"
LABELS = ROOT / "evals" / "items" / "distress_bank_quarter_full.csv"
OUT = ROOT / "index" / "data" / "scores_ratios_logit_v2.csv"
COEF_OUT = ROOT / "evals" / "ratio_logit" / "coefficients_v2.csv"

SPLIT_DATE = date(2021, 12, 31)
EMBARGO_DAYS = 366
TRAIN_END = SPLIT_DATE - timedelta(days=EMBARGO_DAYS)
START = date(2017, 1, 1)
MODEL_VERSION = "ratios_logit_v2"

FEATURE_COLS = [
    "dep_qoq",
    "npl_ratio",
    "npl_qoq_mult",
    "liquidity_ratio",
    "urel_loss_over_assets",
    "tier1",
    "lla_ratio",
    "cre_over_assets",
    "log_assets",
    "asset_qoq",
]


def parse_float(s: str | None) -> float | None:
    if s is None or s == "":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def load_call() -> pd.DataFrame:
    rows = []
    with CALL.open(newline="") as f:
        for row in csv.DictReader(f):
            rd = date.fromisoformat(row["report_date"][:10])
            if rd < START:
                continue
            assets = parse_float(row.get("total_assets"))
            dep = parse_float(row.get("total_deposits"))
            npl = parse_float(row.get("npl_ratio"))
            urel = parse_float(row.get("securities_unrealized_loss"))
            cre = parse_float(row.get("cre_loans"))
            rows.append(
                {
                    "fdic_cert_number": int(row["fdic_cert_number"]),
                    "quarter_end_date": rd,
                    "total_assets": assets,
                    "total_deposits": dep,
                    "npl_ratio": npl,
                    "liquidity_ratio": parse_float(row.get("liquidity_ratio")),
                    "tier1": parse_float(row.get("tier1_capital_ratio")),
                    "lla_ratio": parse_float(row.get("loan_loss_allowance_ratio")),
                    "securities_unrealized_loss": urel,
                    "cre_loans": cre,
                }
            )
    return (
        pd.DataFrame(rows)
        .sort_values(["fdic_cert_number", "quarter_end_date"])
        .reset_index(drop=True)
    )


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("fdic_cert_number", sort=False)
    prev_dep = g["total_deposits"].shift(1)
    prev_npl = g["npl_ratio"].shift(1)
    prev_assets = g["total_assets"].shift(1)

    out = df.copy()
    out["dep_qoq"] = np.where(
        prev_dep.notna() & df["total_deposits"].notna() & (prev_dep > 0),
        (df["total_deposits"] - prev_dep) / prev_dep,
        np.nan,
    )
    out["npl_qoq_mult"] = np.where(
        prev_npl.notna() & df["npl_ratio"].notna() & (prev_npl > 0),
        df["npl_ratio"] / prev_npl,
        np.nan,
    )
    out["asset_qoq"] = np.where(
        prev_assets.notna() & df["total_assets"].notna() & (prev_assets > 0),
        (df["total_assets"] - prev_assets) / prev_assets,
        np.nan,
    )
    out["urel_loss_over_assets"] = np.where(
        df["securities_unrealized_loss"].notna()
        & df["total_assets"].notna()
        & (df["total_assets"] > 0),
        df["securities_unrealized_loss"] / df["total_assets"],
        np.nan,
    )
    out["cre_over_assets"] = np.where(
        df["cre_loans"].notna()
        & df["total_assets"].notna()
        & (df["total_assets"] > 0),
        df["cre_loans"] / df["total_assets"],
        np.nan,
    )
    out["log_assets"] = np.where(
        df["total_assets"].notna() & (df["total_assets"] > 0),
        np.log(df["total_assets"]),
        np.nan,
    )
    return out


def load_labels() -> pd.DataFrame:
    lab = pd.read_csv(LABELS)
    lab["quarter_end_date"] = pd.to_datetime(lab["quarter_end_date"]).dt.date
    lab["fdic_cert_number"] = lab["fdic_cert_number"].astype(int)
    return lab[["fdic_cert_number", "quarter_end_date", "distress_within_4q"]]


def main() -> None:
    print("Loading Call Report…")
    call = add_features(load_call())
    lab = load_labels()
    df = call.merge(lab, on=["fdic_cert_number", "quarter_end_date"], how="inner")
    print(
        f"  joined={len(df):,} pos={int(df.distress_within_4q.sum()):,}  "
        f"train_end={TRAIN_END} split={SPLIT_DATE}"
    )

    train = df[df["quarter_end_date"] <= TRAIN_END]
    score = df[df["quarter_end_date"] > SPLIT_DATE]
    print(
        f"  train {len(train):,}/{int(train.distress_within_4q.sum()):,}pos  "
        f"score {len(score):,}/{int(score.distress_within_4q.sum()):,}pos"
    )

    pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    penalty="l2",
                    C=1.0,
                    class_weight="balanced",
                    max_iter=2000,
                    solver="lbfgs",
                    random_state=0,
                ),
            ),
        ]
    )
    pipe.fit(
        train[FEATURE_COLS].to_numpy(dtype=float),
        train["distress_within_4q"].to_numpy(dtype=int),
    )
    proba = pipe.predict_proba(score[FEATURE_COLS].to_numpy(dtype=float))[:, 1]

    clf = pipe.named_steps["clf"]
    COEF_OUT.parent.mkdir(parents=True, exist_ok=True)
    with COEF_OUT.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["feature", "coef_standardized", "intercept"])
        for name, c in zip(FEATURE_COLS, clf.coef_.ravel()):
            w.writerow([name, f"{c:.6f}", f"{clf.intercept_[0]:.6f}"])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["fdic_cert_number", "quarter_end_date", "risk_score", "model_version"]
        )
        for cert, qd, p in zip(
            score["fdic_cert_number"], score["quarter_end_date"], proba
        ):
            qd_s = qd.isoformat() if isinstance(qd, date) else str(qd)[:10]
            w.writerow([int(cert), qd_s, float(p) if p == p else 0.0, MODEL_VERSION])
    print(f"Wrote {OUT.relative_to(ROOT)} and {COEF_OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
