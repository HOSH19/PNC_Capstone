"""Stronger rivals to Ming's gp50 — fair-training edition (v2).

Fixes vs v1:
  - 366-day embargo: train only on quarters whose 4Q label window closes
    on/before the protocol split (no post-split event peek)
  - Labels come from distress_bank_quarter_full.csv with unclosed tails dropped
  - dep_outflow_soft leaves NaN (no fillna(0) as "no outflow")
  - Label-adjacent features kept (same policy as Ming's delivered GP)

Models:
  - hgb_eng_v2   HistGradientBoosting
  - xgb_eng_v2   XGBoost

Run:
  python3 evals/ratio_logit/train_score_boost.py
"""

from __future__ import annotations

import csv
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[2]
CALL = ROOT / "unified_ffiec_fdic_dataset" / "tables" / "fact_call_report.csv"
LABELS = ROOT / "evals" / "items" / "distress_bank_quarter_full.csv"
OUT_DIR = ROOT / "index" / "data"
FEAT_OUT = ROOT / "evals" / "ratio_logit" / "feature_list_eng_v2.txt"

SPLIT_DATE = date(2021, 12, 31)
# 4Q lookahead ≈ 365d; match Ming walk-forward embargo constant.
EMBARGO_DAYS = 366
TRAIN_END = SPLIT_DATE - timedelta(days=EMBARGO_DAYS)  # 2020-12-30
START = date(2017, 1, 1)


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
            rows.append(
                {
                    "fdic_cert_number": int(row["fdic_cert_number"]),
                    "quarter_end_date": rd,
                    "total_assets": parse_float(row.get("total_assets")),
                    "total_deposits": parse_float(row.get("total_deposits")),
                    "tier1": parse_float(row.get("tier1_capital_ratio")),
                    "total_capital": parse_float(row.get("total_capital_ratio")),
                    "npl_ratio": parse_float(row.get("npl_ratio")),
                    "lla_ratio": parse_float(row.get("loan_loss_allowance_ratio")),
                    "liquidity_ratio": parse_float(row.get("liquidity_ratio")),
                    "urel": parse_float(row.get("securities_unrealized_loss")),
                    "cre_loans": parse_float(row.get("cre_loans")),
                }
            )
    return (
        pd.DataFrame(rows)
        .sort_values(["fdic_cert_number", "quarter_end_date"])
        .reset_index(drop=True)
    )


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=a.index, dtype=float)
    ok = a.notna() & b.notna() & (b != 0)
    out.loc[ok] = a.loc[ok] / b.loc[ok]
    return out


def _qoq(cur: pd.Series, prev: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=cur.index, dtype=float)
    ok = cur.notna() & prev.notna() & (prev > 0)
    out.loc[ok] = (cur.loc[ok] - prev.loc[ok]) / prev.loc[ok]
    return out


def engineer(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    g = df.groupby("fdic_cert_number", sort=False)
    assets = df["total_assets"]

    feat = pd.DataFrame(index=df.index)
    feat["log_assets"] = np.where(
        assets.notna() & (assets > 0), np.log(assets), np.nan
    )
    feat["dep_over_assets"] = _safe_div(df["total_deposits"], assets)
    feat["cre_over_assets"] = _safe_div(df["cre_loans"], assets)
    feat["urel_over_assets"] = _safe_div(df["urel"], assets)
    feat["npl_ratio"] = df["npl_ratio"]
    feat["tier1"] = df["tier1"]
    feat["total_capital"] = df["total_capital"]
    feat["lla_ratio"] = df["lla_ratio"]
    feat["liquidity_ratio"] = df["liquidity_ratio"]
    feat["tier1_gap_to_8"] = df["tier1"] - 8.0
    feat["npl_over_lla"] = _safe_div(df["npl_ratio"], df["lla_ratio"])

    prev_dep = g["total_deposits"].shift(1)
    prev_npl = g["npl_ratio"].shift(1)
    prev_assets = g["total_assets"].shift(1)
    prev_tier1 = g["tier1"].shift(1)
    prev_liq = g["liquidity_ratio"].shift(1)
    prev_cre = g["cre_loans"].shift(1)
    prev_urel = g["urel"].shift(1)

    feat["dep_qoq"] = _qoq(df["total_deposits"], prev_dep)
    feat["asset_qoq"] = _qoq(df["total_assets"], prev_assets)
    feat["npl_qoq_mult"] = _safe_div(df["npl_ratio"], prev_npl)
    feat["npl_qoq_delta"] = df["npl_ratio"] - prev_npl
    feat["tier1_qoq"] = df["tier1"] - prev_tier1
    feat["liq_qoq"] = df["liquidity_ratio"] - prev_liq
    feat["cre_qoq"] = _qoq(df["cre_loans"], prev_cre)
    feat["urel_qoq"] = df["urel"] - prev_urel

    # Soft event proximity — NaN stays NaN (do not treat missing as zero).
    feat["dep_outflow_soft"] = feat["dep_qoq"].clip(upper=0.0)
    feat["npl_spike_soft"] = np.where(
        feat["npl_qoq_mult"].notna() & feat["npl_ratio"].notna(),
        feat["npl_qoq_mult"] * (feat["npl_ratio"] > 2.0).astype(float),
        np.nan,
    )

    lag_sources = {
        "npl_ratio": feat["npl_ratio"],
        "tier1": feat["tier1"],
        "liquidity_ratio": feat["liquidity_ratio"],
        "dep_qoq": feat["dep_qoq"],
        "log_assets": feat["log_assets"],
    }
    bank = df["fdic_cert_number"]
    for col, series in lag_sources.items():
        tmp = pd.DataFrame({"bank": bank, "v": series})
        for lag in (1, 2, 3, 4):
            feat[f"{col}_l{lag}"] = tmp.groupby("bank")["v"].shift(lag)

    for col in ("dep_qoq", "npl_ratio", "tier1"):
        tmp = pd.DataFrame({"bank": bank, "v": feat[col]})
        rolled = tmp.groupby("bank")["v"]
        feat[f"{col}_roll4_mean"] = rolled.transform(
            lambda s: s.rolling(4, min_periods=2).mean()
        )
        feat[f"{col}_roll4_std"] = rolled.transform(
            lambda s: s.rolling(4, min_periods=2).std()
        )

    feat["npl_x_inv_tier1"] = feat["npl_ratio"] / feat["tier1"].clip(lower=0.1)
    feat["dep_qoq_x_log_assets"] = feat["dep_qoq"] * feat["log_assets"]
    feat["urel_x_log_assets"] = feat["urel_over_assets"] * feat["log_assets"]
    feat["cre_x_npl"] = feat["cre_over_assets"] * feat["npl_ratio"]

    feature_cols = list(feat.columns)
    out = df[["fdic_cert_number", "quarter_end_date"]].join(feat)
    return out, feature_cols


def load_labels() -> pd.DataFrame:
    lab = pd.read_csv(LABELS)
    lab["quarter_end_date"] = pd.to_datetime(lab["quarter_end_date"]).dt.date
    lab["fdic_cert_number"] = lab["fdic_cert_number"].astype(int)
    return lab[["fdic_cert_number", "quarter_end_date", "distress_within_4q"]]


def write_scores(path: Path, certs, dates, proba, model_version: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["fdic_cert_number", "quarter_end_date", "risk_score", "model_version"]
        )
        for cert, qd, p in zip(certs, dates, proba):
            qd_s = qd.isoformat() if isinstance(qd, date) else str(qd)[:10]
            score = float(p) if p == p else 0.0
            w.writerow([int(cert), qd_s, score, model_version])


def main() -> None:
    print("Loading + engineering…")
    call = load_call()
    feats, feature_cols = engineer(call)
    lab = load_labels()
    df = feats.merge(lab, on=["fdic_cert_number", "quarter_end_date"], how="inner")
    print(
        f"  rows={len(df):,} banks={df.fdic_cert_number.nunique():,} "
        f"feats={len(feature_cols)} pos={int(df.distress_within_4q.sum()):,}"
    )
    print(f"  train_end (embargo)={TRAIN_END}  split={SPLIT_DATE}")

    FEAT_OUT.write_text("\n".join(feature_cols) + "\n")

    # Embargo: label window must close on/before SPLIT_DATE.
    train = df[df["quarter_end_date"] <= TRAIN_END]
    # Score the protocol test region and beyond (harness filters test_end).
    score = df[df["quarter_end_date"] > SPLIT_DATE]
    X_tr = train[feature_cols].to_numpy(dtype=float)
    y_tr = train["distress_within_4q"].to_numpy(dtype=int)
    X_te = score[feature_cols].to_numpy(dtype=float)
    print(
        f"  train <= {TRAIN_END}: {len(train):,}/{int(y_tr.sum()):,}pos  "
        f"score > {SPLIT_DATE}: {len(score):,}/"
        f"{int(score.distress_within_4q.sum()):,}pos"
    )

    n_pos = max(int(y_tr.sum()), 1)
    n_neg = len(y_tr) - n_pos

    print("Fitting hgb_eng_v2…")
    sw = np.where(y_tr == 1, n_neg / n_pos, 1.0)
    hgb = HistGradientBoostingClassifier(
        max_depth=4,
        learning_rate=0.08,
        max_iter=300,
        min_samples_leaf=40,
        l2_regularization=0.1,
        random_state=0,
    )
    hgb.fit(X_tr, y_tr, sample_weight=sw)
    out_hgb = OUT_DIR / "scores_hgb_eng_v2.csv"
    write_scores(
        out_hgb,
        score["fdic_cert_number"],
        score["quarter_end_date"],
        hgb.predict_proba(X_te)[:, 1],
        "hgb_eng_v2",
    )
    print(f"Wrote {out_hgb.relative_to(ROOT)}")

    print("Fitting xgb_eng_v2…")
    xgb_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "clf",
                XGBClassifier(
                    n_estimators=400,
                    max_depth=4,
                    learning_rate=0.08,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    scale_pos_weight=n_neg / n_pos,
                    tree_method="hist",
                    eval_metric="aucpr",
                    n_jobs=-1,
                    random_state=0,
                ),
            ),
        ]
    )
    xgb_pipe.fit(X_tr, y_tr)
    out_xgb = OUT_DIR / "scores_xgb_eng_v2.csv"
    write_scores(
        out_xgb,
        score["fdic_cert_number"],
        score["quarter_end_date"],
        xgb_pipe.predict_proba(X_te)[:, 1],
        "xgb_eng_v2",
    )
    print(f"Wrote {out_xgb.relative_to(ROOT)}")

    clf = xgb_pipe.named_steps["clf"]
    score_map = clf.get_booster().get_score(importance_type="gain")
    gains = sorted(
        (
            (name, float(score_map.get(f"f{i}", 0.0)))
            for i, name in enumerate(feature_cols)
        ),
        key=lambda x: -x[1],
    )
    print("Top XGB features:")
    for name, g in gains[:12]:
        print(f"  {name:28s} {g:.2f}")


if __name__ == "__main__":
    main()
