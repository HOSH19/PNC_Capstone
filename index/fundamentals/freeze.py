"""Freeze everything the quarterly scorer needs, so it never rebuilds the panel.

A Gaussian Process has no compact fitted form: predicting requires the whole
training set plus an n x n kernel matrix, roughly 194 MB pickled and fragile
across scikit-learn versions. Committing 5,000 training rows (~2 MB) and a
parameter file instead lets score_quarter.py refit in about 15 seconds. Same rows,
same seed, same fit — reproduced end to end the published score agrees to about
1e-6 on a 0-100 scale and the band is identical on every row, the residual being
ordinary BLAS non-determinism rather than anything in the data. That is what
keeps scores comparable from one quarter to the next; resampling each quarter
would let a bank's score drift for reasons that have nothing to do with the bank.

The 5,000 rows are exactly the ones final_model.py fits on: fit_gp draws
N_POS positives and N_NEG negatives from the label-closed training set with
RandomState(0), and this reproduces that draw rather than taking a new one.

Values are stored **after median imputation**, so there is no question at scoring
time about which medians were in force. The medians are stored too — a new
quarter's rows have their own gaps and must be filled with the training medians,
not with their own.

    MODEL_OUT=/tmp/index_fundamentals python3 index/fundamentals/freeze.py

Re-run when the model is re-frozen: move the training cut, re-run the pipeline,
run this, and increment model_version. Nothing else in the scoring path changes.
"""
import json
import os
import pathlib
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import score as SC
from models import N_NEG, N_POS, fit_gp, sample_index

OUT = os.environ.get("MODEL_OUT", "/tmp/index_fundamentals")
HERE = pathlib.Path(__file__).resolve().parent
TODAY = pd.Timestamp(os.environ.get("MODEL_TODAY", "2026-08-04"))
DIM = int(os.environ.get("MODEL_DIM", "50"))
MODEL_VERSION = os.environ.get("MODEL_VERSION", "gp50_prod_v1")
CAL_YEARS = 1
SEED = 0


def log(m):
    print(m, flush=True)


def main():
    d = pd.read_parquet(f"{OUT}/panel.parquet")
    d["qd"] = pd.to_datetime(d.pred_q, format="%Y%m%d")
    order = json.load(open(f"{OUT}/fixed_order.json"))
    sub = order["all"][:DIM]

    cut = TODAY - pd.Timedelta(days=366)
    usable = d[d.qd <= cut]
    cal_from = usable.qd.max() - pd.DateOffset(years=CAL_YEARS)
    fit_set, cal_set = usable[usable.qd <= cal_from], usable[usable.qd > cal_from]
    log(f"fit set    {len(fit_set):,} rows | {int(fit_set.y.sum()):,} positives "
        f"(feature quarters {fit_set.quarter.min()[:6]}~{fit_set.quarter.max()[:6]})")

    med = fit_set[sub].median()
    # A column that is entirely NaN in fit_set gives a NaN median, which would
    # be written into frozen_params.json as a bare NaN token — valid to Python's
    # json but not to any other parser — and imputed back into the frozen rows,
    # contradicting the whole point of storing them post-imputation. Dormant
    # today (worst column is ~35% NaN) and loud if it ever is not.
    if med.isna().any():
        raise SystemExit("median is NaN for: " + ", ".join(med[med.isna()].index))
    y = fit_set.y.values
    idx = sample_index(y, SEED)
    sample = fit_set.iloc[idx]
    frozen = sample[["IDRSSD", "quarter", "y"]].copy()
    for c in sub:
        frozen[c] = sample[c].fillna(med[c]).values
    log(f"frozen     {len(frozen):,} rows | {int(frozen.y.sum()):,} positives "
        f"+ {int((frozen.y == 0).sum()):,} negatives")

    # Refit here rather than reading final_model.json, so the calibrator and
    # anchors are provably the ones these exact rows produce.
    X = lambda t: t[sub].fillna(med).values
    p_cal, _, used = fit_gp(X(fit_set), y, X(cal_set), DIM)
    calib = SC.fit_calibrator(p_cal, cal_set.y.values)
    anchors = SC.fit_score_anchors(SC.calibrate(calib, p_cal))
    log(f"calibrator fitted on {len(cal_set):,} rows "
        f"(prediction quarters {cal_set.pred_q.min()[:6]}~{cal_set.pred_q.max()[:6]})")

    params = {
        "model_version": MODEL_VERSION,
        "dim": DIM,
        "features": sub,
        "raw_fields": sorted({"RC_2170", *(f for f in sub if f != "log_assets")}),
        "seed": SEED,
        "n_pos": N_POS,
        "n_neg": N_NEG,
        "gp_train_rows": used,
        "medians": {c: float(med[c]) for c in sub},
        # How a collected row becomes a model row. RC_2170 is in raw_fields but
        # is NOT one of the 50 features and is not a column in panel.parquet:
        # features.py consumes it and drops it. score_quarter.py must apply this
        # exactly, float32 cast included — the GP was fitted on float32 inputs,
        # and float64 would shift every prediction.
        "transform": {
            "denominator": "RC_2170",
            "zero_denominator": "treat RC_2170 == 0 as missing before dividing",
            "mdrm_item": "float32(raw_value / RC_2170)",
            "log_assets": "float32(log1p(RC_2170))",
            "impute": "fill remaining NaN with medians above, per column",
        },
        "platt": {"coef": float(calib.coef_[0][0]),
                  "intercept": float(calib.intercept_[0])},
        "anchors": anchors,
        "fit_feature_quarters": [fit_set.quarter.min(), fit_set.quarter.max()],
        "fit_pred_quarters": [fit_set.pred_q.min(), fit_set.pred_q.max()],
        "cal_pred_quarters": [cal_set.pred_q.min(), cal_set.pred_q.max()],
        "label_closure_cut": str(cut.date()),
        "frozen_as_of": str(TODAY.date()),
    }

    frozen.to_parquet(HERE / "train_sample.parquet", index=False)
    json.dump(params, open(HERE / "frozen_params.json", "w"), indent=1)
    sz = (HERE / "train_sample.parquet").stat().st_size / 1e6
    log(f"\nwrote {HERE}/train_sample.parquet ({sz:.1f} MB)")
    log(f"wrote {HERE}/frozen_params.json")
    log(f"model_version = {MODEL_VERSION}")
    log(f"collector needs {len(params['raw_fields'])} raw fields")


if __name__ == "__main__":
    main()
