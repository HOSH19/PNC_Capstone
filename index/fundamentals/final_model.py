"""Fit the production model on every row whose label window has closed, and score.

There is no test set here, and that is not an oversight. The walk-forward run in
models.py evaluates the *recipe*; this fits that recipe to all available data and
inherits the estimate. Holding out a slice would only re-measure what the folds
already measured, on less data.

The training cut is set by label completeness, not by preference. The label asks
whether a bank failed within 366 days, so a row is only usable once those 366
days have elapsed. Rows nearer than that record 0 for "has not failed yet", which
is not the same as "will not fail" — training on them teaches the model that
recent quarters are safe.

Calibration runs on the most recent complete year, held out of the GP fit. Note
that it corrects the level, not the ranking, and only as well as the calibration
year's base rate matches the scoring period's — a falling failure rate leaves the
published probability biased high. That is why the score is anchored on
quantiles and the band is what gets published, not the raw probability.
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
from models import fit_gp, fit_xgb, gains

OUT = os.environ.get("MODEL_OUT", "/tmp/index_fundamentals")
TODAY = pd.Timestamp(os.environ.get("MODEL_TODAY", "2026-08-04"))
DIM = int(os.environ.get("MODEL_DIM", "50"))
FIXED_ORDER = os.environ.get("FIXED_ORDER", f"{OUT}/fixed_order.json")
CAL_YEARS = 1                     # most recent complete year, held out of the fit


def log(m):
    print(m, flush=True)


def main():
    d = pd.read_parquet(f"{OUT}/panel.parquet")
    feats = json.load(open(f"{OUT}/fields.json"))["candidates"]
    d["qd"] = pd.to_datetime(d.pred_q, format="%Y%m%d")

    # A label is only real once its 366-day window has closed.
    cut = TODAY - pd.Timedelta(days=366)
    usable, censored = d[d.qd <= cut], d[d.qd > cut]
    log(f"today {TODAY.date()} -> label closure requires prediction quarter <= {cut.date()}")
    log(f"  usable     {len(usable):,} rows | {int(usable.y.sum()):,} positives "
        f"| {usable.IDRSSD.nunique():,} banks")
    # features.py already drops the quarters whose label window cannot close
    # (DROP_UNCLOSED), so this is normally empty — the check stays as a guard in
    # case the panel is rebuilt without it.
    if len(censored):
        log(f"  censored   {len(censored):,} rows (prediction quarters "
            f"{censored.pred_q.min()[:6]}~{censored.pred_q.max()[:6]}) "
            "-- excluded, label window has not closed")
    else:
        log("  censored   0 rows (features.py already dropped unclosed quarters)")

    cal_from = usable.qd.max() - pd.DateOffset(years=CAL_YEARS)
    fit_set, cal_set = usable[usable.qd <= cal_from], usable[usable.qd > cal_from]
    log(f"  fit        {len(fit_set):,} rows | {int(fit_set.y.sum()):,} positives "
        f"(feature quarters {fit_set.quarter.min()[:6]}~{fit_set.quarter.max()[:6]})")
    log(f"  calibrate  {len(cal_set):,} rows | {int(cal_set.y.sum()):,} positives "
        f"(prediction quarters {cal_set.pred_q.min()[:6]}~{cal_set.pred_q.max()[:6]})")

    # The feature list is fixed, not re-ranked here. models.py ranks once on the
    # first fold's training window and every fold uses that same list, so the
    # walk-forward numbers describe this exact model. Re-ranking on the full
    # training set would produce a different list and break that correspondence.
    order = json.load(open(FIXED_ORDER))["all"]
    sub = order[:DIM]
    log(f"\nfinal {DIM} inputs (fixed list, from the first fold's training window):")
    for i, f in enumerate(sub, 1):
        log(f"  {i}. {f}")

    med = fit_set[sub].median()
    X = lambda t: t[sub].fillna(med).values
    y = fit_set.y.values
    p_cal, _, used = fit_gp(X(fit_set), y, X(cal_set), DIM)
    p_all, lv_all, _ = fit_gp(X(fit_set), y, X(d), DIM)
    from models import N_POS, N_NEG
    log(f"\nGP fits on {used:,} rows: {min(N_POS, int(y.sum())):,} positives "
        f"(sampled from {int(y.sum()):,}) + "
        f"{min(N_NEG, len(y) - int(y.sum())):,} negatives")

    calib = SC.fit_calibrator(p_cal, cal_set.y.values)
    anchors = SC.fit_score_anchors(SC.calibrate(calib, p_cal))
    pc = SC.calibrate(calib, p_all)
    sc = SC.prob_to_score(pc, anchors)
    lm = np.log(np.clip(p_all, 1e-9, 1 - 1e-9) / (1 - np.clip(p_all, 1e-9, 1 - 1e-9)))
    iv = SC.score_intervals(lm, lv_all, calib, anchors)

    out = d[["IDRSSD", "quarter", "pred_q", "year", "y"]].copy()
    out["prob_raw"], out["prob"] = p_all, pc
    out["score"] = sc
    out["latent_mean"], out["latent_var"] = lm, lv_all
    for k, v in iv.items():
        out[k] = v
    out["band"] = SC.band_of(sc)
    out["label_closed"] = (out.pred_q.map(
        lambda q: pd.Timestamp(f"{q[:4]}-{q[4:6]}-{q[6:]}")) <= cut)
    for c in sub:
        out[c] = d[c].values
    out.to_parquet(f"{OUT}/scores.parquet", index=False)

    json.dump({"dim": DIM, "features": sub, "n_fit": len(fit_set),
               "n_pos": int(y.sum()), "gp_train_rows": used,
               "fit_feature_quarters": [fit_set.quarter.min(), fit_set.quarter.max()],
               "fit_pred_quarters": [fit_set.pred_q.min(), fit_set.pred_q.max()],
               "cal_pred_quarters": [cal_set.pred_q.min(), cal_set.pred_q.max()],
               "anchors": anchors, "today": str(TODAY.date())},
              open(f"{OUT}/final_model.json", "w"), indent=1)

    latest = out[out.quarter == out.quarter.max()]
    log(f"\nlatest quarter {latest.quarter.iloc[0][:6]}, scored {len(latest):,} banks:")
    log(f"  bands  {dict(latest.band.value_counts())}")
    log(f"  scores {latest.score.min():.1f} ~ {latest.score.max():.1f}")
    log(f"  mean 80% interval width {(latest.score_hi_80 - latest.score_lo_80).mean():.1f} points")
    log(f"  mean 95% interval width {(latest.score_hi_95 - latest.score_lo_95).mean():.1f} points")

    ev = out[out.label_closed]
    log(f"\nband vs actual event rate (over {len(ev):,} label-closed rows):")
    for b, g in ev.groupby("band"):
        log(f"  {b:<10} {len(g):>7,} rows  actual event rate {g.y.mean()*100:.3f}%")
    log(f"\nwrote {OUT}/scores.parquet, final_model.json")


if __name__ == "__main__":
    main()
