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
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/ming/project/PNC_Capstone")
sys.path.insert(0, "/Users/ming/project/PNC_Capstone/index/fundamentals")
import score as SC
from models import fit_gp, fit_xgb, gains

OUT = os.environ.get("MODEL_OUT", "/tmp/index_fundamentals")
TODAY = pd.Timestamp(os.environ.get("MODEL_TODAY", "2026-08-04"))
DIM = int(os.environ.get("MODEL_DIM", "5"))
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
    log(f"今天 {TODAY.date()} → 标签闭合要求 预测季 <= {cut.date()}")
    log(f"  可用   {len(usable):,} 行 | {int(usable.y.sum()):,} 正例 "
        f"| {usable.IDRSSD.nunique():,} 家银行")
    log(f"  截断   {len(censored):,} 行(预测季 {censored.pred_q.min()[:6]}~"
        f"{censored.pred_q.max()[:6]})—— 排除,标签窗口未关闭")

    cal_from = usable.qd.max() - pd.DateOffset(years=CAL_YEARS)
    fit_set, cal_set = usable[usable.qd <= cal_from], usable[usable.qd > cal_from]
    log(f"  拟合   {len(fit_set):,} 行 | {int(fit_set.y.sum()):,} 正例 "
        f"(特征季 {fit_set.quarter.min()[:6]}~{fit_set.quarter.max()[:6]})")
    log(f"  校准   {len(cal_set):,} 行 | {int(cal_set.y.sum()):,} 正例 "
        f"(预测季 {cal_set.pred_q.min()[:6]}~{cal_set.pred_q.max()[:6]})")

    order = gains(fit_xgb(fit_set[feats], fit_set.y.values), feats).sort_values(
        ascending=False).index.tolist()
    sub = order[:DIM]
    log(f"\n最终 {DIM} 个输入(全量训练集上的 XGBoost gain 排序):")
    for i, f in enumerate(sub, 1):
        log(f"  {i}. {f}")

    med = fit_set[sub].median()
    X = lambda t: t[sub].fillna(med).values
    y = fit_set.y.values
    p_cal, _, used = fit_gp(X(fit_set), y, X(cal_set), DIM)
    p_all, lv_all, _ = fit_gp(X(fit_set), y, X(d), DIM)
    log(f"\nGP 实际训练用 {used:,} 行(全部 {int(y.sum()):,} 正例 + 负采样)")

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
    log(f"\n最新一季 {latest.quarter.iloc[0][:6]} 打分 {len(latest):,} 家:")
    log(f"  分档 {dict(latest.band.value_counts())}")
    log(f"  分数 {latest.score.min():.1f} ~ {latest.score.max():.1f}")
    log(f"  80% 区间平均宽度 {(latest.score_hi_80 - latest.score_lo_80).mean():.1f} 分")
    log(f"  95% 区间平均宽度 {(latest.score_hi_95 - latest.score_lo_95).mean():.1f} 分")

    ev = out[out.label_closed]
    log(f"\n分档 vs 实际倒闭率(标签闭合的 {len(ev):,} 行):")
    for b, g in ev.groupby("band"):
        log(f"  {b:<10} {len(g):>7,} 行  实际倒闭率 {g.y.mean()*100:.3f}%")
    log(f"\n写入 {OUT}/scores.parquet, final_model.json")


if __name__ == "__main__":
    main()
