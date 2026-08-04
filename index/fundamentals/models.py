"""Select features, compare models, and pick one — all inside walk-forward folds.

Every choice that could see the answer is made inside a fold's own training set:
the feature ranking, the tier cut, the calibration. Selecting on the full panel
and then reporting fold metrics inflates everything downstream, and the effect is
invisible in the numbers.

Fold structure, for test year Y:

    train    prediction quarters <= (Y-1)Q1
    embargo  the rest of Y-1
    test     prediction quarters in Y

The embargo is not optional. Labels look 366 days forward, so a training row at
(Y-1)Q2 has a label that depends on events inside Y — train on it and the model
is told which banks fail in the test year.

Two families:
  XGBoost — sees the full panel, handles NaN natively, no uncertainty output
  GP      — O(n^3), so all positives plus a negative sample; needs imputation;
            the only one that emits latent_mean / latent_var, which is what the
            published score's confidence band is built from
"""
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/ming/project/PNC_Capstone")
from sklearn.gaussian_process import GaussianProcessClassifier as GPC
from sklearn.gaussian_process.kernels import ConstantKernel as C
from sklearn.gaussian_process.kernels import Matern
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

OUT = os.environ.get("MODEL_OUT", "/tmp/index_fundamentals")
FIRST_TEST, LAST_TEST = 2012, 2025
XGB_TIERS = [5, 10, 22, 50, 100]
GP_DIMS = [3, 5, 8, 12]
N_NEG = 3000                      # GP negative sample; positives all kept


def log(m):
    print(m, flush=True)


def metrics(y, p, k=100):
    if y.sum() < 3:
        return None
    o = np.argsort(-p)
    return {"n": len(y), "pos": int(y.sum()),
            "prauc": average_precision_score(y, p),
            "lift": average_precision_score(y, p) / y.mean(),
            "roc": roc_auc_score(y, p),
            "hits_at_100": int(y[o[:k]].sum()),
            "recall_at_100": float(y[o[:k]].sum() / y.sum()),
            "recall_at_500": float(y[o[:500]].sum() / y.sum())}


def fit_xgb(X, y):
    pos = max(int(y.sum()), 1)
    return XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.08,
        subsample=0.8, colsample_bytree=0.6,
        scale_pos_weight=(len(y) - pos) / pos,      # ~0.45% base rate
        tree_method="hist", eval_metric="aucpr",
        n_jobs=-1, random_state=0,
    ).fit(X, y)


def gains(model, feats):
    g = model.get_booster().get_score(importance_type="gain")
    return pd.Series({f: g.get(f"f{i}", 0.0) for i, f in enumerate(feats)})


def fit_gp(Xtr, ytr, Xte, dim, seed=0):
    """All positives plus a negative sample. length_scale is fixed rather than
    optimised — letting the optimiser run free sent it to ~1e4 and the kernel
    degenerated; sqrt(dim)*1.5 lands in the band that worked empirically."""
    rng = np.random.RandomState(seed)
    pos = np.where(ytr == 1)[0]
    neg = rng.choice(np.where(ytr == 0)[0], min(N_NEG, int((ytr == 0).sum())),
                     replace=False)
    idx = np.concatenate([pos, neg])
    sc = StandardScaler().fit(Xtr[idx])
    gp = GPC(kernel=C(10.0) * Matern(np.ones(dim) * np.sqrt(dim) * 1.5, nu=1.5),
             optimizer=None, random_state=seed).fit(sc.transform(Xtr[idx]), ytr[idx])
    Z = sc.transform(Xte)
    p, lv = [], []
    for i in range(0, len(Z), 5000):                # full kernel matrix OOMs
        b = Z[i:i + 5000]
        p.append(gp.predict_proba(b)[:, 1])
        lv.append(gp.base_estimator_.latent_mean_and_variance(b)[1])
    return np.concatenate(p), np.concatenate(lv), len(idx)


def main():
    d = pd.read_parquet(f"{OUT}/panel.parquet")
    meta = json.load(open(f"{OUT}/fields.json"))
    feats = meta["candidates"]
    d["qd"] = pd.to_datetime(d.pred_q, format="%Y%m%d")
    d["dec"] = d.groupby("pred_q").log_assets.transform(
        lambda s: pd.qcut(s, 10, labels=False, duplicates="drop"))
    log(f"面板 {len(d):,} 行 | {int(d.y.sum()):,} 正例 | 候选特征 {len(feats)}")

    rows, votes, lvs = [], {f: 0 for f in feats}, []
    for Y in range(FIRST_TEST, LAST_TEST + 1):
        te = d[d.year == Y]
        if len(te) == 0:
            continue
        # Strict embargo, derived rather than hardcoded: a training row's label
        # window runs 366 days forward, so it must close before the first test
        # prediction is made. A fixed quarter-end cut left the last training
        # window overlapping the first test window by a day.
        cut = te.qd.min() - pd.Timedelta(days=366)
        tr = d[d.qd <= cut]
        if tr.y.sum() < 20 or te.y.sum() < 5:
            log(f"{Y}: 跳过(测试正例 {int(te.y.sum())})")
            continue
        ytr, yte = tr.y.values, te.y.values

        order = gains(fit_xgb(tr[feats], ytr), feats).sort_values(
            ascending=False).index.tolist()
        for f in order[:30]:
            votes[f] += 1

        preds = {}
        for n in XGB_TIERS:
            sub = order[:n]
            preds[f"xgb{n}"] = (fit_xgb(tr[sub], ytr).predict_proba(te[sub])[:, 1], sub)
        for dim in GP_DIMS:
            sub = order[:dim]
            med = tr[sub].median()
            p, lv, used = fit_gp(tr[sub].fillna(med).values, ytr,
                                 te[sub].fillna(med).values, dim)
            preds[f"gp{dim}"] = (p, sub)
            lvs.append({"year": Y, "dim": dim, "lv_median": float(np.median(lv)),
                        "gp_train_rows": used})
        rk = lambda x: pd.Series(x).rank(pct=True).values
        preds["gp5+xgb10"] = ((rk(preds["gp5"][0]) + rk(preds["xgb10"][0])) / 2, [])

        for name, (p, sub) in preds.items():
            m = metrics(yte, p)
            if m:
                rows.append({"year": Y, "model": name, "feats": ",".join(sub), **m})
            big = te.dec == 9                       # size bracket of the tracked 104
            mb = metrics(yte[big.values], p[big.values])
            if mb:
                rows.append({"year": Y, "model": f"{name}@D9", "feats": "", **mb})
        log(f"{Y}: 训练 {len(tr):,}/{int(ytr.sum())}正 → 测试 {len(te):,}/{int(yte.sum())}正")

    r = pd.DataFrame(rows)
    r.to_csv(f"{OUT}/per_fold.csv", index=False)
    pd.DataFrame(lvs).to_csv(f"{OUT}/latent_var.csv", index=False)

    agg = (r.groupby("model").apply(lambda g: pd.Series({
        "折": len(g), "正例": int(g.pos.sum()),
        "PR-AUC": np.average(g.prauc, weights=g.pos),
        "lift": np.average(g.lift, weights=g.pos),
        "ROC": np.average(g.roc, weights=g.pos),
        "top100和": int(g.hits_at_100.sum()),
        "recall@100": g.hits_at_100.sum() / g.pos.sum(),
        "recall@500": np.average(g.recall_at_500, weights=g.pos)}))
        .sort_values("PR-AUC", ascending=False))
    agg.to_csv(f"{OUT}/summary.csv")

    n_folds = r.year.nunique()
    stab = pd.Series(votes).sort_values(ascending=False)
    stab.to_csv(f"{OUT}/stability.csv")
    log(f"\n=== 特征稳定性({n_folds} 折,进 top-30 的折数)===")
    for f, v in stab.head(15).items():
        log(f"  {v:2d}/{n_folds}  {f}")

    log(f"\n=== 模型对照(按正例加权)===")
    log(f"{'模型':<16}{'折':>4}{'正例':>6}{'PR-AUC':>9}{'lift':>9}{'ROC':>8}"
        f"{'top100':>8}{'recall@500':>12}")
    for m, x in agg.iterrows():
        log(f"{m:<16}{int(x['折']):>4}{int(x['正例']):>6}{x['PR-AUC']:>9.3f}"
            f"{x['lift']:>9.1f}{x['ROC']:>8.3f}{int(x['top100和']):>8}"
            f"{x['recall@500']*100:>11.1f}%")

    lv = pd.DataFrame(lvs).groupby("dim").lv_median.median()
    log(f"\nGP latent_var 中位数(越小区间越窄): {dict(lv.round(3))}")
    log(f"\n写入 {OUT}/summary.csv, per_fold.csv, stability.csv, latent_var.csv")


if __name__ == "__main__":
    main()
