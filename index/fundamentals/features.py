"""Build the labelled panel and narrow the FFIEC field pool down to candidates.

Four filters, in order:

  1. **Stability across the MODELLED quarters.** A field must appear in >=90% of
     the quarters that actually carry labels. Counting over the extracted range
     instead — which includes 2001-2007 lookback padding — penalises every item
     introduced after 2007 and deleted 64 usable fields, the RC-O uninsured
     deposit block and RC-B held-to-maturity fair values among them.

  2. **Missingness vs the label.** Coverage for failing banks within 10 points of
     coverage for surviving banks, above 70% on both sides. Guards against a
     field whose *absence* signals distress.

  3. **Missingness vs SIZE.** Missingness can correlate with any covariate, and
     size is the dangerous one. A field can read 98%+ coverage overall with no
     label gap while every large bank is missing it: the affected banks are ~1.8%
     of the panel and nearly all survivors, so neither the aggregate nor the
     label test moves. The asymmetry matters because the tracked banks are all
     large. Compares the top 1% by assets against the rest, not deciles —
     FFIEC-031 filers are ~56% of the top 1% but only ~9.7% of the top decile.

  4. **Type.** Identifiers, dates and near-constant columns. An acquisition date
     divided by total assets passes every statistical filter.

Filters 2 and 3 touch the label, so they run on the first fold's training window
only — screening on the full panel lets the candidate list see every test year,
and no downstream fold structure can undo that.

Labels are derived from FDIC failure dates rather than joined from
fact_bank_quarter, so that a bank's final pre-failure filing survives — it has no
following quarter to join to. The prediction time is the report date plus the
45-day filing lag, the moment the figures actually become public.
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/Users/ming/project/PNC_Capstone")

SRC = os.environ.get("FFIEC_OUT", "/tmp/ffiec_raw")
OUT = os.environ.get("MODEL_OUT", "/tmp/index_fundamentals")
TA = "RC_2170"
MIN_QUARTER_FRAC = 0.90      # field must appear in >=90% of MODELLED quarters
MAX_LABEL_GAP = 0.10         # pos/neg coverage gap
MIN_SIDE_COV = 0.70          # coverage floor on both sides
MAX_SIZE_GAP = 0.10          # coverage gap, largest 1% vs the rest
LAG_DAYS = 45                # filing lag: report public ~30-35 days after quarter end
LABEL_START = "2008"         # DB labels begin 2008Q1; earlier quarters are lookback only
SCREEN_UNTIL = "20110515"    # label-touching screening sees only the first fold's
                             # training window, never a test year
BIG_FRAC = 0.01              # "largest" = top 1%; FFIEC-031 filers are ~56% of it
                             # but only ~9.7% of the top decile, so a decile
                             # comparison cannot resolve them


def log(m):
    print(m, flush=True)


def stable_fields():
    """Fields present in >=MIN_QUARTER_FRAC of the quarters that are actually modelled.

    Counting over every extracted quarter is wrong: 2001-2007 exist only as
    lookback padding, so an item introduced after 2007 — present in 100% of the
    73 modelled quarters — scores 73/101 = 72% and gets dropped. That deleted 64
    usable fields, among them the RC-O insured/uninsured deposit block and the
    RC-B held-to-maturity fair-value lines, i.e. the two signals behind the 2023
    failures, plus construction-loan nonaccrual items with univariate AUC up to
    0.858.
    """
    import collections
    import pyarrow.parquet as pq
    files = sorted(glob.glob(f"{SRC}/*.parquet"))
    modelled = [f for f in files if os.path.basename(f)[:4] >= LABEL_START]
    cnt = collections.Counter()
    for f in modelled:
        cnt.update(c for c in pq.ParquetFile(f).schema.names
                   if c not in ("IDRSSD", "quarter"))
    stable = sorted(k for k, v in cnt.items() if v >= MIN_QUARTER_FRAC * len(modelled))
    log(f"[1] 稳定性: {len(cnt)} 个字段中 {len(stable)} 个出现在 >={MIN_QUARTER_FRAC:.0%} "
        f"的【建模季度】({len(modelled)} 季,非全部 {len(files)} 季)")
    return files, stable


def labels():
    """Official FDIC failure label, computed from failure dates rather than joined.

    The obvious implementation — join the next quarter's `distress_within_4q`
    from fact_bank_quarter — silently loses the most informative rows. A bank
    files nothing in the quarter it fails, so it has no q+1 row, so the join
    returns NaN and its last filing is dropped. That removed 491 pre-failure
    filings, 20.5% of the recoverable positives, including SVB's 2022Q4 report
    (the one carrying the unrealised losses) and First Republic's 2023Q1.

    So derive it directly: prediction time T = quarter end + 45 days, and y = 1
    when the bank's failure falls within 366 days after T. Same window
    modeling.py uses, but it does not require the bank to still be filing.
    """
    from pipeline import db
    c = db.connect()
    cur = c.cursor()
    cur.execute("select fdic_cert_number, rssd_id, report_date from fact_call_report")
    m = pd.DataFrame(cur.fetchall())
    cur.execute("select fdic_cert_number, min(failure_date) fd from fact_distress_event "
                "where failure_date is not null group by 1")
    fail = pd.DataFrame(cur.fetchall())
    c.close()

    m["q"] = pd.to_datetime(m.report_date).dt.strftime("%Y%m%d")
    m["rssd_id"] = m.rssd_id.astype("int64")
    m = m[~m.duplicated(subset=["rssd_id", "q"])]

    # Prediction time is the report date plus the filing lag, not the next
    # quarter end. FFIEC allows 30-35 days; 45 gives a buffer, and it is the
    # earliest moment the quarter's figures are actually public. Rounding up to
    # the next quarter end instead throws away six weeks of warning and, worse,
    # discards banks that fail inside those six weeks: SVB's 2022Q4 filing —
    # the one carrying the unrealised losses — becomes unusable, because it
    # failed on 2023-03-10, three weeks before 2023Q1 end.
    qe = pd.to_datetime(m.q, format="%Y%m%d")
    pred = qe + pd.Timedelta(days=LAG_DAYS)
    m["pred_q"] = pred.dt.strftime("%Y%m%d")

    m = m.merge(fail, on="fdic_cert_number", how="left")
    delta = (pd.to_datetime(m.fd) - pd.to_datetime(m.pred_q, format="%Y%m%d")).dt.days
    m["dtd"] = delta.where(delta.between(0, 366))
    m["y"] = delta.between(0, 366).fillna(False).astype(np.int8)
    log(f"    标签:{int(m.y.sum()):,} 正例 / {len(m):,} 行 "
        f"(预测时点 = 财报期末 + {LAG_DAYS} 天;按 failure_date 直接推导)")
    return m.set_index(["rssd_id", "q"])[["y", "dtd", "pred_q"]]


def build(files, stable, key):
    parts = []
    for f in files:
        if os.path.basename(f)[:4] < LABEL_START:
            continue
        d = pd.read_parquet(f)
        d["IDRSSD"] = d.IDRSSD.astype("int64")
        cols = [c for c in stable if c in d.columns]
        d = d[["IDRSSD", "quarter"] + cols]
        for c in stable:
            if c not in d.columns:
                d[c] = np.nan
        d = d[["IDRSSD", "quarter"] + stable].astype({c: "float32" for c in stable})
        parts.append(d)
    d = pd.concat(parts, ignore_index=True)
    del parts
    idx = pd.MultiIndex.from_arrays([d.IDRSSD.values, d.quarter.values])
    d["y"] = key.y.reindex(idx).values
    d["pred_q"] = key.pred_q.reindex(idx).values
    d = d[d.y.notna() & d.pred_q.notna()].copy()
    d["y"] = d.y.astype(np.int8)
    d["year"] = d.pred_q.str[:4].astype(int)
    # days_to_distress is NOT written to the panel. It is derived from the
    # failure date and is populated for negatives too, so anything that picked
    # it up as a feature would be reading the answer.
    log(f"    面板 {len(d):,} 行 | {int(d.y.sum()):,} 正例 ({d.y.mean()*100:.3f}%) "
        f"| {d.IDRSSD.nunique():,} 家银行 | 特征季 {d.quarter.min()[:6]}~{d.quarter.max()[:6]}")
    return d


def degenerate(stable, d):
    """Identifiers, dates and near-constant columns that are not quantities.

    Nothing upstream catches these. 99.7% of banks report a literal 0, so
    coverage reads 100% on every axis and the near-zero variance leaves them as
    singleton correlation clusters. RI_9106 is an ACQUISITION DATE — a YYYYMMDD
    integer divided by total assets — and RCO_A545 is an FDIC certificate
    number. Both survived every filter with AUC 0.501.
    """
    import json as _json
    names = {}
    if os.path.exists("/tmp/mdrm_names.json"):
        names = _json.load(open("/tmp/mdrm_names.json"))
    BAD = ("DATE", "NUMBER", "NBR ", "CERTIFICATE", "RSSD", "IDENT", "CODE", "FLAG")
    out = []
    for c in stable:
        desc = names.get(c, "").upper()
        v = d[c]
        if any(b in desc for b in BAD) and "NBR OF FT" not in desc:
            out.append((c, f"描述含标识符/日期: {desc[:40]}"))
        elif v.notna().sum() and (v.fillna(0) == 0).mean() > 0.995:
            out.append((c, f"{(v.fillna(0)==0).mean()*100:.1f}% 为 0"))
        elif v.nunique(dropna=True) < 20:
            out.append((c, f"仅 {v.nunique(dropna=True)} 个不同取值"))
    return dict(out)


def screen(d, stable):
    """Filters 2 and 3 — missingness against the label, and against size.

    Both run on the first fold's training window only (`SCREEN_UNTIL`). Screening
    on the full panel lets the candidate list see every test year, which no
    downstream fold structure can undo.

    The size axis is deliberately not a decile comparison. FFIEC-031 filers are
    ~9.7% of the top asset decile, so a field missing for every one of them moves
    a decile gap by only ~0.09 — under any sane threshold. They are ~56% of the
    top 1%. And the decile has to be built with NaN assets forced into the top
    group: pd.qcut drops NaN rows entirely, which makes the coverage of the very
    field defining the deciles equal 1.0 by construction. Run against the v2
    panel, the decile version of this filter flags nothing — it could not have
    caught the bug it exists for.
    """
    s = d[d.pred_q <= SCREEN_UNTIL]
    log(f"    筛选窗口: 预测时点 <= {SCREEN_UNTIL}({len(s):,} 行, {int(s.y.sum())} 正例)")
    y = s.y.values
    pos, neg = s.loc[y == 1, stable].notna().mean(), s.loc[y == 0, stable].notna().mean()
    label_gap = (neg - pos).abs()
    ok_label = (label_gap <= MAX_LABEL_GAP) & (pos > MIN_SIDE_COV) & (neg > MIN_SIDE_COV)
    log(f"[2] 缺失 vs 标签: {int(ok_label.sum())}/{len(stable)} 通过")

    ta_rank = s.groupby("quarter")[TA].rank(pct=True, na_option="top")
    big_m, rest_m = ta_rank >= 1 - BIG_FRAC, ta_rank < 1 - BIG_FRAC
    big = s.loc[big_m, stable].notna().mean()
    rest = s.loc[rest_m, stable].notna().mean()
    size_gap = (rest - big).abs()
    ok_size = (size_gap <= MAX_SIZE_GAP) & (big > MIN_SIDE_COV)
    log(f"[3] 缺失 vs 规模: {int(ok_size.sum())}/{len(stable)} 通过 "
        f"(最大 {BIG_FRAC:.0%} vs 其余,差距<={MAX_SIZE_GAP:.0%} 且大行侧>{MIN_SIDE_COV:.0%})")
    log(f"    仅因规模轴被剔除: {int((ok_label & ~ok_size).sum())}  ← 标签轴看不到这一类")

    deg = degenerate(stable, d)
    ok_type = pd.Series({c: c not in deg for c in stable})
    log(f"[4] 剔除标识符/日期/近似常量: {len(deg)} 个")
    for c, why in list(deg.items())[:6]:
        log(f"      {c}: {why}")

    clean = [c for c in stable if ok_label[c] and ok_size[c] and ok_type[c]]
    diag = pd.DataFrame({"pos_cov": pos, "neg_cov": neg, "label_gap": label_gap,
                         "big_cov": big, "rest_cov": rest, "size_gap": size_gap,
                         "ok_label": ok_label, "ok_size": ok_size, "ok_type": ok_type})
    diag["drop_reason"] = pd.Series(deg)
    diag["clean"] = diag.index.isin(clean)
    return clean, diag


def normalize(d, feats):
    """Every field over total assets. Raw MDRM items are dollar amounts in
    $thousands; the reported capital ratios live in RCRI, which loses every
    field to the stability filter. Size stays available as log_assets."""
    ta = d[TA].replace(0, np.nan)
    for c in feats:
        d[c] = (d[c] / ta).astype("float32")
    d["log_assets"] = np.log1p(ta).astype("float32")
    return d, [c for c in feats if c != TA] + ["log_assets"]


def prune(d, feats, thresh=0.95):
    """Merge near-duplicates. RCB carries amortized cost and fair value of the
    same securities; RCN carries nested subtotals. Left separate, importance
    splits across the copies and each looks mediocre."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    from sklearn.metrics import roc_auc_score
    s = d.sample(min(60000, len(d)), random_state=0)
    corr = np.array(s[feats].corr(method="spearman").abs().fillna(0).values, copy=True)
    np.fill_diagonal(corr, 1.0)
    dist = (1 - corr + (1 - corr).T) / 2
    np.fill_diagonal(dist, 0)
    cl = fcluster(linkage(squareform(dist, checks=False), method="average"),
                  t=1 - thresh, criterion="distance")
    # AUC picks the cluster representative, so it touches the label — restrict it
    # to the first fold's training window like the other label-touching steps.
    a = d[d.pred_q <= SCREEN_UNTIL]
    auc = {}
    for c in feats:
        v, ok = a[c], a[c].notna()
        auc[c] = (max(roc_auc_score(a.y[ok], v[ok]), 1 - roc_auc_score(a.y[ok], v[ok]))
                  if ok.sum() > 1000 and a.y[ok].sum() > 10 else 0.5)
    reps, groups = [], {}
    for k in np.unique(cl):
        members = [feats[i] for i in np.where(cl == k)[0]]
        best = max(members, key=lambda c: auc[c])
        reps.append(best)
        groups[best] = members
    log(f"[4] 相关性剪枝 |rho|>={thresh}: {len(feats)} -> {len(reps)}(合并,非丢弃)")
    return reps, auc, groups


def main():
    os.makedirs(OUT, exist_ok=True)
    files, stable = stable_fields()
    key = labels()
    d = build(files, stable, key)
    clean, diag = screen(d, stable)
    diag.to_csv(f"{OUT}/field_screen.csv")
    d, feats = normalize(d, clean)
    feats, auc, groups = prune(d, feats)
    d[["IDRSSD", "quarter", "pred_q", "year", "y"] + feats].to_parquet(
        f"{OUT}/panel.parquet", index=False)
    json.dump({"stable": stable, "clean": clean, "candidates": feats,
               "auc": {k: float(v) for k, v in auc.items()},
               "corr_groups": groups},
              open(f"{OUT}/fields.json", "w"), indent=1)
    log(f"\n候选特征 {len(feats)} 个 -> {OUT}/panel.parquet")
    top = sorted(feats, key=lambda c: -auc.get(c, 0.5))[:12]
    log("单变量 AUC 最高的 12 个:")
    for c in top:
        log(f"  {auc[c]:.3f}  {c}")


if __name__ == "__main__":
    main()
