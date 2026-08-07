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
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

SRC = os.environ.get("FFIEC_OUT", "/tmp/ffiec_raw")
OUT = os.environ.get("MODEL_OUT", "/tmp/index_fundamentals")
# Committed next to this file, not derived at runtime — see degenerate().
MDRM_NAMES = os.environ.get(
    "MDRM_NAMES", str(pathlib.Path(__file__).resolve().parent / "mdrm_names.json"))
TA = "RC_2170"
MIN_QUARTER_FRAC = 0.90      # field must appear in >=90% of MODELLED quarters
MAX_LABEL_GAP = 0.10         # pos/neg coverage gap
MIN_SIDE_COV = 0.70          # coverage floor on both sides
MAX_SIZE_GAP = 0.10          # coverage gap, largest 1% vs the rest
LAG_DAYS = 45                # filing lag: report public ~30-35 days after quarter end
LABEL_START = "2017"         # RCN_1403/1407 (the NPL leg) first exist 2017Q1
SCREEN_UNTIL = "20190213"    # label-touching screening sees only the first fold's
                             # training window, never a test year
DEPOSIT_THR = -0.10          # Shu Han evals/distress_definition.md v1
NPL_MULTIPLE_THR = 1.5
NPL_LEVEL_THR = 2.0
HORIZON_Q = 4                # event in Q+1..Q+4 -> label 1 at Q
# A row's label needs HORIZON_Q forward quarters to exist. The last few quarters
# of any extract do not have them, so their positives are silently missing: the
# rate falls 10.9% -> 8.4% -> 6.0% -> 3.3% -> 0% across the final five quarters
# of this build. Those rows are dropped rather than left in the panel labelled 0.
DROP_UNCLOSED = True

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
    log(f"[1] stability: {len(stable)} of {len(cnt)} fields appear in "
        f">={MIN_QUARTER_FRAC:.0%} of MODELLED quarters "
        f"({len(modelled)} of {len(files)} total)")
    return files, stable


def labels():
    """Distress-state label, per `evals/distress_definition.md` v1 (owner: Shu Han).

    An event fires in quarter Q when either leg holds:

      deposit outflow   (dep_Q - dep_{Q-1}) / dep_{Q-1} <= -10%
      NPL spike         npl_Q / npl_{Q-1} >= 1.5  and  npl_Q > 2%

    A row at Q is labelled 1 when an event fires in Q+1..Q+4 — one year ahead,
    matching the horizon the failure model used.

    Two deliberate departures from that document, both widening scope rather
    than changing the rule:

    1. All filers, not the 104 seed banks. The 104-bank restriction gives 120
       positives; every filer gives ~16k. The document excludes the wider set
       for the *combined* eval, where the sentiment axis needs text coverage —
       that constraint binds on evaluation, not on training.
    2. Read from the FFIEC archives rather than `fact_call_report`, which puts
       both inputs on a different basis. Deposits become RCON+RCFN (domestic
       plus foreign offices) instead of domestic only, and the NPL ratio
       becomes consolidated. Three of the four seed-bank event disagreements
       trace to the deposit half; the fourth is `fact_call_report` reporting
       $18.0bn deposits and 5.56% NPL for RSSD 694904 in 2022Q1 where the
       source filing gives $38.09bn and 0.134%.

    2017Q1 is the floor: RCN_1403 / RCN_1407 do not exist before it, so the NPL
    leg cannot fire and the label would silently mean something different.
    """
    import glob
    RAW = os.environ.get("FFIEC_OUT", "/tmp/ffiec_raw")
    need = ["IDRSSD", "RC_2200", "RCN_1403", "RCN_1407", "RCCI_2122"]
    parts = []
    for f in sorted(glob.glob(f"{RAW}/*.parquet")):
        q = os.path.basename(f)[:8]
        if q[:4] < LABEL_START:
            continue
        d = pd.read_parquet(f)
        if "RCN_1403" not in d.columns:
            continue
        k = d[[c for c in need if c in d.columns]].copy()
        k["q"] = q
        parts.append(k)
    m = pd.concat(parts, ignore_index=True)
    m = m.rename(columns={"IDRSSD": "rssd_id"})
    m["rssd_id"] = m.rssd_id.astype("int64")
    m = m[~m.duplicated(subset=["rssd_id", "q"])].sort_values(["rssd_id", "q"])

    # NPL ratio, percent. Same numerator as fact_call_report
    # (unified_ffiec_fdic_dataset/scripts/ffiec_call_reports.py:128) but on the
    # consolidated basis, since extract.py already coalesced RCFD over RCON.
    npl_bal = m.RCN_1403.fillna(0) + m.RCN_1407.fillna(0)
    m["npl"] = np.where(m.RCCI_2122 > 0, npl_bal / m.RCCI_2122 * 100, np.nan)

    g = m.groupby("rssd_id")
    dep0, npl0 = g.RC_2200.shift(1), g.npl.shift(1)
    dep_leg = (((m.RC_2200 - dep0) / dep0 <= DEPOSIT_THR) & (dep0 > 0)).fillna(False)
    npl_leg = (((m.npl / npl0 >= NPL_MULTIPLE_THR) & (m.npl > NPL_LEVEL_THR)
                & (npl0 > 0))).fillna(False)
    m["ev"] = (dep_leg | npl_leg).astype(np.int8)

    # Label at Q asks whether an event fires in Q+1..Q+HORIZON_Q. Looking only
    # forward already excludes Q's own event, and an event quarter still gets
    # y=1 when a *later* event falls inside the horizon -- both as specified.
    ge = m.groupby("rssd_id").ev
    fwd = [ge.shift(-s).fillna(0).astype(np.int8) for s in range(1, HORIZON_Q + 1)]
    m["y"] = np.maximum.reduce(fwd)
    m["dtd"] = np.select([f.astype(bool) for f in fwd], list(range(1, HORIZON_Q + 1)),
                         default=np.nan)

    qe = pd.to_datetime(m.q, format="%Y%m%d")
    m["pred_q"] = (qe + pd.Timedelta(days=LAG_DAYS)).dt.strftime("%Y%m%d")

    if DROP_UNCLOSED:
        qs = sorted(m.q.unique())
        keep = qs[:-HORIZON_Q] if len(qs) > HORIZON_Q else qs
        n0 = len(m)
        m = m[m.q.isin(keep)]
        log(f"    dropped {len(qs) - len(keep)} report quarters whose label cannot close "
            f"({sorted(set(qs) - set(keep))}): {n0:,} -> {len(m):,} rows")

    # Count the legs over the rows that survive, not the pre-drop frame — the
    # two used to be printed on different bases and did not add up.
    kept = m.index if not DROP_UNCLOSED else m.index
    dl, nl = dep_leg.loc[kept], npl_leg.loc[kept]
    log(f"    labels: {int(m.y.sum()):,} positives / {len(m):,} rows / "
        f"{m[m.y == 1].rssd_id.nunique():,} banks "
        f"({int(m.ev.sum()):,} events: deposit leg {int(dl.sum()):,} · "
        f"NPL leg {int(nl.sum()):,} · both {int((dl & nl).sum()):,})")
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
    # dtd is NOT written to the panel. It counts quarters to the next event,
    # so anything that picked it up as a feature would be reading the answer.
    log(f"    panel {len(d):,} rows | {int(d.y.sum()):,} positives ({d.y.mean()*100:.3f}%) "
        f"| {d.IDRSSD.nunique():,} banks | feature quarters "
        f"{d.quarter.min()[:6]}~{d.quarter.max()[:6]}")
    return d


def degenerate(stable, d):
    """Identifiers, dates and near-constant columns that are not quantities.

    Nothing upstream catches these. 99.7% of banks report a literal 0, so
    coverage reads 100% on every axis and the near-zero variance leaves them as
    singleton correlation clusters. RI_9106 is an ACQUISITION DATE — a YYYYMMDD
    integer divided by total assets — and RCO_A545 is an FDIC certificate
    number. Both survived every filter with AUC 0.501.

    The name map is required, not optional. It used to be read from /tmp and
    skipped when absent, which is not a degraded screen but a silent one: a
    missing file drops nobody on the description axis and the candidate pool
    comes out 534 instead of 529, with no warning anywhere in the log. Build it
    with mdrm_names.py; the output is committed alongside this file.

    BAD matches as a substring and has no word boundary, so two real quantities
    are caught by accident: RC_2130 INVEST. IN UNCONSOLI(DATE)D SUBS & CO. and
    RCO_M963 NON-AGENCY RES(IDENT)IAL MBS. Left as-is deliberately. Both are
    noise on this label — univariate AUC 0.510 and screened out anyway
    respectively, against 0.648 for the strongest field — and the pool of 529 in
    REPORT §3 is defined by this behaviour, so tightening the match would move
    the documented funnel and break reproduction of the delivered scores for no
    measurable gain. Revisit when the feature set is next rebuilt.
    """
    import json as _json
    if not os.path.exists(MDRM_NAMES):
        raise SystemExit(
            f"missing the field-description file {MDRM_NAMES}\n"
            "  Without it screen [4]'s identifier/date branch fails silently\n"
            "  (candidate pool comes out 534 instead of 529).\n"
            "  Build it: python3 index/fundamentals/mdrm_names.py")
    names = _json.load(open(MDRM_NAMES))
    BAD = ("DATE", "NUMBER", "NBR ", "CERTIFICATE", "RSSD", "IDENT", "CODE", "FLAG")
    out = []
    for c in stable:
        desc = names.get(c, "").upper()
        v = d[c]
        if any(b in desc for b in BAD) and "NBR OF FT" not in desc:
            out.append((c, f"description contains an identifier/date: {desc[:40]}"))
        elif v.notna().sum() and (v.fillna(0) == 0).mean() > 0.995:
            out.append((c, f"{(v.fillna(0)==0).mean()*100:.1f}% zero"))
        elif v.nunique(dropna=True) < 20:
            out.append((c, f"only {v.nunique(dropna=True)} distinct values"))
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
    log(f"    screening window: prediction time <= {SCREEN_UNTIL} "
        f"({len(s):,} rows, {int(s.y.sum())} positives)")
    y = s.y.values
    pos, neg = s.loc[y == 1, stable].notna().mean(), s.loc[y == 0, stable].notna().mean()
    label_gap = (neg - pos).abs()
    ok_label = (label_gap <= MAX_LABEL_GAP) & (pos > MIN_SIDE_COV) & (neg > MIN_SIDE_COV)
    log(f"[2] missingness vs label: {int(ok_label.sum())}/{len(stable)} pass")

    ta_rank = s.groupby("quarter")[TA].rank(pct=True, na_option="top")
    big_m, rest_m = ta_rank >= 1 - BIG_FRAC, ta_rank < 1 - BIG_FRAC
    big = s.loc[big_m, stable].notna().mean()
    rest = s.loc[rest_m, stable].notna().mean()
    size_gap = (rest - big).abs()
    ok_size = (size_gap <= MAX_SIZE_GAP) & (big > MIN_SIDE_COV)
    log(f"[3] missingness vs size: {int(ok_size.sum())}/{len(stable)} pass "
        f"(top {BIG_FRAC:.0%} vs the rest, gap <={MAX_SIZE_GAP:.0%}, "
        f"large-bank side >{MIN_SIDE_COV:.0%})")
    log(f"    caught only by the size axis: {int((ok_label & ~ok_size).sum())}"
        "  <- the label axis cannot see this class")

    deg = degenerate(stable, d)
    ok_type = pd.Series({c: c not in deg for c in stable})
    log(f"[4] dropped identifiers / dates / near-constants: {len(deg)}")
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
    # The correlation matrix is unsupervised, but keeping it inside the same
    # window as every other screening step removes the question entirely.
    w = d[d.pred_q <= SCREEN_UNTIL]
    s = w.sample(min(60000, len(w)), random_state=0)
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
    log(f"[5] correlation pruning |rho|>={thresh}: {len(feats)} -> {len(reps)} "
        "(merged, not discarded)")
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
    log(f"\n{len(feats)} candidate features -> {OUT}/panel.parquet")
    top = sorted(feats, key=lambda c: -auc.get(c, 0.5))[:12]
    log("highest univariate AUC, top 12:")
    for c in top:
        log(f"  {auc[c]:.3f}  {c}")


if __name__ == "__main__":
    main()
