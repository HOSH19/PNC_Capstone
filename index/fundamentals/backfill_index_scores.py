"""Build the one-time history for bank_index_score, and optionally load it.

The dashboard shows a time series, and the quarterly scorer only ever writes the
current quarter. This produces everything before that, once.

**The production model cannot supply the history.** It trains on report quarters
through 2024Q1, so its scores for those quarters are memory rather than
prediction — a bank that did go on to trigger an event is ranked highly because
the model was fitted knowing it did. Publishing that as a historical track record
would make the axis look better than it is, systematically and invisibly.

So the history comes from the walk-forward instead. Each fold's model was trained
only on data preceding its own test window, so its predictions for that window
are what was genuinely knowable at the time:

    2017Q1-2019Q3   no rows. Those quarters were only ever training data for the
                    first fold, so nothing honest exists for them.
    2019Q4-2025Q1   out-of-fold predictions, model_version gp50_oos_v1
    2025Q2 onward   the production model, model_version gp50_prod_v1

`model_version` is what tells the two apart; 012 already carries it, so no schema
change is needed and a consumer can always see which model produced a row.

Both are mapped through the same Platt calibration and score anchors. Pooled over
every filer the two agree closely — median difference -0.06 points on a 0-100
scale — but that average hides the disagreement on the largest banks, which is
what PROD_FROM below is about.

    python index/fundamentals/backfill_index_scores.py            # write the file
    python index/fundamentals/backfill_index_scores.py --load     # and load it
"""

import argparse
import json
import os
import pathlib
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parents[2]
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import score as SC
from pipeline import db
from score_quarter import bank_ids

OUT = os.environ.get("MODEL_OUT", "/tmp/index_fundamentals")
FROZEN = HERE / "frozen_params.json"
ARTEFACT = pathlib.Path(os.environ.get(
    "BACKFILL_OUT", ROOT / "index" / "data" / "bank_index_score_backfill.parquet"))
CALL_FACT = ROOT / "unified_ffiec_fdic_dataset" / "tables" / "fact_call_report.csv"
# Where the published series switches from the fold models to the production
# model. Not the earliest quarter the production model *could* honestly score
# (that is 2024Q2, the first it did not train on) but the end of the walk-forward,
# so the whole history comes from one consistent source and the handover lands at
# the present rather than inside the chart.
#
# The two models disagree by more than rounding on the largest banks, which are
# barely represented in a 5,000-row training sample. Over 2019Q4-2025Q1 the
# production model puts 1-6 of the 104 tracked banks in the distress band each
# quarter where the folds put 0-2 — consistently, in every quarter, both models
# scoring the same filings. Splicing at 2024Q2 therefore drew a step in the middle
# of the series: 0 distressed banks in 2024Q1, 6 in 2024Q2, none of which was a
# change in any bank. Splicing at the end leaves one handover, at the edge, where
# "this is what the current model says" is the natural reading.
PROD_FROM = "20250401"
OOS_MODEL = "gp50_oos_v1"
JOB = "index_backfill"

COLS = ["fdic_cert_number", "quarter_end_date", "bank_id", "distress_prob", "score",
        "band", "score_lo_80", "score_hi_80", "score_lo_95", "score_hi_95",
        "latent_mean", "latent_var", "n_missing_features", "model_version"]


def log(m):
    print(m, flush=True)


def calibrator(params):
    from sklearn.linear_model import LogisticRegression
    c = LogisticRegression()
    c.coef_ = np.array([[params["platt"]["coef"]]])
    c.intercept_ = np.array([params["platt"]["intercept"]])
    c.classes_ = np.array([0, 1])
    return c


def publish(prob_raw, latent_var, params):
    """Raw GP probability -> the published columns, one mapping for both halves.

    latent_mean is derived from the probability rather than taken from the GP,
    matching final_model.py — score.py's intervals are written against that
    convention and would not line up otherwise.
    """
    calib, anchors = calibrator(params), params["anchors"]
    pc = SC.calibrate(calib, prob_raw)
    sc = SC.prob_to_score(pc, anchors)
    clipped = np.clip(prob_raw, 1e-9, 1 - 1e-9)
    lm = np.log(clipped / (1 - clipped))
    out = {"distress_prob": pc, "score": sc, "band": SC.band_of(sc),
           "latent_mean": lm, "latent_var": latent_var}
    out.update(SC.score_intervals(lm, latent_var, calib, anchors))
    return out


def cert_lookup() -> dict:
    c = pd.read_csv(CALL_FACT, usecols=["rssd_id", "report_date", "fdic_cert_number"])
    c = c.dropna(subset=["rssd_id", "fdic_cert_number"])
    c["q"] = pd.to_datetime(c.report_date).dt.strftime("%Y%m%d")
    return {(int(r.rssd_id), r.q): int(r.fdic_cert_number) for r in c.itertuples()}


def assemble(params) -> pd.DataFrame:
    for name, why in (("oos_predictions.parquet", "models.py"),
                      ("scores.parquet", "final_model.py")):
        if not pathlib.Path(f"{OUT}/{name}").exists():
            raise SystemExit(f"missing {OUT}/{name} — run {why} first")
    oos = pd.read_parquet(f"{OUT}/oos_predictions.parquet")
    prod = pd.read_parquet(f"{OUT}/scores.parquet")
    dim = params["dim"]
    pcol, lcol = f"gp{dim}@fixed", f"gp{dim}@fixed_latent_var"
    if lcol not in oos.columns:
        raise SystemExit(
            f"{OUT}/oos_predictions.parquet has no {lcol}. Re-run models.py — "
            "without it the historical rows have no confidence intervals.")

    hist = oos[oos.quarter < PROD_FROM].copy()
    cur = prod[prod.quarter >= PROD_FROM].copy()

    def span(f):
        # .min() on an empty string Series returns NaN, and slicing a float
        # raises three lines into a log statement. An empty half is a real
        # situation — a stale scores.parquet, or running between models.py and
        # final_model.py — and should name itself.
        return f"{f.quarter.min()[:6]}~{f.quarter.max()[:6]}" if len(f) else "empty"

    log(f"history  {len(hist):,} rows  {span(hist)}  ({OOS_MODEL})")
    log(f"current  {len(cur):,} rows  {span(cur)}  ({params['model_version']})")
    # An empty current half is the normal state: PROD_FROM sits at the end of the
    # walk-forward, so the production model has nothing to contribute until the
    # first quarterly run. An empty history is not — it means the backfill would
    # publish nothing.
    if hist.empty:
        raise SystemExit(
            f"no history before {PROD_FROM} in {OUT}/oos_predictions.parquet. "
            "Check that it and scores.parquet are from the same run of models.py.")

    frames = []
    for src, prob, lv, version in (
            (hist, hist[pcol].values, hist[lcol].values, OOS_MODEL),
            (cur, cur.prob_raw.values, cur.latent_var.values, params["model_version"])):
        if src.empty:
            continue
        f = pd.DataFrame({"rssd_id": src.IDRSSD.values, "quarter": src.quarter.values})
        for k, v in publish(prob, lv, params).items():
            f[k] = v
        f["model_version"] = version
        frames.append(f)
    d = pd.concat(frames, ignore_index=True)

    # n_missing_features belongs to the inputs, not the predictions, and neither
    # source carries it. The panel it was scored from is a /tmp artefact, so
    # rather than invent a number the column is left NULL and the quarterly
    # scorer fills it going forward. 012 allows NULL and the comment there is
    # about presentation, not correctness.
    d["n_missing_features"] = None

    certs, ids = cert_lookup(), bank_ids()
    d["fdic_cert_number"] = [certs.get((int(r), q))
                             for r, q in zip(d.rssd_id, d.quarter)]
    d["bank_id"] = [ids.get(int(r)) for r in d.rssd_id]
    d["quarter_end_date"] = pd.to_datetime(d.quarter, format="%Y%m%d").dt.date

    dropped = int(d.fdic_cert_number.isna().sum())
    if dropped:
        log(f"  {dropped:,} rows have no certificate and cannot key into 012 — dropped")
        d = d[d.fdic_cert_number.notna()]
    dupes = d.duplicated(subset=["fdic_cert_number", "quarter_end_date"]).sum()
    if dupes:
        raise SystemExit(f"{dupes} duplicate (cert, quarter) pairs — 012's key would reject")

    # float32 for the published numbers. End-to-end reproduction of this pipeline
    # agrees to ~1e-5 on a 0-100 score, so float64's extra digits are noise, and
    # the file is committed — halving it is worth more than precision nobody has.
    # 012's columns are numeric, so the width here only affects the artefact.
    for c in ("distress_prob", "score", "score_lo_80", "score_hi_80",
              "score_lo_95", "score_hi_95", "latent_mean", "latent_var"):
        d[c] = d[c].astype("float32")
    return d[COLS].sort_values(["quarter_end_date", "fdic_cert_number"])


def load(conn, d: pd.DataFrame) -> int:
    r = d.astype(object).where(pd.notna(d), None)
    sql = (f"INSERT INTO bank_index_score ({','.join(COLS)}) "
           f"VALUES ({','.join(['%s'] * len(COLS))}) "
           f"ON CONFLICT (fdic_cert_number, quarter_end_date) DO UPDATE SET "
           + ",".join(f"{c}=EXCLUDED.{c}" for c in COLS[2:]) + ", computed_at=now()")
    with conn.cursor() as cur:
        cur.executemany(sql, [tuple(x) for x in r.itertuples(index=False)])
    return len(r)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--load", action="store_true",
                    help="also write the rows into bank_index_score")
    ap.add_argument("--out", help="where to write the parquet")
    args = ap.parse_args()
    out_path = pathlib.Path(args.out) if args.out else ARTEFACT

    params = json.load(open(FROZEN))
    d = assemble(params)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    d.to_parquet(out_path, index=False)
    log(f"\n{len(d):,} rows -> {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
    log(f"  quarters {d.quarter_end_date.min()} ~ {d.quarter_end_date.max()} "
        f"({d.quarter_end_date.nunique()})")
    log(f"  banks    {d.fdic_cert_number.nunique():,} "
        f"({d.bank_id.notna().sum():,} rows carry a bank_id)")
    for v, g in d.groupby("model_version"):
        log(f"  {v:16} {len(g):>7,} rows  bands {dict(g.band.value_counts())}")

    if not args.load:
        log("\nnot loaded — pass --load to write bank_index_score")
        return 0
    if not os.environ.get("SUPABASE_DB_URL"):
        raise SystemExit("SUPABASE_DB_URL is not set")

    started, ok, conn = time.monotonic(), False, db.connect()
    try:
        n = load(conn, d)
        conn.commit()
        log(f"loaded {n:,} rows into bank_index_score")
        ok = True
    finally:
        try:
            conn.rollback()
            db.write_heartbeat(conn, JOB, len(d), len(d) if ok else 0,
                               time.monotonic() - started, ok)
        except Exception as e:
            log(f"WARN heartbeat not written: {type(e).__name__}: {e}")
        finally:
            conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
