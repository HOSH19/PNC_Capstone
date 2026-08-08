"""Score one quarter from the frozen model and write bank_index_score.

Reads raw fields out of index_model_input, applies the frozen transform, refits
the GP on the 5,000 frozen rows, and writes scores. It never touches the panel,
extract.py or models.py's fold machinery — the two committed artefacts plus one
quarter of inputs are the whole dependency.

Refitting each run is not retraining. The rows are fixed, the seed is fixed, and
the kernel is fixed, so the fit is a replay: a bank's score moves only when the
bank's filing moves. A GP cannot be serialised compactly (the fitted object
carries its whole training set and an n x n kernel matrix, ~194 MB, and breaks
across scikit-learn versions), so replaying the fit — about 15 seconds — is the
cheap way to hold the model constant.

The transform is read from frozen_params.json rather than reimplemented. It has
to match features.py exactly, and the float32 cast is the part that silently
would not: the GP was fitted on float32 inputs, and feeding it float64 moves
every prediction.

    python -m index.fundamentals.score_quarter                  # latest quarter
    python -m index.fundamentals.score_quarter --quarter 2025-03-31
    python -m index.fundamentals.score_quarter --all --dry-run  # every quarter, no writes
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

FROZEN = pathlib.Path(os.environ.get("FROZEN_PARAMS", HERE / "frozen_params.json"))
SAMPLE = pathlib.Path(os.environ.get("TRAIN_SAMPLE", HERE / "train_sample.parquet"))
BANKS = ROOT / "db" / "seed" / "banks.csv"
JOB = "index_score_quarter"


def bank_ids() -> dict:
    """rssd_id -> bank_id, for the tracked banks only.

    012 carries bank_id as a convenience and indexes on it, and the dashboard
    joins by it. Only the seeded banks have one; every other filer keeps NULL,
    which is correct rather than missing — there is no bank_id to give them.
    """
    if not BANKS.exists():
        log(f"WARN {BANKS} not found — bank_id will be NULL for every row")
        return {}
    b = pd.read_csv(BANKS, usecols=["bank_id", "rssd_id"]).dropna()
    return {int(r.rssd_id): r.bank_id for r in b.itertuples()}


def log(m):
    print(m, flush=True)


def load_frozen():
    params = json.load(open(FROZEN))
    sample = pd.read_parquet(SAMPLE)
    missing = [f for f in params["features"] if f not in sample.columns]
    if missing:
        raise SystemExit(f"{SAMPLE.name} is missing features: {missing}")
    return params, sample


def build_model(params, sample):
    """Refit the GP on the frozen rows. Same rows, same seed, same kernel."""
    from sklearn.gaussian_process import GaussianProcessClassifier as GPC
    from sklearn.gaussian_process.kernels import ConstantKernel as C
    from sklearn.gaussian_process.kernels import Matern
    from sklearn.preprocessing import StandardScaler

    dim, feats = params["dim"], params["features"]
    X = sample[feats].values
    scaler = StandardScaler().fit(X)
    gp = GPC(kernel=C(10.0) * Matern(np.ones(dim) * np.sqrt(dim) * 1.5, nu=1.5),
             optimizer=None, random_state=params["seed"]).fit(
                 scaler.transform(X), sample.y.values)
    return scaler, gp


def transform(raw: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Raw filed values -> model features, per frozen_params["transform"].

    RC_2170 (total assets) is a raw field but not a feature: every MDRM item is
    divided by it and log_assets is derived from it. A zero denominator becomes
    missing rather than infinity — a filer reporting zero assets is a filing
    error, not a bank of infinite ratios.
    """
    feats = params["features"]
    ta = raw["rc_2170"].replace(0, np.nan)
    out = pd.DataFrame(index=raw.index)
    for f in feats:
        if f == "log_assets":
            out[f] = np.log1p(ta).astype("float32")
        else:
            col = f.lower()
            src = raw[col] if col in raw.columns else np.nan
            out[f] = (src / ta).astype("float32")
    # Frozen medians, not this quarter's — a quarter's own medians would make
    # every quarter's imputation a different model.
    return out.fillna(pd.Series(params["medians"]))


def score_rows(raw: pd.DataFrame, params: dict, scaler, gp) -> pd.DataFrame:
    from sklearn.linear_model import LogisticRegression

    feats = params["features"]
    Z = scaler.transform(transform(raw, params)[feats].values)
    p, lv, lm = [], [], []
    for i in range(0, len(Z), 5000):            # the full kernel matrix OOMs
        b = Z[i:i + 5000]
        p.append(gp.predict_proba(b)[:, 1])
        mean, var = gp.base_estimator_.latent_mean_and_variance(b)
        lm.append(mean)
        lv.append(var)
    p, lm, lv = np.concatenate(p), np.concatenate(lm), np.concatenate(lv)

    calib = LogisticRegression()
    calib.coef_ = np.array([[params["platt"]["coef"]]])
    calib.intercept_ = np.array([params["platt"]["intercept"]])
    calib.classes_ = np.array([0, 1])
    anchors = params["anchors"]

    pc = SC.calibrate(calib, p)
    sc = SC.prob_to_score(pc, anchors)
    # final_model.py derives latent_mean from the probability rather than taking
    # the GP's own latent mean, and score.py's intervals are calibrated against
    # that convention. Same derivation here.
    lm_from_p = np.log(np.clip(p, 1e-9, 1 - 1e-9) / (1 - np.clip(p, 1e-9, 1 - 1e-9)))
    iv = SC.score_intervals(lm_from_p, lv, calib, anchors)

    out = raw[["rssd_id", "report_date", "fdic_cert_number"]].copy()
    ids = bank_ids()
    out["bank_id"] = [ids.get(int(r)) if pd.notna(r) else None for r in out.rssd_id]
    out["distress_prob"] = pc
    out["score"] = sc
    out["band"] = SC.band_of(sc)
    for k, v in iv.items():
        out[k] = v
    out["latent_mean"] = lm_from_p
    out["latent_var"] = lv
    out["n_missing_features"] = imputed_count(raw, params)
    out["model_version"] = params["model_version"]
    return out


def imputed_count(raw: pd.DataFrame, params: dict) -> np.ndarray:
    """How many of the 50 inputs this row had to have filled in.

    012 keeps this so a score built on mostly-imputed inputs can be presented as
    low-confidence rather than shown as fact. Counted before imputation, on the
    ratio, so a row whose denominator is missing counts every feature as missing
    — which is exactly what happens to it.
    """
    feats = [f for f in params["features"] if f != "log_assets"]
    ta = raw["rc_2170"].replace(0, np.nan)
    present = {f: raw[f.lower()] for f in feats if f.lower() in raw.columns}
    absent = len(feats) - len(present)           # field not collected at all
    # An empty dict gives a 0-row frame, whose per-row sum is shape (0,) and
    # will not broadcast against the input. That means every field is missing,
    # so say so directly instead of raising a shape error three lines later.
    per_row = (pd.DataFrame(present).isna().sum(axis=1).values + absent
               if present else np.full(len(raw), absent))
    return np.where(ta.isna(), len(params["features"]), per_row).astype("int16")


def read_inputs(conn, quarter: str | None, every: bool) -> pd.DataFrame:
    cols = "rssd_id, report_date, fdic_cert_number, " + ", ".join(
        f.lower() for f in json.load(open(FROZEN))["raw_fields"])
    if every:
        sql, args = f"SELECT {cols} FROM index_model_input ORDER BY report_date", ()
    elif quarter:
        sql, args = f"SELECT {cols} FROM index_model_input WHERE report_date = %s", (quarter,)
    else:
        sql, args = (f"SELECT {cols} FROM index_model_input WHERE report_date = "
                     "(SELECT max(report_date) FROM index_model_input)"), ()
    with conn.cursor() as cur:
        cur.execute(sql, args)
        rows = cur.fetchall()
    d = pd.DataFrame(rows)
    # psycopg returns numeric as decimal.Decimal, which pandas keeps as object
    # dtype. numpy ufuncs have no Decimal loop, so np.log1p in transform() raises
    # TypeError and nothing is ever written. Reading the same values from parquet
    # gives float64, which is why this only breaks against a real database.
    for c in d.columns:
        if c not in ("rssd_id", "report_date", "fdic_cert_number"):
            d[c] = pd.to_numeric(d[c], errors="coerce")
    return d


def in_sample_cut(params) -> str:
    """The last report quarter this model trained on. Anything at or before it
    is memory, not prediction."""
    return params["fit_feature_quarters"][1]


def upsert_scores(conn, rows: pd.DataFrame) -> int:
    """Idempotent by (fdic_cert_number, quarter_end_date), matching 012's key.

    Rows without a certificate are dropped: 012 keys on it, so they have nowhere
    to go. They are counted and logged rather than silently discarded.
    """
    cols = ["fdic_cert_number", "quarter_end_date", "bank_id",
            "distress_prob", "score", "band",
            "score_lo_80", "score_hi_80", "score_lo_95", "score_hi_95",
            "latent_mean", "latent_var", "n_missing_features", "model_version"]
    r = rows.rename(columns={"report_date": "quarter_end_date"})
    r = r[r.fdic_cert_number.notna()][cols].astype(object).where(
        pd.notna(r[r.fdic_cert_number.notna()][cols]), None)
    sql = (f"INSERT INTO bank_index_score ({','.join(cols)}) "
           f"VALUES ({','.join(['%s'] * len(cols))}) "
           f"ON CONFLICT (fdic_cert_number, quarter_end_date) DO UPDATE SET "
           + ",".join(f"{c}=EXCLUDED.{c}" for c in cols[2:])
           + ", computed_at=now()")
    with conn.cursor() as cur:
        cur.executemany(sql, [tuple(x) for x in r.itertuples(index=False)])
    return len(r)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quarter", help="report_date to score, YYYY-MM-DD")
    ap.add_argument("--all", action="store_true", help="every quarter in the input table")
    ap.add_argument("--dry-run", action="store_true", help="score and report, write nothing")
    ap.add_argument("--from-parquet", help="read inputs from a parquet instead of the DB")
    ap.add_argument("--force-in-sample", action="store_true",
                    help="allow writing quarters the model trained on (see the guard)")
    args = ap.parse_args()
    if args.all and args.quarter:
        raise SystemExit("--all and --quarter are mutually exclusive")
    if not args.from_parquet and not os.environ.get("SUPABASE_DB_URL"):
        raise SystemExit(
            "SUPABASE_DB_URL is not set. Reading inputs needs the database even "
            "for --dry-run; pass --from-parquet to score a file instead.")

    params, sample = load_frozen()
    log(f"model {params['model_version']} | {params['dim']} features "
        f"| frozen {params['frozen_as_of']} | {len(sample):,} training rows")

    started, ok = time.monotonic(), False
    conn = None
    try:
        if args.from_parquet:
            raw = pd.read_parquet(args.from_parquet)
            if args.quarter:
                raw = raw[raw.report_date.astype(str) == args.quarter]
        else:
            conn = db.connect()
            raw = read_inputs(conn, args.quarter, args.all)
        if raw.empty:
            log("no input rows — nothing to score")
            return 1
        log(f"scoring {len(raw):,} rows over "
            f"{raw.report_date.astype(str).nunique()} quarter(s)")

        t0 = time.monotonic()
        scaler, gp = build_model(params, sample)
        log(f"GP refitted in {time.monotonic() - t0:.0f}s")
        out = score_rows(raw, params, scaler, gp)

        bands = dict(out.band.value_counts())
        log(f"  bands  {bands}")
        log(f"  scores {out.score.min():.1f} ~ {out.score.max():.1f} "
            f"(median {out.score.median():.1f})")
        log(f"  mean 80% interval width "
            f"{(out.score_hi_80 - out.score_lo_80).mean():.1f} points")
        n_id = int(out.bank_id.notna().sum())
        log(f"  {n_id:,} rows carry a bank_id (the tracked banks); the rest are NULL")
        no_cert = int(out.fdic_cert_number.isna().sum())
        if no_cert:
            log(f"  {no_cert:,} rows have no certificate and cannot be written to 012")

        if args.dry_run:
            log("dry run — nothing written")
            ok = True
            return 0

        # The production model trained through fit_feature_quarters[1], so its
        # scores for those quarters are memory: a bank that did trigger an event
        # ranks highly because the fit knew it did. bank_index_score's history
        # comes from the walk-forward instead (backfill_index_scores.py), and an
        # unguarded --all here would overwrite those honest rows with in-sample
        # ones under the production model_version — silently, since ON CONFLICT
        # DO UPDATE does not look at what it is replacing.
        cut = in_sample_cut(params)
        stale = out.report_date.astype(str).str.replace("-", "") <= cut
        if stale.any() and not args.force_in_sample:
            raise SystemExit(
                f"{int(stale.sum()):,} of {len(out):,} rows are report quarters "
                f"<= {cut}, which this model trained on. Writing them would "
                "replace the walk-forward history with in-sample scores. Use "
                "backfill_index_scores.py for history, or --force-in-sample if "
                "you are deliberately rebuilding.")
        if conn is None:
            conn = db.connect()
        n = upsert_scores(conn, out)
        conn.commit()
        log(f"wrote {n:,} rows to bank_index_score")
        ok = True
    finally:
        if conn:
            try:
                conn.rollback()
                db.write_heartbeat(conn, JOB, len(sample), 0 if args.dry_run else 1,
                                   time.monotonic() - started, ok)
            except Exception as e:
                log(f"WARN heartbeat not written: {type(e).__name__}: {e}")
            finally:
                conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
