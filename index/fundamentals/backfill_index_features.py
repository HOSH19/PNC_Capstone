"""Build the per-feature detail behind each score, and optionally load it.

`bank_index_score` says a bank scored 99.1 and sits in the sound band.
`bank_index_feature` is what that rests on: one row per bank-quarter-feature,
carrying the value the model actually saw.

**The values are the model's, not fact_call_report's.** 012 is explicit about
this — "values are the ones the MODEL saw (post winsorize/impute), so they can
differ slightly from raw fact_call_report; that is intentional". So a row here
is `raw_field / total_assets`, the same transform score_quarter.py applies,
rather than a conventional regulatory ratio. `RCRI_P742` is common equity tier 1
over total assets, which is not the tier 1 capital ratio anyone quotes (that one
divides by risk-weighted assets); showing it under that name would be wrong.
`display_name` comes from FEATURES.md and says what each field actually is.

`log_assets` is the one row that is not a ratio: it is log1p(total assets), the
size term, and reads around 20 rather than around 0.

`threshold_text` and `status` stay NULL. A Gaussian Process has no per-feature
threshold — its evidence is overall similarity to historically distressed banks,
so there is no "must be >= 9.0" to render. 012 anticipates this: consumers must
treat any feature row as optional and render only what is present.

Two sources, because the model's inputs live in two places:

    2019Q4-2025Q1   the panel models.py built, which holds the transformed
                    values directly
    2025Q2 onward   index_model_input, transformed here with the frozen recipe

    python index/fundamentals/backfill_index_features.py          # write the file
    python index/fundamentals/backfill_index_features.py --load   # and load it
"""

import argparse
import json
import os
import pathlib
import re
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

from pipeline import db
from score_quarter import bank_ids, transform

OUT = os.environ.get("MODEL_OUT", "/tmp/index_fundamentals")
FROZEN = HERE / "frozen_params.json"
FEATURES_MD = HERE / "FEATURES.md"
ARTEFACT = pathlib.Path(os.environ.get(
    "FEATURE_OUT", ROOT / "index" / "data" / "bank_index_feature_backfill.parquet"))
CALL_FACT = ROOT / "unified_ffiec_fdic_dataset" / "tables" / "fact_call_report.csv"
# Match bank_index_score exactly. A feature row is the detail behind a score, so
# one without a score is orphaned — and the score table starts at 2019Q4 because
# earlier quarters were only ever training data for the first fold.
SCORE_FROM = "20191201"
PANEL_UNTIL = "20250401"          # the panel covers up to here; the table after
JOB = "index_feature_backfill"

COLS = ["fdic_cert_number", "quarter_end_date", "feature_name", "display_name",
        "value", "is_imputed", "threshold_text", "status"]


def log(m):
    print(m, flush=True)


def display_names() -> dict:
    """Readable names, parsed from FEATURES.md's table rather than duplicated.

    Two files with the same 50 descriptions would drift; the markdown is already
    generated from mdrm_names.json and reviewed, so it is the source here too.
    The rank prefix keeps the dashboard's default ordering meaningful without
    needing another column.
    """
    out = {}
    for line in FEATURES_MD.read_text().split("\n"):
        m = re.match(r"^\| (\d+) \| `([\w]+)` \| (.*?) \| (.*?) \|$", line)
        if m:
            rank, code, _, reading = m.groups()
            out[code] = f"#{int(rank):02d} {reading}"
    if len(out) < 50:
        raise SystemExit(f"parsed only {len(out)} names from {FEATURES_MD.name}")
    return out


def cert_lookup() -> dict:
    c = pd.read_csv(CALL_FACT, usecols=["rssd_id", "report_date", "fdic_cert_number"])
    c = c.dropna(subset=["rssd_id", "fdic_cert_number"])
    c["q"] = pd.to_datetime(c.report_date).dt.strftime("%Y%m%d")
    return {(int(r.rssd_id), r.q): int(r.fdic_cert_number) for r in c.itertuples()}


def from_panel(params, seeds) -> pd.DataFrame:
    """History. The panel already holds the transformed values, NaN where the
    input was missing — which is exactly what is_imputed records."""
    path = f"{OUT}/panel.parquet"
    if not pathlib.Path(path).exists():
        raise SystemExit(f"missing {path} — run features.py first")
    feats = params["features"]
    d = pd.read_parquet(path, columns=["IDRSSD", "quarter"] + feats)
    d = d[d.IDRSSD.isin(seeds) & (d.quarter >= SCORE_FROM) & (d.quarter < PANEL_UNTIL)]
    log(f"panel    {len(d):,} bank-quarters  {d.quarter.min()[:6]}~{d.quarter.max()[:6]}")
    return d


def from_table(params, seeds, conn) -> pd.DataFrame:
    """Recent quarters, transformed here with the same recipe score_quarter uses
    — imported rather than reimplemented, so the two cannot drift."""
    cols = "rssd_id, report_date, " + ", ".join(
        f.lower() for f in params["raw_fields"])
    with conn.cursor() as cur:
        cur.execute(f"SELECT {cols} FROM index_model_input WHERE report_date >= %s",
                    (f"{PANEL_UNTIL[:4]}-{PANEL_UNTIL[4:6]}-{PANEL_UNTIL[6:]}",))
        raw = pd.DataFrame(cur.fetchall())
    if raw.empty:
        log("table    no rows at or after the panel's end")
        return pd.DataFrame()
    for c in raw.columns:
        if c not in ("rssd_id", "report_date"):
            raw[c] = pd.to_numeric(raw[c], errors="coerce")
    raw = raw[raw.rssd_id.isin(seeds)]
    # Before imputation, so NaN still marks what was missing. transform() fills
    # with the frozen medians, so the two are compared to recover is_imputed.
    pre = pd.DataFrame({f: (raw[f.lower()] / raw["rc_2170"].replace(0, np.nan))
                        if f != "log_assets" else np.log1p(raw["rc_2170"].replace(0, np.nan))
                        for f in params["features"]})
    post = transform(raw, params)
    post[pre.isna()] = np.nan          # re-open the gaps transform filled
    post["IDRSSD"] = raw.rssd_id.values
    post["quarter"] = pd.to_datetime(raw.report_date).dt.strftime("%Y%m%d").values
    log(f"table    {len(post):,} bank-quarters  "
        f"{post.quarter.min()[:6]}~{post.quarter.max()[:6]}")
    return post


def melt(d: pd.DataFrame, params, names, certs, ids) -> pd.DataFrame:
    feats = params["features"]
    m = d.melt(id_vars=["IDRSSD", "quarter"], value_vars=feats,
               var_name="feature_name", value_name="value")
    m["is_imputed"] = m.value.isna()
    # The model never sees NaN — it imputes with the frozen median, so that is
    # the value behind the score and the one to publish. is_imputed is what says
    # the number was filled in rather than filed.
    med = pd.Series(params["medians"])
    m["value"] = m.value.fillna(m.feature_name.map(med))
    m["display_name"] = m.feature_name.map(names)
    m["fdic_cert_number"] = [certs.get((int(r), q))
                             for r, q in zip(m.IDRSSD, m.quarter)]
    m["quarter_end_date"] = pd.to_datetime(m.quarter, format="%Y%m%d").dt.date
    m["threshold_text"] = None
    m["status"] = None
    dropped = int(m.fdic_cert_number.isna().sum())
    if dropped:
        log(f"  {dropped:,} rows have no certificate — dropped")
        m = m[m.fdic_cert_number.notna()]
    m["value"] = m.value.astype("float32")
    return m[COLS].sort_values(["quarter_end_date", "fdic_cert_number", "feature_name"])


def load(conn, d: pd.DataFrame, chunk=20000) -> int:
    r = d.astype(object).where(pd.notna(d), None)
    sql = (f"INSERT INTO bank_index_feature ({','.join(COLS)}) "
           f"VALUES ({','.join(['%s'] * len(COLS))}) "
           f"ON CONFLICT (fdic_cert_number, quarter_end_date, feature_name) "
           f"DO UPDATE SET " + ",".join(f"{c}=EXCLUDED.{c}" for c in COLS[3:]))
    rows = [tuple(x) for x in r.itertuples(index=False)]
    with conn.cursor() as cur:
        for i in range(0, len(rows), chunk):
            cur.executemany(sql, rows[i:i + chunk])
            conn.commit()
            log(f"  {min(i + chunk, len(rows)):,}/{len(rows):,}")
    return len(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--load", action="store_true", help="also write bank_index_feature")
    ap.add_argument("--out", help="where to write the parquet")
    args = ap.parse_args()
    out_path = pathlib.Path(args.out) if args.out else ARTEFACT

    params = json.load(open(FROZEN))
    names, certs, ids = display_names(), cert_lookup(), bank_ids()
    seeds = set(ids)
    log(f"{len(params['features'])} features x {len(seeds)} tracked banks")

    conn = db.connect() if os.environ.get("SUPABASE_DB_URL") else None
    try:
        parts = [from_panel(params, seeds)]
        if conn is not None:
            t = from_table(params, seeds, conn)
            if not t.empty:
                parts.append(t)
        else:
            log("table    skipped, SUPABASE_DB_URL is not set")
        d = melt(pd.concat(parts, ignore_index=True), params, names, certs, ids)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        d.to_parquet(out_path, index=False)
        log(f"\n{len(d):,} rows -> {out_path} "
            f"({out_path.stat().st_size / 1e6:.1f} MB)")
        log(f"  {d.fdic_cert_number.nunique()} banks x "
            f"{d.quarter_end_date.nunique()} quarters x "
            f"{d.feature_name.nunique()} features")
        log(f"  imputed {d.is_imputed.mean():.2%} of values")
        if not args.load:
            log("\nnot loaded — pass --load to write bank_index_feature")
            return 0

        started, ok = time.monotonic(), False
        n = load(conn, d)
        log(f"loaded {n:,} rows into bank_index_feature")
        ok = True
    finally:
        if conn:
            try:
                conn.rollback()
                if args.load:
                    db.write_heartbeat(conn, JOB, len(seeds), 1 if ok else 0,
                                       time.monotonic() - started, ok)
            except Exception as e:
                log(f"WARN heartbeat not written: {type(e).__name__}: {e}")
            finally:
                conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
