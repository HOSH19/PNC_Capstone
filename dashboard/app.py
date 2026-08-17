"""Bank Stability Monitor — Streamlit dashboard (Phase 5).

Mirrors the first-screen concept in dashboard/concept/ (mentor discussion,
2026-07-12). Backed by mock data for now — real reads against the index and
score tables land once Phase 2 (scoring) and Phase 3 (index) exist; see
db/migrations/011_scoring_tables.sql for the current shared contract.

Fundamentals is organized as a CAMELS-style profile grouped into Capital,
Credit Quality, Liquidity, and Profitability (Management/Sensitivity metrics
that don't fit a quarterly ratio format — enforcement actions, unrealized
losses vs. threshold — surface as Key Alerts instead). It's the first panel
on the page and spans full width since it now carries alerts + a grouped
table + a per-metric history view. The composite Stable/Watch/Elevated Risk
status badge is driven solely by fundamentals x sentiment per the original
rubric.

Visual styling (cards, pills, gauge, segmented bar, keyword/category bars,
sparklines) is a custom CSS + Altair layer on top of Streamlit's default
widgets — see render_* functions below and the CSS block in main().
"""

import json
import os
import re
from datetime import date, timedelta
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import psycopg
import streamlit as st
from dotenv import load_dotenv
from psycopg.rows import dict_row
from sklearn.gaussian_process import GaussianProcessClassifier as _GPC
from sklearn.gaussian_process.kernels import ConstantKernel as _C
from sklearn.gaussian_process.kernels import Matern as _Matern
from sklearn.preprocessing import StandardScaler as _StandardScaler

st.set_page_config(page_title="Bank Stability Monitor", layout="wide")

load_dotenv()

# Column-for-column matches db/migrations/012_index_tables.sql
# (bank_index_score / bank_index_feature). Swapping the source is a change to
# _INDEX_SCORES/_INDEX_FEATURES only — load_fundamentals_mock() and every
# render_* function stay as-is.


@st.cache_data(ttl=3600)
def _load_data():
    """All 5 live/backfill reads below, cached per session (refreshed hourly).
    Streamlit reruns this whole module on every widget interaction, so
    without caching, switching banks/metrics/pages was re-running all 5 SQL
    queries plus the CSV backfill read on every single click."""
    # bank_index_score: live read (read-only, per docs/roles/ — no writes, no other
    # module's code imported, just the shared table). WHERE bank_id IS NOT NULL
    # scopes to the 105 tracked banks; live already carries the production
    # quarters (gp50_prod_v1, 2025Q2 on) on top of the gp50_oos_v1 backtest, which
    # the committed backfill parquet in index/data/ doesn't have. quarter_end_date
    # comes back as a native date; cast to string so _quarter_label()'s "-" split
    # keeps working unchanged.
    with psycopg.connect(os.environ["SUPABASE_DB_URL"], row_factory=dict_row) as _conn:
        with _conn.cursor() as _cur:
            _cur.execute(
                "SELECT fdic_cert_number, quarter_end_date, bank_id, distress_prob, "
                "score, band, score_lo_80, score_hi_80, score_lo_95, score_hi_95, "
                "latent_mean, latent_var, n_missing_features, model_version, "
                "config_version, computed_at "
                "FROM bank_index_score WHERE bank_id IS NOT NULL"
            )
            index_scores = pd.DataFrame(_cur.fetchall())

            # bank_index_feature: real, for load_feature_attribution() (the "what's
            # unusual about this bank" panel) — separate from _INDEX_FEATURES
            # (still mock), which the CAMELS-shaped panel keeps using untouched.
            # No bank_id column on this table (see 012_index_tables.sql), so the
            # tracked-bank scope is reapplied via fdic_cert_number against
            # bank_index_score — same set bank_id IS NOT NULL selected above.
            # Was unscoped: fact_call_report covers the full national filer
            # panel (167k+ bank-quarters per the fundamentals report), so an
            # unfiltered read here pulled far more into memory than the
            # dashboard ever uses, which is the likely cause of the Cloud OOM.
            _cur.execute(
                "SELECT fdic_cert_number, quarter_end_date, feature_name, "
                "display_name, value, is_imputed FROM bank_index_feature "
                "WHERE fdic_cert_number IN "
                "(SELECT fdic_cert_number FROM bank_index_score WHERE bank_id IS NOT NULL)"
            )
            model_features = pd.DataFrame(_cur.fetchall())

            # fact_call_report: real CAMELS-style ratios, replacing the mock CSV
            # the metric-breakdown table used to read. Only the 5 fields with a
            # defensible % definition are kept — liquidity_ratio's real scale
            # didn't match the old mock's threshold band (never reconciled, see
            # earlier note) and fee_income_ratio was never populated in either
            # source; both stay excluded rather than wired in unverified.
            # Scoped to tracked banks for the same reason as bank_index_feature
            # above — this table is the full national filer panel otherwise.
            _cur.execute(
                "SELECT fdic_cert_number, report_date, total_assets, "
                "tier1_capital_ratio, total_capital_ratio, npl_ratio, "
                "loan_loss_allowance_ratio, cre_loans FROM fact_call_report "
                "WHERE fdic_cert_number IN "
                "(SELECT fdic_cert_number FROM bank_index_score WHERE bank_id IS NOT NULL)"
            )
            call_report = pd.DataFrame(_cur.fetchall())

            # bank_sentiment_quarter: live read (db/migrations/016_sentiment_
            # aggregate.sql, same read-only contract as bank_index_score above).
            # Only 2025Q1 onward has real signal; earlier quarters read back as
            # n_scored=0 filler against the call-report calendar, so the date
            # filter both avoids that filler and avoids double-counting the
            # 2020-2024 window the CSV backfill below already covers.
            _cur.execute(
                "SELECT bank_id, quarter_end_date, n_scored, n_negative, n_positive, "
                "mean_p_negative, mean_p_positive FROM bank_sentiment_quarter "
                "WHERE quarter_end_date > '2024-12-31'"
            )
            sentiment_live = pd.DataFrame(_cur.fetchall())

            # raw_item x item_score: live scored+attributed items for RECENT
            # ITEMS (Jiwon, PR #15 comment 2026-08-15, step 3 — "live window
            # only", i.e. the DB's ongoing GDELT/EDGAR polling, not the
            # file-based backtest corpus). Item-level, so scoped to the 4 demo
            # banks rather than all 104 like the quarter-level tables above.
            _cur.execute(
                "SELECT ri.bank_id, ri.title, ri.url, ri.domain, ri.source, "
                "ri.published_at, s.label FROM raw_item ri "
                "JOIN item_score s ON s.raw_item_id = ri.id "
                "WHERE ri.attributed AND ri.bank_id = ANY(%(bank_ids)s) "
                "ORDER BY ri.published_at DESC",
                {"bank_ids": ["wfc", "wal", "pnc", "jpm"]},
            )
            recent_items = pd.DataFrame(_cur.fetchall())

    index_scores["quarter_end_date"] = index_scores["quarter_end_date"].astype(str)
    model_features["quarter_end_date"] = model_features["quarter_end_date"].astype(str)
    model_features["value"] = model_features["value"].astype(float)  # numeric column -> Decimal by default
    call_report["report_date"] = call_report["report_date"].astype(str)
    for _col in ("total_assets", "tier1_capital_ratio", "total_capital_ratio",
                 "npl_ratio", "loan_loss_allowance_ratio", "cre_loans"):
        call_report[_col] = call_report[_col].astype(float)
    sentiment_live["quarter_end_date"] = sentiment_live["quarter_end_date"].astype(str)
    for _col in ("n_scored", "n_negative", "n_positive"):
        sentiment_live[_col] = sentiment_live[_col].astype(int)
    for _col in ("mean_p_negative", "mean_p_positive"):
        sentiment_live[_col] = sentiment_live[_col].astype(float)

    # History: index/data/sentiment_quarter_backfill.csv (2020Q1-2024Q4, same
    # schema, committed 2026-08-15 per Jiwon's PR #15 comment). Union on
    # (bank_id, quarter_end_date) — the two windows don't overlap by construction
    # of the live query's date filter above.
    sentiment_hist = pd.read_csv(
        Path(__file__).parents[1] / "index" / "data" / "sentiment_quarter_backfill.csv",
        usecols=["bank_id", "quarter_end_date", "n_scored", "n_negative", "n_positive",
                 "mean_p_negative", "mean_p_positive"],
    )
    sentiment = pd.concat([sentiment_hist, sentiment_live], ignore_index=True)
    sentiment = sentiment[sentiment["n_scored"] > 0].sort_values(
        ["bank_id", "quarter_end_date"]
    )
    recent_items["published_at"] = pd.to_datetime(recent_items["published_at"], utc=True)

    return index_scores, model_features, call_report, sentiment, recent_items


_INDEX_SCORES, _MODEL_FEATURES, _CALL_REPORT, _SENTIMENT, _RECENT_ITEMS = _load_data()


@st.cache_resource
def _load_gp_model():
    """Refit of the frozen gp50_prod_v1 model (index/fundamentals/{train_sample.parquet,
    frozen_params.json}, copied from the scoring branch — same files
    freeze.py wrote and final_model.py fits from). optimizer=None means the
    kernel hyperparameters are fixed rather than learned, so this refit is
    deterministic given the same data and seed — validated by scoring real bank-
    quarters through it and matching bank_index_score's published distress_prob
    to 4 decimal places for every gp50_prod_v1 row checked. Only gp50_prod_v1 is
    reproduced this way; gp50_oos_v1 (backtest) rows come from separate per-fold
    models this refit does not represent, so load_feature_attribution() only
    runs for a bank's latest quarter when it's gp50_prod_v1.

    cache_resource (not cache_data) because the fitted GPC/scaler are unpicklable
    the way cache_data expects, and because this fit is O(n^3) in training rows —
    the single most expensive step on the page, and the main reason bank/metric/
    pagination switches felt slow before this was cached."""
    fundamentals_dir = Path(__file__).resolve().parent.parent / "index" / "fundamentals"
    frozen = json.loads((fundamentals_dir / "frozen_params.json").read_text())
    gp_features = frozen["features"]
    gp_dim = frozen["dim"]
    gp_seed = frozen["seed"]
    platt_coef = frozen["platt"]["coef"]
    platt_intercept = frozen["platt"]["intercept"]

    train_sample = pd.read_parquet(fundamentals_dir / "train_sample.parquet")
    gp_x = train_sample[gp_features].values.astype(np.float64)
    gp_y = train_sample["y"].values
    gp_scaler = _StandardScaler().fit(gp_x)
    gp_model = _GPC(
        kernel=_C(10.0) * _Matern(np.ones(gp_dim) * np.sqrt(gp_dim) * 1.5, nu=1.5),
        optimizer=None, random_state=gp_seed,
    ).fit(gp_scaler.transform(gp_x), gp_y)
    return gp_model, gp_scaler, gp_features, gp_dim, platt_coef, platt_intercept


_gp_model, _gp_scaler, _GP_FEATURES, _GP_DIM, _PLATT_COEF, _PLATT_INTERCEPT = _load_gp_model()


def _gp_distress_prob(rows: np.ndarray) -> np.ndarray:
    """rows: (n, 50) in _GP_FEATURES order, already the same raw/assets ratio
    transform bank_index_feature.value already carries. Returns calibrated
    distress_prob, matching bank_index_score's published column exactly."""
    raw = _gp_model.predict_proba(_gp_scaler.transform(rows))[:, 1]
    logit = _PLATT_COEF * raw + _PLATT_INTERCEPT
    return 1 / (1 + np.exp(-logit))

_BAND_LABEL = {"sound": "Sound", "neutral": "Neutral", "distress": "Distress signal"}
_STATUS_DISPLAY = {
    "within_range": "within range",
    "near_threshold": "near threshold",
    "breach": "outside range",
}


def _floor_status(value, floor: float) -> str | None:
    """>= floor is fine; within 15% of it above is a warning, below is a breach."""
    if value is None or pd.isna(value):
        return None
    margin = floor * 0.15
    if value < floor:
        return "breach"
    return "near_threshold" if value < floor + margin else "within_range"


def _ceiling_status(value, ceiling: float) -> str | None:
    """<= ceiling is fine; within 15% of it below is a warning, above is a breach."""
    if value is None or pd.isna(value):
        return None
    margin = ceiling * 0.15
    if value > ceiling:
        return "breach"
    return "near_threshold" if value > ceiling - margin else "within_range"


# feature_name -> (group, display name, threshold text, status function).
# Thresholds are only set where there's a real regulatory reference point:
# 8%/10% are the PCA "well capitalized" minimums for Tier 1 / Total risk-based
# capital (12 CFR 324.403); 300% is the interagency CRE-concentration
# guidance (2006) for CRE loans as a share of capital. npl_ratio and
# loan_loss_allowance_ratio have no equivalent hard regulatory line — shown
# as context only (threshold "—", status left unset) rather than inventing
# one. liquidity_ratio and fee_income_ratio are excluded — see the DB fetch
# comment above.
_CALL_REPORT_METRICS = {
    "tier1_capital_ratio": ("Capital", "Tier 1 capital ratio %", "≥ 8.0", lambda v: _floor_status(v, 8.0)),
    "total_capital_ratio": ("Capital", "Total capital ratio %", "≥ 10.0", lambda v: _floor_status(v, 10.0)),
    "npl_ratio": ("Credit Quality", "NPL ratio %", None, lambda v: None),
    "loan_loss_allowance_ratio": ("Credit Quality", "Loan-loss allowance / loans %", None, lambda v: None),
    "cre_to_capital": ("Credit Quality", "CRE loans / capital %", "≤ 300", lambda v: _ceiling_status(v, 300.0)),
}
# feature_name -> CAMELS group; not a schema column, so the mapping lives
# here alongside the other display concerns the dashboard owns.
_FEATURE_GROUP = {name: meta[0] for name, meta in _CALL_REPORT_METRICS.items()}


def _build_call_report_features() -> pd.DataFrame:
    """fact_call_report (wide, one row per bank-quarter) -> the same long
    shape bank_index_feature uses (one row per bank-quarter-feature), so
    load_fundamentals_mock's existing grouping/history logic needs no changes.
    cre_to_capital is derived: cre_loans / (total_assets * tier1/100) — there's
    no capital-in-dollars column, so capital is backed out from the ratio.
    Rows with a NULL underlying value are dropped rather than passed through
    (npl_ratio in particular is ~42% NULL DB-wide), same as bank_index_feature
    already does for genuinely missing inputs.
    """
    d = _CALL_REPORT.copy()
    cap_dollars = d["total_assets"] * d["tier1_capital_ratio"] / 100
    d["cre_to_capital"] = (100 * d["cre_loans"] / cap_dollars).round(1)
    for _col in ("tier1_capital_ratio", "total_capital_ratio", "npl_ratio", "loan_loss_allowance_ratio"):
        d[_col] = d[_col].round(2)

    long_rows = []
    for feature_name, (_, display, threshold_text, status_fn) in _CALL_REPORT_METRICS.items():
        sub = d[["fdic_cert_number", "report_date", feature_name]].dropna()
        for _, row in sub.iterrows():
            value = row[feature_name]
            long_rows.append({
                "fdic_cert_number": row["fdic_cert_number"],
                "quarter_end_date": row["report_date"],
                "feature_name": feature_name,
                "display_name": display,
                "value": value,
                "threshold_text": threshold_text,
                "status": status_fn(value),
            })
    return pd.DataFrame(long_rows)


_INDEX_FEATURES = _build_call_report_features()


def _quarter_label(quarter_end_date: str) -> str:
    year, month, _ = quarter_end_date.split("-")
    q = {"03": "Q1", "06": "Q2", "09": "Q3", "12": "Q4"}[month]
    return f"{q} '{year[2:]}"


def _format_threshold(text) -> str:
    # threshold_text is NULL for metrics with no established regulatory
    # threshold (npl_ratio, loan_loss_allowance_ratio — see
    # _CALL_REPORT_METRICS) — render code needs a plain string, not NaN.
    if pd.isna(text):
        return "—"
    return text.replace(">=", "≥").replace("<=", "≤").replace(" - ", " – ")


def load_fundamentals_mock(bank_id: str) -> dict | None:
    """Build a `fundamentals` dict (MOCK_BANKS shape) from the index mock CSVs.

    Returns None if `bank_id` isn't in the mock score sample, same as a
    hand-written entry omitting fundamentals.
    """
    bank_scores = _INDEX_SCORES[_INDEX_SCORES["bank_id"] == bank_id].sort_values(
        "quarter_end_date"
    )
    if bank_scores.empty:
        return None
    latest = bank_scores.iloc[-1]

    cert = bank_scores["fdic_cert_number"].iloc[0]
    bank_features = _INDEX_FEATURES[
        _INDEX_FEATURES["fdic_cert_number"] == cert
    ].sort_values("quarter_end_date")

    features = []
    for name, group_df in bank_features.groupby("feature_name", sort=False):
        # FEATURE_QUARTERS (the history-chart x-axis) is a fixed 5-label
        # window; fact_call_report goes back decades, so history has to be
        # windowed to match, the same way the score trend already is below.
        values = list(group_df.sort_values("quarter_end_date")["value"])[-len(FEATURE_QUARTERS):]
        row = group_df.iloc[-1]
        features.append(
            {
                "group": _FEATURE_GROUP.get(name, "Capital"),
                "name": row["display_name"],
                "prior": values[-2] if len(values) > 1 else values[-1],
                "latest": values[-1],
                "threshold": _format_threshold(row["threshold_text"]),
                # status is NULL for the same no-threshold metrics — falls
                # back to a fourth, unstyled state (render_fundamentals'
                # status_colors has no "unknown" entry, so it renders muted).
                "status": (
                    _STATUS_DISPLAY.get(row["status"], row["status"])
                    if pd.notna(row["status"])
                    else "unknown"
                ),
                "history": values,
            }
        )

    return {
        "score": round(latest["score"]),
        "label": _BAND_LABEL.get(latest["band"], latest["band"]),
        "trend": [round(s) for s in bank_scores["score"]],
        "quarters": [_quarter_label(d) for d in bank_scores["quarter_end_date"]],
        "features": features,
    }


# Negative cut is backtest-validated: evals/reports/2026-08-14_combined_ladder.md
# swept 0.05/0.10/0.20 and 0.10 won on PR-AUC (0.0810), matching the default
# already shipped in pipeline/combine_axes.py. The positive cut has no
# backtest behind it — Jiwon's PR #15 comment (2026-08-15) calls it
# "display only, no agreed rule yet... tunable later."
SENTIMENT_NEG_SHARE_CUT = 0.1
SENTIMENT_POS_SHARE_CUT = 0.2


def load_sentiment(bank_id: str) -> dict | None:
    """Build a `sentiment` dict (MOCK_BANKS shape) from _SENTIMENT (live
    bank_sentiment_quarter unioned with the 2020-2024 CSV backfill).

    Returns None if the bank has no scored quarters yet, same as a
    hand-written entry omitting sentiment — compute_status()'s
    fundamentals-only branch already handles that case.
    """
    rows = _SENTIMENT[_SENTIMENT["bank_id"] == bank_id]
    if rows.empty:
        return None
    latest = rows.iloc[-1]

    neg_share = latest["n_negative"] / latest["n_scored"]
    pos_share = latest["n_positive"] / latest["n_scored"]
    if neg_share >= SENTIMENT_NEG_SHARE_CUT:
        label = "Negative"
    elif pos_share > neg_share and pos_share >= SENTIMENT_POS_SHARE_CUT:
        label = "Positive"
    else:
        label = "Neutral"

    neutral_n = latest["n_scored"] - latest["n_negative"] - latest["n_positive"]
    return {
        # Net sentiment: mean p(negative) - mean p(positive) over scored
        # items, in [-1, 1]. Only used for display/trend charting —
        # compute_status() keys off `label`, never this value.
        "score": round(float(latest["mean_p_negative"] - latest["mean_p_positive"]), 2),
        "label": label,
        "n_items": int(latest["n_scored"]),
        "pct": {
            "negative": round(100 * latest["n_negative"] / latest["n_scored"]),
            "neutral": round(100 * neutral_n / latest["n_scored"]),
            "positive": round(100 * latest["n_positive"] / latest["n_scored"]),
        },
        "trend": [
            round(float(v), 2)
            for v in (rows["mean_p_negative"] - rows["mean_p_positive"])
        ],
    }


_FEED_DISPLAY = {"gdelt": "GDELT", "edgar": "EDGAR"}


def load_recent_items(bank_id: str) -> list[tuple]:
    """(title, label, source, days_ago, feed) tuples for render_recent_items,
    from _RECENT_ITEMS (raw_item x item_score, live-scored + attributed).

    Returns every attributed+scored item for the bank, not just a recent
    slice — the date filter needs the full history to filter over, and
    render_recent_items paginates the display 5-at-a-time on its own."""
    rows = _RECENT_ITEMS[_RECENT_ITEMS["bank_id"] == bank_id]
    today = date.today()
    out = []
    for _, r in rows.iterrows():
        days_ago = (today - r["published_at"].date()).days
        feed = _FEED_DISPLAY.get(r["source"], r["source"].upper())
        out.append((r["title"], r["label"], r["domain"] or r["source"], days_ago, feed))
    return out


def _format_feature_value(v) -> str:
    if v is None or pd.isna(v):
        return "—"
    return f"{v:,.2f}"


def _clean_feature_name(display_name: str) -> str:
    """bank_index_feature.display_name carries the model's global gain rank
    ('#06 ...') and, for one feature, the raw transform in backticks
    ('`log(1 + total assets)` — bank size'). Strip both so the table shows
    a business-readable label, not model internals."""
    name = re.sub(r"^#\d+\s*", "", display_name)
    name = re.sub(r"`[^`]*`\s*", "", name)
    name = name.strip(" —").strip()
    return name[:1].upper() + name[1:] if name else name


def load_feature_attribution(bank_id: str, top_n: int = 5) -> list[dict] | None:
    """Which of the 50 raw model features actually move this bank's latest
    distress_prob — real local attribution, via the refit gp50_prod_v1 above.
    For each feature, swap just that one to the peer average for the quarter
    (holding the other 49 at this bank's actual values) and measure how much
    the calibrated probability moves. Positive contribution = this bank's
    actual value is pushing distress_prob UP relative to a typical peer;
    negative = pulling it down (protective).

    Only covers a bank's latest quarter when it's scored by gp50_prod_v1 —
    gp50_oos_v1 (backtest) quarters come from separate per-fold models this
    refit doesn't represent, so those return None rather than a wrong number.
    """
    prod_scores = _INDEX_SCORES[
        (_INDEX_SCORES["bank_id"] == bank_id) & (_INDEX_SCORES["model_version"] == "gp50_prod_v1")
    ]
    if prod_scores.empty:
        return None
    latest = prod_scores.sort_values("quarter_end_date").iloc[-1]
    cert, quarter = latest["fdic_cert_number"], latest["quarter_end_date"]

    bank_row = _MODEL_FEATURES[
        (_MODEL_FEATURES["fdic_cert_number"] == cert) & (_MODEL_FEATURES["quarter_end_date"] == quarter)
    ].set_index("feature_name")
    if not all(f in bank_row.index for f in _GP_FEATURES):
        return None

    peer_mean = (
        _MODEL_FEATURES[_MODEL_FEATURES["quarter_end_date"] == quarter]
        .groupby("feature_name")["value"].mean()
    )

    x_actual = bank_row.loc[_GP_FEATURES, "value"].values.astype(np.float64)
    rows = np.tile(x_actual, (_GP_DIM + 1, 1))
    for i, name in enumerate(_GP_FEATURES):
        rows[i + 1, i] = peer_mean.get(name, x_actual[i])
    probs = _gp_distress_prob(rows)
    baseline_prob, perturbed_probs = probs[0], probs[1:]

    contributions = []
    for i, name in enumerate(_GP_FEATURES):
        row = bank_row.loc[name]
        contributions.append({
            "name": _clean_feature_name(row["display_name"]),
            "value": row["value"],
            "peer_mean": peer_mean.get(name),
            "contribution": baseline_prob - perturbed_probs[i],
            "is_imputed": bool(row["is_imputed"]),
        })

    contributions.sort(key=lambda d: abs(d["contribution"]), reverse=True)
    return contributions[:top_n]


FEATURE_QUARTERS = ["Q1 '25", "Q2 '25", "Q3 '25", "Q4 '25", "Q1 '26"]

MOCK_BANKS = {
    "Wells Fargo": {
        "name": "Wells Fargo & Company",
        "ticker": "WFC",
        "cert": "3511",
        "rssd": "451965",
        "summary": (
            "Fundamentals sit in the neutral band, but news flow is "
            "persistently negative around regulatory penalties — the "
            "pattern the early-warning view is built to surface."
        ),
        "sentiment": load_sentiment("wfc"),
        "keywords": {
            "negative": [
                ("regulatory fines", 0.92),
                ("consent order", 0.81),
                ("asset cap", 0.74),
                ("overcharging customers", 0.66),
                ("probe widens", 0.52),
            ],
            "positive": [
                ("dividend raise", 0.58),
                ("buyback program", 0.47),
                ("earnings beat", 0.41),
                ("cost discipline", 0.33),
            ],
        },
        "alerts": [],
        "fundamentals": load_fundamentals_mock("wfc"),
        "recent_items": load_recent_items("wfc"),
        "feature_drivers": load_feature_attribution("wfc"),
    },
    "Western Alliance": {
        "name": "Western Alliance Bancorporation",
        "ticker": "WAL",
        "cert": "57512",
        "rssd": "3138146",
        "summary": (
            "Fundamentals dipped below the 80 distress line while negative "
            "sentiment accelerates — both axes now point the same way, "
            "which is the strongest configuration of the warning signal."
        ),
        "sentiment": load_sentiment("wal"),
        "keywords": {
            "negative": [
                ("deposit outflows", 0.95),
                ("CRE exposure", 0.84),
                ("credit downgrade", 0.77),
                ("liquidity questions", 0.69),
                ("short interest", 0.55),
            ],
            "positive": [
                ("capital raise completed", 0.46),
                ("insured deposit mix", 0.39),
            ],
        },
        "alerts": [],
        "fundamentals": load_fundamentals_mock("wal"),
        "recent_items": load_recent_items("wal"),
        "feature_drivers": load_feature_attribution("wal"),
    },
    # Placeholder demo entries — no concept mockup / narrative summary
    # drafted for these, kept minimal on purpose. Sentiment is real
    # (load_sentiment), same as fundamentals; only keywords/alerts/
    # recent_items stay unwired (steps 4-5 of Jiwon's PR #15 wiring plan).
    "PNC Financial": {
        "name": "PNC Financial Services Group",
        "ticker": "PNC",
        "cert": "6384",
        "rssd": "817824",
        "summary": "Placeholder — no illustrative narrative drafted yet. Fundamentals and sentiment below are wired to live/historical data.",
        "sentiment": load_sentiment("pnc"),
        "keywords": None,
        "alerts": [],
        "fundamentals": load_fundamentals_mock("pnc"),
        "recent_items": load_recent_items("pnc"),
        "feature_drivers": load_feature_attribution("pnc"),
    },
    "JPMorgan Chase": {
        "name": "JPMorgan Chase & Co.",
        "ticker": "JPM",
        "cert": "628",
        "rssd": "852218",
        "summary": "Placeholder — no illustrative narrative drafted yet. Fundamentals and sentiment below are wired to live/historical data.",
        "sentiment": load_sentiment("jpm"),
        "keywords": None,
        "alerts": [],
        "fundamentals": load_fundamentals_mock("jpm"),
        "recent_items": load_recent_items("jpm"),
        "feature_drivers": load_feature_attribution("jpm"),
    },
}

# ---------------------------------------------------------------------------
# Visual design tokens — hand-matched to dashboard/concept/first-screen-*.png
# (dark card UI: near-black page, slightly-lighter bordered cards, a
# red/amber/green risk palette, and a teal accent for the selected bank pill).
# ---------------------------------------------------------------------------
BG_PAGE = "#0b0e14"
BG_CARD = "#131722"
BORDER = "#232937"
TEXT_PRIMARY = "#e8ebf1"
TEXT_MUTED = "#8891a3"
ACCENT_TEAL = "#2dd4bf"
RED = "#f2705c"
AMBER = "#e0a94a"
GREEN = "#4caf7d"

CAMELS_GROUPS = ["Capital", "Credit Quality", "Liquidity", "Profitability"]

# Four-level classification from README.md's original concept (mentor
# discussion, 2026-07-12): Stable / Watch / Elevated Risk / Imminent
# Disruption, driven by fundamentals x sentiment. The cutoffs below (the
# score<30 "critical" line, the -0.4 "strong negative" sentiment line) are a
# first judgment call, not a team-agreed threshold — nothing in the mock data
# currently reaches Imminent Disruption, so that tier is unverified against a
# demo case. Revisit once "confirmed distress events" (the other Imminent
# Disruption trigger in README.md) has a real source to check against.
IMMINENT_CRITICAL_SCORE = 30
STRONG_NEGATIVE_SENTIMENT = -0.4

STATUS_COLOR = {
    "Stable": GREEN,
    "Watch": AMBER,
    "Elevated Risk": RED,
    "Imminent Disruption": "#b91c3c",
}

FUNDAMENTALS_STATUS_COLOR = {
    "Sound": GREEN,
    "Neutral": AMBER,
    "Distress signal": RED,
}

SENTIMENT_LABEL_COLOR = {
    "Positive": GREEN,
    "Neutral": AMBER,
    "Negative": RED,
}


def compute_status(bank: dict) -> str:
    """Stable / Watch / Elevated Risk / Imminent Disruption, per README.md's
    ladder. Fundamentals-only when sentiment isn't scored for this bank
    (most MOCK_BANKS entries) — can't confirm the "both axes" tiers without
    a sentiment read, so those banks cap out at Elevated Risk."""
    fundamentals = bank.get("fundamentals")
    score = fundamentals["score"] if fundamentals else None
    sentiment = bank.get("sentiment")

    if sentiment is None:
        if score is None or score >= 90:
            return "Stable"
        if score <= 80:
            return "Elevated Risk"
        return "Watch"

    sent_negative = sentiment["label"] == "Negative"
    if score is not None and score < IMMINENT_CRITICAL_SCORE and sent_negative:
        return "Imminent Disruption"
    # Fundamentals floor: score <= 80 is Elevated Risk regardless of
    # sentiment. Without it, calm news rated a distressed bank BETTER than
    # no sentiment read at all (the None branch above already gives the
    # same score Elevated Risk) — and banks are often quiet in the news
    # right before trouble. Same rule as pipeline/combine_axes.py.
    if score is not None and score <= 80:
        return "Elevated Risk"
    if sent_negative or (score is not None and score < 90):
        return "Watch"
    return "Stable"


def pill_html(text: str, color: str, bg: str, border: str, size: str = "0.85rem") -> str:
    return (
        f'<span style="display:inline-flex;align-items:center;gap:6px;'
        f"padding:4px 12px;border-radius:999px;font-weight:600;font-size:{size};"
        f'color:{color};background:{bg};border:1px solid {border};white-space:nowrap;">'
        f"{text}</span>"
    )


def stat_tile_html(label: str, value: str, value_color: str = None) -> str:
    vc = value_color or TEXT_PRIMARY
    return (
        '<div style="display:flex;flex-direction:column;gap:3px;">'
        f'<span style="color:{TEXT_MUTED};font-size:0.7rem;text-transform:uppercase;letter-spacing:0.05em;">{label}</span>'
        f'<span style="color:{vc};font-size:1.2rem;font-weight:700;'
        f'font-family:\'SFMono-Regular\',Consolas,\'Liberation Mono\',Menlo,monospace;">{value}</span>'
        "</div>"
    )


def _keyword_bars_html(rows: list, color: str) -> str:
    parts = []
    for word, weight in rows:
        pct = int(round(weight * 100))
        parts.append(
            '<div class="kw-row">'
            f'<div class="kw-label">{word}</div>'
            f'<div class="kw-track"><div class="kw-fill" style="width:{pct}%;background:{color};"></div></div>'
            f'<div class="kw-value">{weight:.2f}</div>'
            "</div>"
        )
    return '<div class="kw-col-wrap">' + "".join(parts) + "</div>"


def _quarterly_score_chart(quarters: list, scores: list, color: str, width: int = None) -> alt.Chart:
    df = pd.DataFrame({"quarter": quarters, "score": scores})
    y_pad = max(2, (max(scores) - min(scores)) * 0.4)
    chart = (
        alt.Chart(df)
        .mark_line(point=alt.OverlayMarkDef(filled=True, size=90, color=color), color=color, strokeWidth=2.5)
        .encode(
            x=alt.X("quarter:N", sort=quarters, title=None, axis=alt.Axis(
                labelColor=TEXT_MUTED, labelFontSize=12, labelAngle=0, grid=False,
                domainColor=BORDER, tickColor=BORDER,
            )),
            y=alt.Y("score:Q", title=None, scale=alt.Scale(domain=[min(scores) - y_pad, max(scores) + y_pad]), axis=alt.Axis(
                labelColor=TEXT_MUTED, labelFontSize=12, gridColor=BORDER, domainColor=BORDER, tickColor=BORDER,
            )),
            tooltip=[alt.Tooltip("quarter:N", title="Quarter"), alt.Tooltip("score:Q", title="Score")],
        )
        .properties(height=180, background="transparent")
        .configure_view(strokeWidth=0)
    )
    if width:
        chart = chart.properties(width=width)
    return chart


def _feature_history_chart(quarters: list, values: list, color: str) -> alt.Chart:
    df = pd.DataFrame({"quarter": quarters, "value": values})
    y_pad = max(0.3, (max(values) - min(values)) * 0.3)
    return (
        alt.Chart(df)
        .mark_line(point=alt.OverlayMarkDef(filled=True, size=70, color=color), color=color, strokeWidth=2.5)
        .encode(
            x=alt.X("quarter:N", sort=quarters, title=None, axis=alt.Axis(
                labelColor=TEXT_MUTED, labelFontSize=11, labelAngle=0, grid=False,
                domainColor=BORDER, tickColor=BORDER,
            )),
            y=alt.Y("value:Q", title=None, scale=alt.Scale(domain=[min(values) - y_pad, max(values) + y_pad]), axis=alt.Axis(
                labelColor=TEXT_MUTED, labelFontSize=11, gridColor=BORDER, domainColor=BORDER, tickColor=BORDER,
            )),
            tooltip=[alt.Tooltip("quarter:N", title="Quarter"), alt.Tooltip("value:Q", title="Value")],
        )
        .properties(height=160, background="transparent")
        .configure_view(strokeWidth=0)
    )


def _qoq_delta_html(prior: float, latest: float, threshold: str) -> str:
    """Arrow shows the raw direction of the number; color shows whether that
    direction is good or bad given the feature's threshold (>=, <=, or a
    band, where improvement means moving toward the band's center)."""
    threshold = threshold.strip()
    if threshold.startswith("≥"):
        improved = latest >= prior
    elif threshold.startswith("≤"):
        improved = latest <= prior
    elif "–" in threshold:
        lo, hi = (float(x.strip()) for x in threshold.split("–"))
        center = (lo + hi) / 2
        improved = abs(latest - center) <= abs(prior - center)
    else:
        improved = latest >= prior

    pct = None if prior == 0 else (latest - prior) / abs(prior) * 100
    arrow = "▲" if latest > prior else ("▼" if latest < prior else "—")
    color = GREEN if improved else RED
    pct_str = f"{abs(pct):.1f}%" if pct is not None else "—"
    return f'<span style="color:{color};font-weight:600;">{arrow} {pct_str}</span>'


CSS = f"""
<style>
.stApp {{
    background-color: {BG_PAGE};
}}
[data-testid="stMetricValue"] {{
    font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
}}
[data-testid="stMetricLabel"] {{
    color: {TEXT_MUTED};
    font-size: 0.78rem;
}}
div[class*="st-key-card-"] {{
    background: {BG_CARD};
    border: 1px solid {BORDER} !important;
    border-radius: 14px;
    padding: 6px 8px 14px 8px;
    margin-bottom: 8px;
}}
.st-key-bank-picker [data-testid="stWidgetLabel"] p {{
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-size: 0.72rem;
    color: {TEXT_MUTED};
}}
.panel-title {{
    font-size: 0.8rem;
    font-weight: 700;
    letter-spacing: 0.03em;
    color: {TEXT_PRIMARY};
    margin-bottom: 2px;
}}
.panel-caption {{
    color: {TEXT_MUTED};
    font-size: 0.82rem;
    margin-bottom: 14px;
}}
.st-key-recent-in-keywords {{
    border-left: 1px solid {BORDER};
    padding-left: 20px;
}}
.seg-bar {{
    display: flex;
    width: 100%;
    height: 8px;
    border-radius: 999px;
    overflow: hidden;
    background: {BORDER};
    margin: 10px 0 8px 0;
}}
.seg-bar-legend {{
    display: flex;
    gap: 18px;
    font-size: 0.82rem;
    margin-bottom: 10px;
}}
.kw-row {{
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 9px;
}}
.kw-label {{
    flex: 0 0 46%;
    font-size: 0.85rem;
    color: {TEXT_PRIMARY};
}}
.kw-track {{
    flex: 1;
    height: 8px;
    border-radius: 999px;
    background: {BORDER};
    overflow: hidden;
}}
.kw-fill {{
    height: 100%;
    border-radius: 999px;
}}
.kw-value {{
    flex: 0 0 40px;
    text-align: right;
    font-size: 0.8rem;
    color: {TEXT_MUTED};
    font-family: monospace;
}}
.gauge-track {{
    position: relative;
    height: 10px;
    border-radius: 999px;
    background: linear-gradient(
        to right,
        {RED} 0%, {RED} 60%,
        {AMBER} 60%, {AMBER} 80%,
        {GREEN} 80%, {GREEN} 100%
    );
    opacity: 0.9;
    margin: 6px 0 4px 0;
}}
.gauge-marker {{
    position: absolute;
    top: -5px;
    width: 3px;
    height: 20px;
    background: #fff;
    border-radius: 2px;
    transform: translateX(-50%);
}}
.gauge-ticks {{
    display: flex;
    justify-content: space-between;
    font-size: 0.72rem;
    color: {TEXT_MUTED};
    margin-top: 4px;
}}
.fund-table {{
    width: 100%;
    border-collapse: collapse;
    margin-top: 10px;
    font-size: 0.8rem;
}}
.fund-table th {{
    text-align: left;
    color: {TEXT_MUTED};
    font-weight: 600;
    text-transform: uppercase;
    font-size: 0.64rem;
    letter-spacing: 0.04em;
    padding: 4px 8px;
    border-bottom: 1px solid {BORDER};
}}
.fund-table td:nth-child(2),
.fund-table td:nth-child(3),
.fund-table td:nth-child(4) {{
    text-align: right;
    font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
}}
.fund-table th:nth-child(2),
.fund-table th:nth-child(3),
.fund-table th:nth-child(4) {{
    text-align: right;
}}
.fund-table td {{
    padding: 5px 8px;
    border-bottom: 1px solid {BORDER};
    color: {TEXT_PRIMARY};
}}
.info-dot {{
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 14px;
    height: 14px;
    border-radius: 50%;
    border: 1px solid {TEXT_MUTED};
    color: {TEXT_MUTED};
    font-size: 0.62rem;
    cursor: help;
    margin-left: 4px;
    vertical-align: middle;
}}
/* <details>/<summary> version: tap/click to open, unlike .info-dot's
   hover-only native title tooltip which touch devices can't trigger. */
.info-pop {{
    position: relative;
    display: inline-block;
    vertical-align: middle;
    margin-left: 4px;
}}
.info-pop summary {{
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 14px;
    height: 14px;
    border-radius: 50%;
    border: 1px solid {TEXT_MUTED};
    color: {TEXT_MUTED};
    font-size: 0.62rem;
    cursor: pointer;
    list-style: none;
}}
.info-pop summary::-webkit-details-marker {{ display: none; }}
.info-pop[open] summary {{ color: {ACCENT_TEAL}; border-color: {ACCENT_TEAL}; }}
.info-pop .info-pop-body {{
    position: absolute;
    top: 20px;
    left: 0;
    width: 240px;
    background: {BG_CARD};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 8px 10px;
    font-size: 0.72rem;
    font-weight: 400;
    text-transform: none;
    letter-spacing: normal;
    color: {TEXT_PRIMARY};
    z-index: 20;
    box-shadow: 0 4px 12px rgba(0,0,0,0.4);
}}
.subtitle-list {{
    margin: 4px 0 14px 0;
    padding-left: 18px;
    color: {TEXT_MUTED};
    font-size: 0.82rem;
}}
.subtitle-list li {{
    margin-bottom: 4px;
}}
.alert-row {{
    display: flex;
    gap: 9px;
    align-items: flex-start;
    margin-bottom: 8px;
    padding: 9px 12px;
    border-radius: 8px;
}}
.alerts-scroll {{
    max-height: 110px;
    overflow-y: auto;
    padding-right: 4px;
}}
.recent-item {{
    border-left: 3px solid var(--item-color, {BORDER});
    padding: 2px 0 2px 12px;
    margin-bottom: 14px;
}}
.recent-title {{
    font-weight: 600;
    font-size: 0.92rem;
    color: {TEXT_PRIMARY};
}}
.recent-caption {{
    color: {TEXT_MUTED};
    font-size: 0.8rem;
    margin-top: 3px;
}}
</style>
"""


def render_header(bank: dict) -> None:
    name_col, status_col = st.columns([5, 1], gap="small", vertical_alignment="center")
    with name_col:
        st.subheader(bank["name"])
    with status_col:
        status = compute_status(bank)
        color = STATUS_COLOR[status]
        st.markdown(
            '<div style="display:flex;justify-content:flex-end;">'
            + pill_html(f"● {status}", color, f"{color}22", f"{color}66", size="1.05rem")
            + "</div>",
            unsafe_allow_html=True,
        )
    st.caption(f"{bank['ticker']} · cert {bank['cert']} · rssd {bank['rssd']}")
    st.write(bank["summary"])


def render_key_alerts(bank: dict) -> None:
    st.markdown('<div class="panel-title" style="margin-bottom:8px;">KEY ALERTS</div>', unsafe_allow_html=True)
    alerts = list(bank.get("alerts", []))
    fundamentals = bank.get("fundamentals")
    if fundamentals:
        for f in fundamentals["features"]:
            if f["status"] == "outside range":
                alerts.append({
                    "severity": "high",
                    "text": f"{f['name']} outside range: {f['latest']} vs threshold {f['threshold']}",
                })
    # Model feature drivers pushing distress_prob up — same real local
    # attribution as the Model Feature Drivers panel below (load_
    # feature_attribution), surfaced here for whatever's actually moving
    # the score. Protective drivers (negative contribution) never alert;
    # below the 1pp floor a driver is noise, not something to flag.
    for d in bank.get("feature_drivers") or []:
        pp = d["contribution"] * 100
        if pp < 1:
            continue
        alerts.append({
            "severity": "high" if pp >= 5 else "medium",
            "text": (
                f"{d['name']}: {_format_feature_value(d['value'])} vs peer avg "
                f"{_format_feature_value(d['peer_mean'])} — {pp:+.2f}pp toward distress"
            ),
        })
    if not alerts:
        st.markdown(
            f'<div style="color:{TEXT_MUTED};font-size:0.85rem;margin-bottom:10px;">✅ No active alerts this quarter.</div>',
            unsafe_allow_html=True,
        )
        return
    sev_rank = {"high": 0, "medium": 1, "low": 2}
    sev_color = {"high": RED, "medium": AMBER, "low": TEXT_MUTED}
    alerts = sorted(alerts, key=lambda a: sev_rank.get(a["severity"], 1))
    html = ""
    for a in alerts:
        c = sev_color.get(a["severity"], AMBER)
        html += (
            f'<div class="alert-row" style="background:{c}12;border-left:3px solid {c};">'
            f'<span style="color:{c};">⚠</span>'
            f'<span style="color:{TEXT_PRIMARY};font-size:0.85rem;">{a["text"]}</span>'
            "</div>"
        )
    if len(alerts) > 2:
        st.markdown(f'<div class="alerts-scroll">{html}</div>', unsafe_allow_html=True)
        st.caption(f"Showing all {len(alerts)}, most severe first — scroll for more ↕")
    else:
        st.markdown(html, unsafe_allow_html=True)


def render_sentiment(bank: dict) -> None:
    st.markdown('<div class="panel-title">NEWS SENTIMENT — BY QUARTER</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="panel-caption">GDELT news + EDGAR 8-K excerpts · FinBERT (fine-tuned), 3-class</div>',
        unsafe_allow_html=True,
    )
    sentiment = bank["sentiment"]
    if sentiment is None:
        st.info("No sentiment data yet.")
        return
    sent_color = SENTIMENT_LABEL_COLOR.get(sentiment["label"], TEXT_MUTED)
    st.markdown(
        pill_html(sentiment["label"], sent_color, f"{sent_color}22", f"{sent_color}66")
        # A 2-article quarter and a 400-article quarter shouldn't read as
        # equally confident (Jiwon, PR #15 comment 2026-08-15).
        + f'<span style="color:{TEXT_MUTED};font-size:0.85rem;margin-left:8px;">'
        f"· {sentiment['n_items']} articles this quarter</span>",
        unsafe_allow_html=True,
    )
    trend = sentiment["trend"]
    delta = round(trend[-1] - trend[-2], 2) if len(trend) > 1 else None
    st.metric(
        f"{sentiment['n_items']} items scored",
        sentiment["score"],
        delta=delta,
        help="Change vs prior quarter (green = improving, red = worsening)",
    )
    pct = sentiment["pct"]
    st.markdown(
        '<div class="seg-bar">'
        f'<div style="width:{pct["negative"]}%;background:{RED};"></div>'
        f'<div style="width:{pct["neutral"]}%;background:{AMBER};"></div>'
        f'<div style="width:{pct["positive"]}%;background:{GREEN};"></div>'
        "</div>"
        '<div class="seg-bar-legend">'
        f'<span style="color:{RED};">● Negative {pct["negative"]}%</span>'
        f'<span style="color:{AMBER};">● Neutral {pct["neutral"]}%</span>'
        f'<span style="color:{GREEN};">● Positive {pct["positive"]}%</span>'
        "</div>",
        unsafe_allow_html=True,
    )
    st.line_chart(sentiment["trend"], height=160, color=RED)


def render_keywords(bank: dict) -> None:
    st.markdown(
        '<div class="panel-title">STANDOUT KEYWORDS — WHAT DRIVES THE SENTIMENT</div>', unsafe_allow_html=True
    )
    st.markdown(
        '<div class="panel-caption">Keyword clustering / PCA over scored articles (explainability)</div>',
        unsafe_allow_html=True,
    )
    keywords = bank["keywords"]
    if keywords is None:
        st.info("No keyword data yet.")
        return
    st.markdown(f'<div style="color:{RED};font-weight:600;margin-bottom:10px;">Negative drivers</div>', unsafe_allow_html=True)
    st.markdown(_keyword_bars_html(keywords["negative"], RED), unsafe_allow_html=True)
    st.divider()
    st.markdown(f'<div style="color:{GREEN};font-weight:600;margin-bottom:10px;">Positive drivers</div>', unsafe_allow_html=True)
    st.markdown(_keyword_bars_html(keywords["positive"], GREEN), unsafe_allow_html=True)


def render_fundamentals(bank: dict) -> None:
    st.markdown('<div class="panel-title">FUNDAMENTALS RISK PROFILE</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="panel-caption">Composite score from a Gaussian Process classifier · bands ≤80 distress / '
        "80–90 neutral / ≥90 sound. See Metric Breakdown above for the underlying Capital / Credit Quality "
        "ratios — the score itself isn't a simple average of them.</div>",
        unsafe_allow_html=True,
    )

    fundamentals = bank.get("fundamentals")
    if fundamentals is None:
        st.info("No fundamentals data yet.")
        return

    status_color = FUNDAMENTALS_STATUS_COLOR.get(fundamentals["label"], TEXT_MUTED)
    st.markdown(
        pill_html(fundamentals["label"], status_color, f"{status_color}22", f"{status_color}66"),
        unsafe_allow_html=True,
    )

    trend = fundamentals.get("trend")
    delta = (trend[-1] - trend[-2]) if trend and len(trend) > 1 else None
    st.metric(
        "Composite score",
        fundamentals["score"],
        delta=delta,
        help="Change vs prior quarter" if delta is not None else None,
    )
    marker_pct = max(0, min(100, (fundamentals["score"] - 50) / 50 * 100))
    st.markdown(
        f'<div class="gauge-track"><div class="gauge-marker" style="left:{marker_pct}%;"></div></div>'
        '<div class="gauge-ticks"><span>50</span><span>80</span><span>90</span><span>100</span></div>',
        unsafe_allow_html=True,
    )
    if trend:
        quarters = fundamentals.get("quarters") or [f"Q-{i}" for i in range(len(trend) - 1, -1, -1)]
        window = 4
        end_idx = len(quarters) - 1
        start_idx = max(0, end_idx - window + 1)
        shown_quarters = quarters[start_idx:end_idx + 1]
        shown_trend = trend[start_idx:end_idx + 1]
        st.caption(f"Composite score, {shown_quarters[0]} – {shown_quarters[-1]}")
        st.altair_chart(_quarterly_score_chart(shown_quarters, shown_trend, ACCENT_TEAL), width="stretch", theme=None)


def render_metric_breakdown(bank: dict) -> None:
    status_colors = {"within range": GREEN, "near threshold": AMBER, "outside range": RED}
    fundamentals = bank.get("fundamentals")
    if fundamentals is None:
        return

    n_features = len(fundamentals["features"])
    n_flagged = sum(1 for f in fundamentals["features"] if f["status"] != "within range")
    flag_label = f"{n_flagged} flagged" if n_flagged else "all clear"
    st.markdown(
        '<div class="panel-title">METRIC BREAKDOWN — CAPITAL · CREDIT QUALITY · LIQUIDITY · PROFITABILITY</div>'
        f'<div class="panel-caption">FFIEC Call Report · {n_features} metrics · {flag_label}</div>',
        unsafe_allow_html=True,
    )
    all_feature_names = [f["name"] for f in fundamentals["features"]]
    shown_names = st.multiselect(
        "Metrics shown",
        all_feature_names,
        default=all_feature_names,
        key=f"metric_filter_{bank['ticker']}",
        help="Choose which metrics appear in the breakdown table below",
    )
    shown_features = [f for f in fundamentals["features"] if f["name"] in shown_names]

    rows_html = ""
    for group in CAMELS_GROUPS:
        group_features = [f for f in shown_features if f["group"] == group]
        for f in group_features:
            c = status_colors.get(f["status"], TEXT_MUTED)
            badge = pill_html(f["status"], c, f"{c}22", f"{c}66", size="0.72rem")
            qoq = _qoq_delta_html(f["prior"], f["latest"], f["threshold"])
            rows_html += (
                f"<tr><td>{f['name']}</td><td>{f['prior']}</td><td>{f['latest']}</td><td>{qoq}</td>"
                f"<td>{f['threshold']}</td><td>{badge}</td></tr>"
            )
    qoq_help = (
        "Arrow = raw direction of the number. Color = whether that move is good (green) or bad (red) for "
        "this feature. This profile is built to extend — additional Call Report ratios (e.g. net interest "
        "margin, loan growth) can be added per group as analysts need them."
    )
    if not shown_features:
        st.caption("No metrics selected.")
    else:
        st.markdown(
            "<table class='fund-table'><thead><tr>"
            "<th>Feature</th><th>Prior Q</th><th>Latest Q</th>"
            f'<th>Δ QoQ <details class="info-pop"><summary>?</summary>'
            f'<div class="info-pop-body">{qoq_help}</div></details></th>'
            "<th>Threshold</th><th>Status</th>"
            f"</tr></thead><tbody>{rows_html}</tbody></table>",
            unsafe_allow_html=True,
        )

    feature_names = [f["name"] for f in fundamentals["features"]]
    st.markdown("<div style='margin-top:16px;'></div>", unsafe_allow_html=True)
    selected = st.selectbox(
        "View a metric's history",
        feature_names,
        key=f"feature_history_{bank['ticker']}",
        help="Pick a metric, then open the expander below to see its trend over the last 5 quarters",
    )
    chosen = next(f for f in fundamentals["features"] if f["name"] == selected)
    hist_color = status_colors.get(chosen["status"], ACCENT_TEAL)
    with st.expander(f"View trend — {selected}"):
        st.altair_chart(
            _feature_history_chart(FEATURE_QUARTERS, chosen["history"], hist_color),
            width="stretch", theme=None,
        )


def render_feature_drivers(bank: dict) -> None:
    st.markdown(
        '<div class="panel-title">MODEL FEATURE DRIVERS — WHY THIS SCORE</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="panel-caption">Top 5 features by effect on this bank\'s score</div>',
        unsafe_allow_html=True,
    )
    drivers = bank.get("feature_drivers")
    if not drivers:
        st.info("No attribution available — this bank's latest quarter isn't scored by gp50_prod_v1.")
        return

    rows_html = ""
    for d in drivers:
        pp = d["contribution"] * 100
        c = RED if pp > 0 else GREEN
        direction = "toward distress" if pp > 0 else "protective"
        contrib_html = f'<span style="color:{c};font-weight:600;">{pp:+.2f}pp</span> {direction}'
        name = d["name"] + (
            f' <span style="color:{TEXT_MUTED};font-size:0.72rem;">(imputed)</span>' if d["is_imputed"] else ""
        )
        rows_html += (
            f"<tr><td>{name}</td>"
            f"<td>{_format_feature_value(d['value'])}</td>"
            f"<td>{_format_feature_value(d['peer_mean'])}</td>"
            f"<td>{contrib_html}</td></tr>"
        )
    st.markdown(
        "<table class='fund-table'><thead><tr>"
        "<th>Feature</th><th>Latest</th>"
        "<th>Peer avg, this Q</th><th>Contribution to distress_prob</th>"
        f"</tr></thead><tbody>{rows_html}</tbody></table>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Contribution = distress_prob with this feature at its actual value, minus distress_prob with it "
        "swapped to the peer average (others held fixed) — how much this specific input, as it stands, "
        "moves this bank's score away from a typical peer."
    )


def render_recent_items(bank: dict) -> None:
    st.markdown('<div class="panel-title">RECENT ITEMS</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="panel-caption">Latest scored articles & filings for this bank</div>', unsafe_allow_html=True
    )
    items = bank["recent_items"]
    if not items:
        st.info("No recent items yet.")
        return

    today = date.today()
    parsed = [
        {"title": t, "label": lb, "source": s, "date": today - timedelta(days=d), "feed": f}
        for t, lb, s, d, f in items
    ]
    min_d = min(p["date"] for p in parsed)
    max_d = max(p["date"] for p in parsed)
    date_range = st.date_input(
        "Filter by date",
        value=(min_d, max_d),
        min_value=min_d,
        max_value=max_d,
        key=f"recent_date_range_{bank['ticker']}",
    )
    if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
        start, end = date_range
    else:
        start, end = min_d, max_d

    filtered = sorted((p for p in parsed if start <= p["date"] <= end), key=lambda p: p["date"], reverse=True)
    if not filtered:
        st.caption("No items in the selected date range.")
        return

    page_size = 5
    n_pages = (len(filtered) - 1) // page_size + 1
    page_key = f"recent_page_{bank['ticker']}"
    if page_key not in st.session_state:
        st.session_state[page_key] = 0
    # Clamp rather than reset to 0: if the date filter just shrank the set
    # out from under a later page, land on the new last page instead of
    # silently jumping back to the start.
    page = min(st.session_state[page_key], n_pages - 1)
    st.session_state[page_key] = page

    label_color = {"negative": RED, "neutral": AMBER, "positive": GREEN}
    html_parts = []
    for p in filtered[page * page_size : (page + 1) * page_size]:
        c = label_color.get(p["label"], TEXT_MUTED)
        badge = pill_html(p["label"], c, f"{c}22", f"{c}66", size="0.7rem")
        age_days = (today - p["date"]).days
        age_str = "today" if age_days == 0 else f"{age_days}d ago"
        html_parts.append(
            f'<div class="recent-item" style="--item-color:{c};">'
            f'<div class="recent-title">{p["title"]} {badge}</div>'
            f'<div class="recent-caption">{p["source"]} · {age_str} · {p["feed"]}</div>'
            "</div>"
        )
    st.markdown("".join(html_parts), unsafe_allow_html=True)

    if n_pages > 1:
        prev_col, label_col, next_col = st.columns([1, 2, 1])
        with prev_col:
            if st.button("◀ Prev", key=f"recent_prev_{bank['ticker']}", disabled=page == 0, width="stretch"):
                st.session_state[page_key] = page - 1
                st.rerun()
        with label_col:
            st.markdown(
                f'<div style="text-align:center;color:{TEXT_MUTED};font-size:0.8rem;padding-top:8px;">'
                f"Page {page + 1} of {n_pages} · {len(filtered)} items</div>",
                unsafe_allow_html=True,
            )
        with next_col:
            if st.button("Next ▶", key=f"recent_next_{bank['ticker']}", disabled=page >= n_pages - 1, width="stretch"):
                st.session_state[page_key] = page + 1
                st.rerun()


def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)

    title_col, search_col = st.columns([3, 1])
    with title_col:
        st.markdown(
            f'<span style="font-size:1.7rem;font-weight:800;color:{TEXT_PRIMARY};">Bank Stability Monitor</span> '
            f'<span style="color:{TEXT_MUTED};font-size:0.95rem;">PNC Capstone · early-warning dashboard</span> '
            + pill_html("CONCEPT MOCKUP · ILLUSTRATIVE DATA", AMBER, f"{AMBER}14", f"{AMBER}55"),
            unsafe_allow_html=True,
        )
    bank_names = list(MOCK_BANKS.keys())

    with search_col:
        # Scoped to the 4 demo banks (MOCK_BANKS), same set the pills below
        # offer — selectbox gives native type-to-filter for free. Setting
        # the pills' own session_state key (before they're instantiated
        # below) is how the two stay in sync without a callback.
        searched = st.selectbox(
            "Search",
            options=bank_names,
            index=None,
            placeholder="Search a bank... (e.g. Wells Fargo)",
            label_visibility="collapsed",
            key="bank_search",
        )
        if searched:
            st.session_state["bank_pills"] = searched

    st.divider()

    with st.container(key="bank-picker"):
        demo_name = st.pills(
            "104 tracked · demo:", bank_names, default=bank_names[0], key="bank_pills"
        )
    if not demo_name:
        demo_name = bank_names[0]
    bank = MOCK_BANKS[demo_name]

    render_header(bank)
    st.divider()

    # Reading order for a risk analyst: what needs attention right now (Key
    # Alerts), then the model's verdict and why it reached it side by side
    # (Fundamentals Risk Profile + Model Feature Drivers) — conclusion and
    # explanation stay adjacent — then the raw evidence backing that verdict
    # (Metric Breakdown).
    with st.container(border=True, key="card-alerts"):
        render_key_alerts(bank)

    fund_col, drivers_col = st.columns(2)
    with fund_col:
        with st.container(border=True, key="card-fundamentals", height="stretch"):
            render_fundamentals(bank)
    with drivers_col:
        with st.container(border=True, key="card-feature-drivers", height="stretch"):
            render_feature_drivers(bank)

    with st.container(border=True, key="card-metric-breakdown"):
        render_metric_breakdown(bank)

    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True, key="card-sentiment", height="stretch"):
            render_sentiment(bank)
    with col2:
        with st.container(border=True, key="card-keywords", height="stretch"):
            kw_col, recent_col = st.columns([1, 1])
            with kw_col:
                render_keywords(bank)
            with recent_col:
                with st.container(key="recent-in-keywords"):
                    render_recent_items(bank)

    st.divider()
    st.caption(
        "How to read this screen — The fundamentals score comes from a "
        "Gaussian Process classifier over Call Report features; sentiment "
        "from a BERT-based 3-class model trained on LLM-assisted labels. Key "
        "Alerts surfaces any metric outside its threshold plus non-ratio "
        "flags (e.g. open enforcement actions) — it isn't itself part of "
        "the composite score. Sources: GDELT DOC "
        "2.0, SEC EDGAR (8-K/10-Q/10-K), FFIEC/FDIC unified dataset, "
        "Fed/FDIC enforcement actions."
    )


if __name__ == "__main__":
    main()
