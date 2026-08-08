"""Bank Stability Monitor — Streamlit dashboard (Phase 5).

Mirrors the first-screen concept in dashboard/concept/ (mentor discussion,
2026-07-12). Backed by mock data for now — real reads against the index and
score tables land once Phase 2 (scoring) and Phase 3 (index) exist; see
db/migrations/011_scoring_tables.sql for the current shared contract.

Extends the original two-signal concept (fundamentals x news sentiment) with
two more panels sourced from eda/reports/ming_sources_eda.md's findings:
price risk (market_daily — dense coverage, leading signal) and CFPB
complaints (cfpb_complaint — consumer-facing banks only, reactive/severity
signal, now framed as "Customer Complaint Risk"). A macro banner
(fred_observation) is shown once, globally, since EDA found it's a systemic
backdrop rather than a per-bank signal.

Fundamentals is organized as a CAMELS-style profile grouped into Capital,
Credit Quality, Liquidity, and Profitability (Management/Sensitivity metrics
that don't fit a quarterly ratio format — enforcement actions, unrealized
losses vs. threshold — surface as Key Alerts instead). It's the first panel
on the page and spans full width since it now carries alerts + a grouped
table + a per-metric history view. The composite Stable/Watch/Elevated Risk
status badge is still driven solely by fundamentals x sentiment per the
original rubric; folding price/complaints into that combined score is a
scoring-methodology decision for the team, not made here.

Also adds trend/momentum and peer-percentile context, framed for the
external counterparty-risk-analyst persona (monitors exposure to these
banks from another firm) — comparisons are scoped to US peer groups. Peer
groups and percentiles are illustrative mock groupings, not a computed
cohort — see RISKS in the accompanying review notes.

Visual styling (cards, pills, gauge, segmented bar, keyword/category bars,
sparklines) is a custom CSS + Altair layer on top of Streamlit's default
widgets — see render_* functions below and the CSS block in main().
"""

from datetime import date, timedelta
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Bank Stability Monitor", layout="wide")

# Wired to index/mock/*.csv, column-for-column matches to
# db/migrations/012_index_tables.sql (bank_index_score / bank_index_feature).
# Swapping to a real DB read later is a change to _INDEX_SCORES/_INDEX_FEATURES
# only — load_fundamentals_mock() and every render_* function stay as-is.
_INDEX_MOCK_DIR = Path(__file__).resolve().parent.parent / "index" / "mock"
_INDEX_SCORES = pd.read_csv(_INDEX_MOCK_DIR / "bank_index_score_sample.csv")
_INDEX_FEATURES = pd.read_csv(_INDEX_MOCK_DIR / "bank_index_feature_sample.csv")

_BAND_LABEL = {"sound": "Sound", "neutral": "Neutral", "distress": "Distress signal"}
_STATUS_DISPLAY = {
    "within_range": "within range",
    "near_threshold": "near threshold",
    "breach": "outside range",
}
# feature_name -> CAMELS group; not a schema column (see 012's comment on
# bank_index_feature), so the mapping lives here alongside the other display
# concerns the dashboard owns.
_FEATURE_GROUP = {
    "tier1_capital_ratio": "Capital",
    "npl_ratio": "Credit Quality",
    "cre_to_capital": "Credit Quality",
    "liquidity_ratio": "Liquidity",
    "fee_income_ratio": "Profitability",
}


def _quarter_label(quarter_end_date: str) -> str:
    year, month, _ = quarter_end_date.split("-")
    q = {"03": "Q1", "06": "Q2", "09": "Q3", "12": "Q4"}[month]
    return f"{q} '{year[2:]}"


def _format_threshold(text) -> str:
    # threshold_text is NULL until Ming's scoring loader backfills it
    # (backtest not done yet) — render code needs a plain string, not NaN.
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
        values = list(group_df.sort_values("quarter_end_date")["value"])
        row = group_df.iloc[-1]
        features.append(
            {
                "group": _FEATURE_GROUP.get(name, "Capital"),
                "name": row["display_name"],
                "prior": values[-2] if len(values) > 1 else values[-1],
                "latest": values[-1],
                "threshold": _format_threshold(row["threshold_text"]),
                # status is NULL until Ming's scoring loader backfills it too —
                # falls back to a fourth, unstyled state (render_fundamentals'
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

MACRO = {
    "series": "St. Louis Fed Financial Stress Index",
    "status": "Normal",
    "note": (
        "Industry-wide backdrop, not a per-bank signal — EDA shows this "
        "spikes at 2008 / 2020 / 2023 and lifts every bank's risk together. "
        "Currently in its baseline range."
    ),
}

FEATURE_QUARTERS = ["Q1 '25", "Q2 '25", "Q3 '25", "Q4 '25", "Q1 '26"]
MONTHLY_LABELS = [
    "Aug '25", "Sep '25", "Oct '25", "Nov '25", "Dec '25", "Jan '26",
    "Feb '26", "Mar '26", "Apr '26", "May '26", "Jun '26", "Jul '26",
]

MOCK_BANKS = {
    "Wells Fargo": {
        "name": "Wells Fargo & Company",
        "ticker": "WFC",
        "cert": "3511",
        "rssd": "451965",
        "status": "Watch",
        "summary": (
            "Fundamentals sit in the neutral band, but news flow is "
            "persistently negative around regulatory penalties — the "
            "pattern the early-warning view is built to surface."
        ),
        "momentum": {"duration": "2nd consecutive quarter in Watch", "direction": "deteriorating"},
        "peer_group": "US Systemically Important Banks (G-SIBs)",
        "peer_group_n": 8,
        "sentiment": {
            "score": -0.31,
            "label": "Negative",
            "n_items": 182,
            "pct": {"negative": 48, "neutral": 30, "positive": 22},
            "trend": [
                0.05, 0.02, -0.04, -0.02, -0.08, -0.06, -0.11, -0.09, -0.14,
                -0.12, -0.17, -0.15, -0.20, -0.18, -0.23, -0.21, -0.26, -0.24,
                -0.29, -0.27, -0.31,
            ],
            "peer_percentile_negative": 72,
        },
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
        "alerts": [
            {"severity": "high", "text": "2 open Fed/FDIC enforcement actions this quarter (up from 1) — regulatory conduct risk"},
        ],
        "fundamentals": load_fundamentals_mock("wfc"),
        "recent_items": [
            ("Regulator signals fresh penalties over consumer-billing practices", "negative", "reuters.com", 2, "GDELT"),
            ("8-K — Item 8.01: settlement of outstanding consent order disclosed", "neutral", "sec.gov", 3, "EDGAR"),
            ("Bank raises quarterly dividend, extends buyback", "positive", "marketwatch.com", 5, "GDELT"),
            ("Analysts flag slower fee income amid asset-cap constraints", "negative", "ft.com", 6, "GDELT"),
        ],
        "price_risk": {
            "risk_badge": "Moderate",
            "return_30d": -2.8,
            "vol_30d": 0.24,
            "largest_move": {"pct": -3.4, "date": "Jul 18"},
            "sparkline_30d": [
                100.0, 99.7, 100.1, 99.8, 100.0, 99.6, 99.9, 99.4, 99.7, 99.2,
                99.5, 99.0, 98.7, 99.0, 98.5, 98.8, 98.3, 98.6, 95.3, 95.6,
                96.0, 95.7, 96.1, 95.8, 96.2, 95.9, 96.3, 96.7, 97.0, 97.2,
            ],
            "vol_3y": 0.27,
            "max_drawdown_3y": -0.35,
            "coverage": "103/104 banks have ≥5yr daily history",
            "peer_percentile_vol": 40,
            "peer_rank": "4th least volatile of 8 G-SIB peers",
        },
        "complaints": {
            "total": 168000,
            "risk_badge": "Moderate",
            "coverage_tier": "High coverage — top 10 nationally by volume",
            "yoy_growth_pct": 18,
            "timely_response_rate": 99,
            "top_categories": [
                ("Credit Reporting", 26), ("Mortgage", 24), ("Checking/Savings", 20),
                ("Credit Card", 17), ("Loans", 13),
            ],
            "monthly_trend": [980, 1010, 1050, 1020, 1080, 1110, 1140, 1130, 1180, 1200, 1230, 1260],
            "keywords": [
                ("countrywide (legacy mortgage)", 0.88),
                ("hamp / assistance center", 0.71),
                ("identity theft", 0.58),
                ("everyday checking", 0.44),
            ],
        },
    },
    "Western Alliance": {
        "name": "Western Alliance Bancorporation",
        "ticker": "WAL",
        "cert": "57512",
        "rssd": "3138146",
        "status": "Elevated Risk",
        "summary": (
            "Fundamentals dipped below the 80 distress line while negative "
            "sentiment accelerates — both axes now point the same way, "
            "which is the strongest configuration of the warning signal."
        ),
        "momentum": {"duration": "2nd consecutive quarter worsening", "direction": "deteriorating"},
        "peer_group": "Regional banks, CRE-exposed",
        "peer_group_n": 20,
        "sentiment": {
            "score": -0.47,
            "label": "Negative",
            "n_items": 64,
            "pct": {"negative": 62, "neutral": 26, "positive": 12},
            "trend": [
                0.00, -0.03, -0.07, -0.05, -0.11, -0.09, -0.15, -0.13, -0.19,
                -0.17, -0.24, -0.22, -0.29, -0.27, -0.34, -0.32, -0.39, -0.37,
                -0.44, -0.42, -0.47,
            ],
            "peer_percentile_negative": 90,
        },
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
        "alerts": [
            {"severity": "high", "text": "2 open Fed/FDIC enforcement actions this quarter (up from 1)"},
        ],
        "fundamentals": load_fundamentals_mock("wal"),
        "recent_items": [
            ("Ratings agency places regional lender on negative watch", "negative", "reuters.com", 0, "GDELT"),
            ("8-K — Item 7.01: investor presentation on liquidity position", "neutral", "sec.gov", 1, "EDGAR"),
            ("CRE concentration draws renewed analyst scrutiny", "negative", "barrons.com", 2, "GDELT"),
            ("Completed capital raise shores up balance sheet", "positive", "marketwatch.com", 4, "GDELT"),
        ],
        "price_risk": {
            "risk_badge": "High",
            "return_30d": -9.6,
            "vol_30d": 0.58,
            "largest_move": {"pct": -11.4, "date": "Jul 09"},
            "sparkline_30d": [
                100.0, 99.0, 98.5, 97.8, 97.0, 96.5, 95.8, 95.0, 94.2, 83.5,
                84.5, 85.5, 86.0, 85.2, 86.8, 87.5, 86.9, 88.0, 87.4, 88.8,
                88.0, 89.2, 88.5, 89.8, 89.0, 90.2, 89.6, 90.8, 90.2, 90.4,
            ],
            "vol_3y": 0.55,
            "max_drawdown_3y": -0.74,
            "coverage": "103/104 banks have ≥5yr daily history",
            "peer_percentile_vol": 95,
            "peer_rank": "2nd most volatile of 20 CRE-exposed peers",
        },
        "complaints": {
            "total": 40,
            "risk_badge": "Insufficient data",
            "coverage_tier": "Thin coverage — under 100 total (commercial bank, near-absent per EDA)",
            "yoy_growth_pct": None,
            "timely_response_rate": None,
            "top_categories": [],
            "monthly_trend": [],
            "keywords": [],
        },
    },
    # Placeholder demo entries — no concept mockup yet for sentiment /
    # fundamentals, kept minimal on purpose. Price + complaints panels are
    # filled in below since those signals score every bank regardless of a
    # news-sentiment story (that's the point of the price panel per EDA).
    "PNC Financial": {
        "name": "PNC Financial Services Group",
        "ticker": "PNC",
        "cert": "6384",
        "rssd": "817824",
        "status": "Stable",
        "summary": "Placeholder — no illustrative sentiment data drafted yet. Fundamentals below are wired to index/mock/*.csv.",
        "momentum": {"duration": "steady", "direction": "stable"},
        "peer_group": "Super-regional / diversified banks",
        "peer_group_n": 15,
        "sentiment": None,
        "keywords": None,
        "alerts": [],
        "fundamentals": load_fundamentals_mock("pnc"),
        "recent_items": [],
        "price_risk": {
            "risk_badge": "Low",
            "return_30d": 1.4,
            "vol_30d": 0.19,
            "largest_move": {"pct": 2.1, "date": "Jul 22"},
            "sparkline_30d": [
                100.0, 100.2, 99.9, 100.3, 100.1, 100.4, 100.0, 100.5, 100.2, 100.6,
                100.3, 100.7, 100.4, 100.8, 100.5, 100.9, 100.6, 101.0, 100.7, 101.1,
                100.8, 101.2, 103.3, 102.9, 103.1, 102.7, 103.0, 101.9, 101.6, 101.4,
            ],
            "vol_3y": 0.30,
            "max_drawdown_3y": -0.30,
            "coverage": "103/104 banks have ≥5yr daily history",
            "peer_percentile_vol": 50,
            "peer_rank": "8th of 15 super-regional peers",
        },
        "complaints": {
            "total": 31000,
            "risk_badge": "Low",
            "coverage_tier": "High coverage — top 10 nationally by volume",
            "yoy_growth_pct": 15,
            "timely_response_rate": 98,
            "top_categories": [
                ("Checking/Savings", 30), ("Mortgage", 22), ("Credit Card", 18),
                ("Credit Reporting", 16), ("Loans", 14),
            ],
            "monthly_trend": [190, 205, 200, 215, 220, 210, 225, 230, 235, 228, 240, 248],
            "keywords": [
                ("virtual wallet", 0.81),
                ("national city (legacy)", 0.62),
                ("link accounts", 0.49),
                ("escrow", 0.38),
            ],
        },
    },
    "JPMorgan Chase": {
        "name": "JPMorgan Chase & Co.",
        "ticker": "JPM",
        "cert": "628",
        "rssd": "852218",
        "status": "Stable",
        "summary": "Placeholder — no illustrative sentiment data drafted yet. Fundamentals below are wired to index/mock/*.csv.",
        "momentum": {"duration": "steady", "direction": "stable"},
        "peer_group": "US Systemically Important Banks (G-SIBs)",
        "peer_group_n": 8,
        "sentiment": None,
        "keywords": None,
        "alerts": [],
        "fundamentals": load_fundamentals_mock("jpm"),
        "recent_items": [],
        "price_risk": {
            "risk_badge": "Low",
            "return_30d": 2.3,
            "vol_30d": 0.17,
            "largest_move": {"pct": 2.6, "date": "Jul 15"},
            "sparkline_30d": [
                100.0, 100.2, 100.4, 100.3, 100.5, 100.7, 100.6, 100.8, 101.0, 100.9,
                101.1, 101.3, 101.2, 101.4, 101.6, 104.2, 103.9, 104.1, 103.8, 104.0,
                103.7, 103.9, 103.6, 103.8, 103.5, 103.7, 103.4, 103.6, 102.9, 102.3,
            ],
            "vol_3y": 0.26,
            "max_drawdown_3y": -0.28,
            "coverage": "103/104 banks have ≥5yr daily history",
            "peer_percentile_vol": 25,
            "peer_rank": "7th least volatile of 8 G-SIB peers",
        },
        "complaints": {
            "total": 168000,
            "risk_badge": "Low",
            "coverage_tier": "High coverage — top 10 nationally by volume",
            "yoy_growth_pct": 16,
            "timely_response_rate": 99,
            "top_categories": [
                ("Credit Card", 34), ("Checking/Savings", 20), ("Credit Reporting", 19),
                ("Mortgage", 14), ("Loans", 13),
            ],
            "monthly_trend": [1020, 1040, 1060, 1050, 1090, 1110, 1130, 1150, 1140, 1170, 1190, 1210],
            "keywords": [
                ("sapphire / freedom cards", 0.85),
                ("amazon co-brand", 0.63),
                ("marriott co-brand", 0.41),
            ],
        },
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

STATUS_STYLE = {
    "Stable": GREEN,
    "Watch": AMBER,
    "Elevated Risk": RED,
    "Imminent Disruption": RED,
}

RISK_BADGE_COLOR = {
    "Low": GREEN,
    "Moderate": AMBER,
    "High": RED,
    "Insufficient data": TEXT_MUTED,
}

MOMENTUM_ICON = {"improving": "🟢", "stable": "⚪", "deteriorating": "🔴"}

CAMELS_GROUPS = ["Capital", "Credit Quality", "Liquidity", "Profitability"]


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


def _price_sparkline_chart(values: list, color: str, height: int = 150) -> alt.Chart:
    df = pd.DataFrame({"day": range(1, len(values) + 1), "value": values})
    y_pad = max(0.5, (max(values) - min(values)) * 0.15)
    return (
        alt.Chart(df)
        .mark_line(color=color, strokeWidth=2, point=alt.OverlayMarkDef(filled=True, size=25, color=color))
        .encode(
            x=alt.X("day:Q", title="Trading Day", axis=alt.Axis(
                labelColor=TEXT_MUTED, titleColor=TEXT_MUTED, labelFontSize=10, titleFontSize=11,
                grid=False, domainColor=BORDER, tickColor=BORDER, format="d",
            )),
            y=alt.Y("value:Q", title="Indexed Price (Day 1 = 100)", scale=alt.Scale(domain=[min(values) - y_pad, max(values) + y_pad]), axis=alt.Axis(
                labelColor=TEXT_MUTED, titleColor=TEXT_MUTED, labelFontSize=10, titleFontSize=11,
                gridColor=BORDER, domainColor=BORDER, tickColor=BORDER,
            )),
            tooltip=[alt.Tooltip("day:Q", title="Trading day"), alt.Tooltip("value:Q", title="Indexed price", format=".1f")],
        )
        .properties(height=height, background="transparent")
        .configure_view(strokeWidth=0)
    )


def _monthly_trend_chart(labels: list, values: list, color: str, height: int = 150) -> alt.Chart:
    df = pd.DataFrame({"month": labels, "count": values})
    return (
        alt.Chart(df)
        .mark_line(color=color, strokeWidth=2, point=alt.OverlayMarkDef(filled=True, size=30, color=color))
        .encode(
            x=alt.X("month:N", sort=labels, title="Month", axis=alt.Axis(
                labelColor=TEXT_MUTED, titleColor=TEXT_MUTED, labelFontSize=10, titleFontSize=11,
                grid=False, domainColor=BORDER, tickColor=BORDER, labelAngle=-40,
            )),
            y=alt.Y("count:Q", title="Complaints / Month", axis=alt.Axis(
                labelColor=TEXT_MUTED, titleColor=TEXT_MUTED, labelFontSize=10, titleFontSize=11,
                gridColor=BORDER, domainColor=BORDER, tickColor=BORDER,
            )),
            tooltip=[alt.Tooltip("month:N", title="Month"), alt.Tooltip("count:Q", title="Complaints")],
        )
        .properties(height=height, background="transparent")
        .configure_view(strokeWidth=0)
    )


def _category_bar_chart(rows: list, color: str) -> alt.Chart:
    df = pd.DataFrame(rows, columns=["category", "pct"])
    return (
        alt.Chart(df)
        .mark_bar(color=color, cornerRadiusTopRight=3, cornerRadiusBottomRight=3)
        .encode(
            y=alt.Y("category:N", sort="-x", title=None, axis=alt.Axis(
                labelColor=TEXT_MUTED, labelFontSize=11, domain=False, ticks=False,
            )),
            x=alt.X("pct:Q", title=None, axis=alt.Axis(
                labelColor=TEXT_MUTED, labelFontSize=10, gridColor=BORDER, domainColor=BORDER,
            )),
            tooltip=[alt.Tooltip("category:N", title="Category"), alt.Tooltip("pct:Q", title="% of complaints")],
        )
        .properties(height=155, background="transparent")
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
    status_color = STATUS_STYLE.get(bank["status"], TEXT_MUTED)
    left, right = st.columns([4, 1])
    with left:
        st.subheader(bank["name"])
        st.caption(f"{bank['ticker']} · cert {bank['cert']} · rssd {bank['rssd']}")
        st.write(bank["summary"])
    with right:
        badge = pill_html(f"● {bank['status']}", status_color, f"{status_color}22", f"{status_color}66")
        st.markdown(f'<div style="text-align:right;">{badge}</div>', unsafe_allow_html=True)
        momentum = bank.get("momentum")
        if momentum:
            icon = MOMENTUM_ICON.get(momentum["direction"], "⚪")
            st.markdown(
                f'<div style="text-align:right;color:{TEXT_MUTED};font-size:0.8rem;margin-top:6px;">'
                f"{icon} {momentum['duration']}</div>",
                unsafe_allow_html=True,
            )


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
    st.markdown('<div class="panel-title">NEWS SENTIMENT — ROLLING 30 DAYS</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="panel-caption">GDELT news + EDGAR 8-K excerpts · FinBERT (fine-tuned), 3-class</div>',
        unsafe_allow_html=True,
    )
    sentiment = bank["sentiment"]
    if sentiment is None:
        st.info("No sentiment data yet.")
        return
    delta = round(sentiment["trend"][-1] - sentiment["trend"][-8], 2)
    st.metric(
        f"{sentiment['label']} · {sentiment['n_items']} items scored",
        sentiment["score"],
        delta=delta,
        help="Change vs ~1 week ago (green = improving, red = worsening)",
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
    peer_pct = sentiment.get("peer_percentile_negative")
    if peer_pct is not None:
        st.caption(f"More negative than {peer_pct}% of {bank['peer_group']} peers")


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
    st.markdown(
        '<div class="panel-title">FUNDAMENTALS RISK PROFILE — CAPITAL · CREDIT QUALITY · LIQUIDITY · PROFITABILITY</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="panel-caption">FFIEC Call Report + Fed/FDIC enforcement actions · composite score from a '
        "Gaussian Process classifier · bands ≤80 distress / 80–90 neutral / ≥90 sound. Rows below are the "
        "underlying metrics shown for transparency — the score itself isn't a simple average of them.</div>",
        unsafe_allow_html=True,
    )

    fundamentals = bank.get("fundamentals")
    status_colors = {"within range": GREEN, "near threshold": AMBER, "outside range": RED}

    if fundamentals is None:
        render_key_alerts(bank)
        st.info("No fundamentals data yet.")
        return

    left_col, right_col = st.columns([2, 3])

    with left_col:
        render_key_alerts(bank)
        trend = fundamentals.get("trend")
        delta = (trend[-1] - trend[-2]) if trend and len(trend) > 1 else None
        st.metric(
            fundamentals["label"],
            fundamentals["score"],
            delta=delta,
            help="Change vs prior quarter" if delta is not None else None,
        )
        peer_pct = fundamentals.get("peer_percentile")
        if peer_pct is not None:
            st.caption(f"Safer than {peer_pct}% of {bank['peer_group']} peers (n={bank['peer_group_n']})")
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

    with right_col:
        n_features = len(fundamentals["features"])
        n_flagged = sum(1 for f in fundamentals["features"] if f["status"] != "within range")
        flag_label = f"{n_flagged} flagged" if n_flagged else "all clear"
        st.markdown(
            f'<div class="panel-title" style="margin-bottom:2px;">METRIC BREAKDOWN</div>'
            f'<div class="panel-caption" style="margin-bottom:10px;">{n_features} metrics · {flag_label}</div>',
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
            "this feature. This profile is built to extend — additional Call Report ratios (e.g. total capital "
            "ratio, net interest margin, loan growth) can be added per group as analysts need them."
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


def render_macro_banner() -> None:
    st.markdown(
        f'<div style="border:1px solid {BORDER};background:{BG_CARD};border-radius:10px;'
        f'padding:10px 16px;margin-bottom:14px;color:{TEXT_PRIMARY};font-size:0.85rem;">'
        f"🌐 <strong>Macro backdrop — {MACRO['series']}: {MACRO['status']}</strong> · "
        f'<span style="color:{TEXT_MUTED};">{MACRO["note"]}</span></div>',
        unsafe_allow_html=True,
    )


def render_price_risk(bank: dict) -> None:
    st.markdown('<div class="panel-title">PRICE RISK</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="panel-caption">yfinance daily close · market-observed, leading signal, densest coverage</div>',
        unsafe_allow_html=True,
    )
    price_risk = bank.get("price_risk")
    if price_risk is None:
        st.info("No price data yet.")
        return

    badge_color = RISK_BADGE_COLOR.get(price_risk["risk_badge"], TEXT_MUTED)
    st.markdown(
        pill_html(f"Market Risk: {price_risk['risk_badge']}", badge_color, f"{badge_color}22", f"{badge_color}66"),
        unsafe_allow_html=True,
    )
    st.markdown("<div style='margin-top:12px;'></div>", unsafe_allow_html=True)

    ret = price_risk["return_30d"]
    ret_color = GREEN if ret >= 0 else RED
    move = price_risk["largest_move"]
    move_color = GREEN if move["pct"] >= 0 else RED

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(stat_tile_html("30-Day Return", f"{ret:+.1f}%", ret_color), unsafe_allow_html=True)
    with c2:
        st.markdown(stat_tile_html("30-Day Realized Vol", f"{price_risk['vol_30d']:.2f}"), unsafe_allow_html=True)
    with c3:
        st.markdown(
            stat_tile_html("Largest Daily Move", f"{move['pct']:+.1f}% ({move['date']})", move_color),
            unsafe_allow_html=True,
        )

    st.markdown("<div style='margin-top:14px;'></div>", unsafe_allow_html=True)
    st.caption("Last 30 trading days")
    st.altair_chart(
        _price_sparkline_chart(price_risk["sparkline_30d"], ACCENT_TEAL), width="stretch", theme=None
    )

    with st.expander("3-year context & peer comparison"):
        st.caption(price_risk["coverage"])
        e1, e2 = st.columns(2)
        with e1:
            st.markdown(stat_tile_html("3-Year Annualized Volatility", f"{price_risk['vol_3y']:.2f}"), unsafe_allow_html=True)
        with e2:
            st.markdown(
                stat_tile_html("3-Year Max Drawdown", f"{price_risk['max_drawdown_3y']:.0%}", RED),
                unsafe_allow_html=True,
            )
        peer_rank = price_risk.get("peer_rank")
        if peer_rank:
            st.markdown(
                f'<div style="margin-top:12px;font-size:0.85rem;color:{TEXT_MUTED};">'
                f'{peer_rank} — riskier than <span style="color:{TEXT_PRIMARY};font-weight:600;">'
                f'{price_risk["peer_percentile_vol"]}%</span> of {bank["peer_group"]} peers</div>',
                unsafe_allow_html=True,
            )


def render_complaints(bank: dict) -> None:
    st.markdown('<div class="panel-title">CUSTOMER COMPLAINT RISK</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="panel-caption">cfpb_complaint · reactive/conduct-risk signal, consumer-facing banks only</div>',
        unsafe_allow_html=True,
    )
    complaints = bank.get("complaints")
    if complaints is None:
        st.info("No complaint data yet.")
        return

    badge_color = RISK_BADGE_COLOR.get(complaints["risk_badge"], TEXT_MUTED)
    st.markdown(
        pill_html(f"Complaint Risk: {complaints['risk_badge']}", badge_color, f"{badge_color}22", f"{badge_color}66"),
        unsafe_allow_html=True,
    )
    st.markdown("<div style='margin-top:12px;'></div>", unsafe_allow_html=True)

    st.metric(f"Total complaints — {bank['ticker']} (2011–2026)", f"{complaints['total']:,}")
    st.markdown(
        '<ul class="subtitle-list">'
        f"<li>{complaints['coverage_tier']}</li>"
        "</ul>",
        unsafe_allow_html=True,
    )

    if complaints["yoy_growth_pct"] is None:
        st.caption("Volume too thin for a reliable year-over-year trend or category breakdown.")
        return

    c1, c2 = st.columns(2)
    with c1:
        yoy = complaints["yoy_growth_pct"]
        st.markdown(stat_tile_html("YoY Complaint Growth", f"{yoy:+.0f}%", AMBER if yoy > 10 else TEXT_PRIMARY), unsafe_allow_html=True)
    with c2:
        st.markdown(stat_tile_html("Timely Response Rate", f"{complaints['timely_response_rate']}%", GREEN), unsafe_allow_html=True)

    st.markdown("<div style='margin-top:16px;'></div>", unsafe_allow_html=True)
    st.caption("Top complaint categories (% of total)")
    st.altair_chart(_category_bar_chart(complaints["top_categories"], AMBER), width="stretch", theme=None)

    st.caption("Monthly complaint volume, last 12 months")
    st.altair_chart(
        _monthly_trend_chart(MONTHLY_LABELS, complaints["monthly_trend"], AMBER), width="stretch", theme=None
    )

    if complaints["keywords"]:
        with st.expander("Distinctive complaint themes (TF-IDF)"):
            st.markdown(_keyword_bars_html(complaints["keywords"], AMBER), unsafe_allow_html=True)
            st.caption(
                "Number by each bar = TF-IDF distinctiveness score (0–1) for that term in this bank's complaint "
                "narratives — how much it sets this bank apart from peers, not a complaint count or percentage."
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

    label_color = {"negative": RED, "neutral": AMBER, "positive": GREEN}
    html_parts = []
    for p in filtered:
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
    with search_col:
        st.text_input(
            "Search",
            placeholder="Search a bank... (e.g. Wells Fargo)",
            disabled=True,
            label_visibility="collapsed",
        )

    st.divider()
    render_macro_banner()

    with st.container(key="bank-picker"):
        bank_names = list(MOCK_BANKS.keys())
        demo_name = st.pills("104 tracked · demo:", bank_names, default=bank_names[0])
    if not demo_name:
        demo_name = bank_names[0]
    bank = MOCK_BANKS[demo_name]

    render_header(bank)
    st.divider()

    with st.container(border=True, key="card-fundamentals"):
        render_fundamentals(bank)

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
    st.markdown('<div class="panel-title">SUPPLEMENTARY RISK SIGNALS</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="panel-caption">Price risk and customer complaint risk — additional context signals from '
        "the Ming sources EDA, not yet folded into the Stable/Watch/Elevated Risk badge above.</div>",
        unsafe_allow_html=True,
    )
    with st.expander("📈 Price Risk  ·  💳 Customer Complaint Risk"):
        col5, col6 = st.columns(2)
        with col5:
            with st.container(border=True, key="card-price", height="stretch"):
                render_price_risk(bank)
        with col6:
            with st.container(border=True, key="card-complaints", height="stretch"):
                render_complaints(bank)

    st.divider()
    st.caption(
        "How to read this screen — The status badge follows the project "
        "rubric: Stable · Watch · Elevated Risk · Imminent Disruption, from "
        "the fundamentals score (Gaussian Process classifier over Call "
        "Report features, min/max thresholds) crossed with news sentiment "
        "(BERT-based 3-class model trained on LLM-assisted labels). Key "
        "Alerts surfaces any metric outside its threshold plus non-ratio "
        "flags (e.g. open enforcement actions) — it isn't itself part of "
        "the composite score. Price risk and Customer Complaint Risk (added "
        "from the Ming sources EDA) are shown as supplementary context, not "
        "yet folded into the badge: price is dense (nearly all banks) and "
        "leading, complaints are consumer-bank-only and reactive — see "
        "coverage captions on each panel. Trend deltas and peer-percentile "
        "lines compare each bank against an illustrative US peer group of "
        "similar banks — real cohort definitions and computed percentiles "
        "are a scoring-methodology decision, not made here. Sources: GDELT "
        "DOC 2.0, SEC EDGAR (8-K/10-Q/10-K), FFIEC/FDIC unified dataset, "
        "Fed/FDIC enforcement actions, yfinance daily prices, CFPB Consumer "
        "Complaint Database, FRED macro series. All numbers on this page "
        "are illustrative — concept only, not model output."
    )


if __name__ == "__main__":
    main()
