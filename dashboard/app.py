"""Bank Stability Monitor — Streamlit dashboard (Phase 5).

Mirrors the first-screen concept in dashboard/concept/ (mentor discussion,
2026-07-12). Backed by mock data for now — real reads against the index and
score tables land once Phase 2 (scoring) and Phase 3 (index) exist; see
db/migrations/011_scoring_tables.sql for the current shared contract.
"""

import streamlit as st

st.set_page_config(page_title="Bank Stability Monitor", layout="wide")

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
        "fundamentals": {
            "score": 84,
            "label": "Neutral",
            "features": [
                ("Tier 1 capital ratio %", 12.1, 11.8, "≥ 9.0", "within range"),
                ("Liquidity ratio %", 6.4, 5.9, "2.0 – 7.0", "within range"),
                ("NPL ratio %", 0.9, 1.2, "≤ 2.0", "within range"),
                ("Fee income / revenue %", 28.4, 26.1, "≥ 25.0", "near threshold"),
                ("CRE / total capital %", 64, 69, "≤ 120", "within range"),
            ],
        },
        "recent_items": [
            ("Regulator signals fresh penalties over consumer-billing practices", "negative", "reuters.com", "2d", "GDELT"),
            ("8-K — Item 8.01: settlement of outstanding consent order disclosed", "neutral", "sec.gov", "3d", "EDGAR"),
            ("Bank raises quarterly dividend, extends buyback", "positive", "marketwatch.com", "5d", "GDELT"),
            ("Analysts flag slower fee income amid asset-cap constraints", "negative", "ft.com", "6d", "GDELT"),
        ],
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
        "fundamentals": {
            "score": 76,
            "label": "Distress signal",
            "features": [
                ("Tier 1 capital ratio %", 10.4, 9.6, "≥ 9.0", "near threshold"),
                ("Liquidity ratio %", 3.1, 2.2, "2.0 – 7.0", "near threshold"),
                ("NPL ratio %", 1.6, 2.3, "≤ 2.0", "outside range"),
                ("Fee income / revenue %", 19.8, 18.9, "≥ 25.0", "outside range"),
                ("CRE / total capital %", 141, 148, "≤ 120", "outside range"),
            ],
        },
        "recent_items": [
            ("Ratings agency places regional lender on negative watch", "negative", "reuters.com", "8h", "GDELT"),
            ("8-K — Item 7.01: investor presentation on liquidity position", "neutral", "sec.gov", "1d", "EDGAR"),
            ("CRE concentration draws renewed analyst scrutiny", "negative", "barrons.com", "2d", "GDELT"),
            ("Completed capital raise shores up balance sheet", "positive", "marketwatch.com", "4d", "GDELT"),
        ],
    },
    # Placeholder demo entries — no concept mockup yet, kept minimal on purpose.
    "PNC Financial": {
        "name": "PNC Financial Services Group",
        "ticker": "PNC",
        "cert": "6384",
        "rssd": "817824",
        "status": "Stable",
        "summary": "Placeholder — no illustrative data drafted yet.",
        "sentiment": None,
        "keywords": None,
        "fundamentals": None,
        "recent_items": [],
    },
    "JPMorgan Chase": {
        "name": "JPMorgan Chase & Co.",
        "ticker": "JPM",
        "cert": "628",
        "rssd": "852218",
        "status": "Stable",
        "summary": "Placeholder — no illustrative data drafted yet.",
        "sentiment": None,
        "keywords": None,
        "fundamentals": None,
        "recent_items": [],
    },
}

STATUS_COLOR = {
    "Stable": "green",
    "Watch": "orange",
    "Elevated Risk": "red",
    "Imminent Disruption": "red",
}


def render_header(bank: dict) -> None:
    left, right = st.columns([4, 1])
    with left:
        st.subheader(bank["name"])
        st.caption(f"{bank['ticker']} · cert {bank['cert']} · rssd {bank['rssd']}")
        st.write(bank["summary"])
    with right:
        color = STATUS_COLOR.get(bank["status"], "gray")
        st.markdown(f":{color}[● **{bank['status']}**]")


def render_sentiment(bank: dict) -> None:
    st.markdown("**NEWS SENTIMENT — ROLLING 30 DAYS**")
    st.caption("GDELT news + EDGAR 8-K excerpts · FinBERT (fine-tuned), 3-class")
    sentiment = bank["sentiment"]
    if sentiment is None:
        st.info("No sentiment data yet.")
        return
    st.metric(f"{sentiment['label']} · {sentiment['n_items']} items scored", sentiment["score"])
    pct = sentiment["pct"]
    st.caption(
        f":red[Negative {pct['negative']}%]  ·  "
        f":orange[Neutral {pct['neutral']}%]  ·  "
        f":green[Positive {pct['positive']}%]"
    )
    st.line_chart(sentiment["trend"], height=180)


def render_keywords(bank: dict) -> None:
    st.markdown("**STANDOUT KEYWORDS — WHAT DRIVES THE SENTIMENT**")
    st.caption("Keyword clustering / PCA over scored articles (explainability)")
    keywords = bank["keywords"]
    if keywords is None:
        st.info("No keyword data yet.")
        return
    neg_col, pos_col = st.columns(2)
    with neg_col:
        st.markdown(":red[Negative drivers]")
        for word, weight in keywords["negative"]:
            st.progress(weight, text=f"{word} · {weight:.2f}")
    with pos_col:
        st.markdown(":green[Positive drivers]")
        for word, weight in keywords["positive"]:
            st.progress(weight, text=f"{word} · {weight:.2f}")


def render_fundamentals(bank: dict) -> None:
    st.markdown("**FUNDAMENTALS RISK PROFILE — QUARTER OVER QUARTER**")
    st.caption(
        "FFIEC Call Report features · Gaussian Process classifier · "
        "bands ≤80 distress / 80–90 neutral / ≥90 sound"
    )
    fundamentals = bank["fundamentals"]
    if fundamentals is None:
        st.info("No fundamentals data yet.")
        return
    st.metric(fundamentals["label"], fundamentals["score"])
    st.table(
        {
            "Feature": [f[0] for f in fundamentals["features"]],
            "Prior Q": [f[1] for f in fundamentals["features"]],
            "Latest Q": [f[2] for f in fundamentals["features"]],
            "Threshold": [f[3] for f in fundamentals["features"]],
            "Status": [f[4] for f in fundamentals["features"]],
        }
    )


def render_recent_items(bank: dict) -> None:
    st.markdown("**RECENT ITEMS**")
    st.caption("Latest scored articles & filings for this bank")
    items = bank["recent_items"]
    if not items:
        st.info("No recent items yet.")
        return
    label_color = {"negative": "red", "neutral": "orange", "positive": "green"}
    for title, label, source, age, feed in items:
        color = label_color.get(label, "gray")
        st.markdown(f"**{title}**  :{color}[{label}]")
        st.caption(f"{source} · {age} · {feed}")
        st.divider()


def main() -> None:
    st.title("Bank Stability Monitor")
    st.caption("PNC Capstone · early-warning dashboard")
    st.warning("Concept mockup · illustrative data", icon="⚠️")

    st.text_input("Search a bank...", placeholder="e.g. Wells Fargo", disabled=True)
    demo_name = st.radio(
        "104 tracked · demo:",
        list(MOCK_BANKS.keys()),
        horizontal=True,
    )
    bank = MOCK_BANKS[demo_name]

    render_header(bank)
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        render_sentiment(bank)
    with col2:
        render_keywords(bank)

    st.divider()

    col3, col4 = st.columns([2, 1])
    with col3:
        render_fundamentals(bank)
    with col4:
        render_recent_items(bank)

    st.divider()
    st.caption(
        "How to read this screen — Combined classification follows the "
        "project rubric: Stable · Watch · Elevated Risk · Imminent "
        "Disruption, from the fundamentals score (Gaussian Process "
        "classifier over Call Report features, min/max thresholds) crossed "
        "with news sentiment (BERT-based 3-class model trained on "
        "LLM-assisted labels). Sources: GDELT DOC 2.0, SEC EDGAR "
        "(8-K/10-Q/10-K), FFIEC/FDIC unified dataset. All numbers on this "
        "page are illustrative — concept only, not model output."
    )


if __name__ == "__main__":
    main()
