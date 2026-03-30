import json
import base64
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="MLII Challenger Dashboard",
    page_icon="🏆",
    layout="wide"
)

DATA_PATH = Path("leaderboard_master.csv")
CONFIG_PATH = Path("week_config.json")
LOGO_PATH = Path("logo.svg")


def get_logo_base64(path: Path):
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Lora:wght@600;700&family=Montserrat:wght@400;500;600;700&display=swap');

    :root {
        --uchicago-maroon: #800000;
        --uchicago-brick: #A4343A;
        --soft-bg: #F6F3F1;
        --card-bg: #FFFFFF;
        --border: #E5DED8;
        --text-main: #1C1C1C;
        --text-soft: #667085;
    }

    html, body, [class*="css"] {
        font-family: 'Montserrat', sans-serif;
        color: var(--text-main);
    }

    .stApp {
        background-color: var(--soft-bg);
    }

    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 2rem;
        max-width: 1160px;
    }

    h1, h2, h3 {
        font-family: 'Lora', serif;
        color: var(--uchicago-maroon);
        letter-spacing: -0.02em;
        margin-bottom: 0.4rem;
    }

    .hero-wrap {
        display: flex;
        align-items: center;
        gap: 20px;
        margin-bottom: 0.15rem;
    }

    .hero-logo {
        width: 200px;
        height: 200px;
        object-fit: contain;
    }

    .hero-title {
        font-family: 'Lora', serif;
        font-size: 2.45rem;
        font-weight: 700;
        color: var(--uchicago-maroon);
        line-height: 1.02;
        margin: 0;
    }

    .hero-subtitle {
        color: var(--text-soft);
        font-size: 0.95rem;
        margin-top: 0.15rem;
        margin-bottom: 1rem;
    }

    .summary-strip {
        background: linear-gradient(90deg, rgba(128,0,0,0.05) 0%, rgba(164,52,58,0.035) 100%);
        padding: 1rem 1.15rem;
        border-radius: 18px;
        border: 1px solid var(--border);
        margin-bottom: 1rem;
    }

    .mini-kicker {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--uchicago-brick);
        font-weight: 700;
        margin-bottom: 0.3rem;
    }

    .focus-line {
        font-size: 1.02rem;
        font-weight: 600;
        color: var(--text-main);
        margin-bottom: 0.35rem;
    }

    .topic-line {
        font-size: 0.93rem;
        color: var(--text-soft);
    }

    .section-title {
        font-family: 'Lora', serif;
        font-size: 1.45rem;
        font-weight: 700;
        color: var(--uchicago-maroon);
        margin-top: 0.7rem;
        margin-bottom: 0.65rem;
    }

    .podium-card {
        background: var(--card-bg);
        border: 1px solid var(--border);
        border-radius: 20px;
        padding: 1.1rem 1rem 1rem 1rem;
        text-align: center;
        box-shadow: 0 8px 20px rgba(0,0,0,0.045);
        min-height: 200px;
    }

    .podium-gold {
        border-top: 6px solid #D4A017;
    }

    .podium-silver {
        border-top: 6px solid #A8A8A8;
    }

    .podium-bronze {
        border-top: 6px solid #A97142;
    }

    .podium-medal {
        font-size: 1.9rem;
        margin-bottom: 0.3rem;
    }

    .podium-place {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--text-soft);
        margin-bottom: 0.45rem;
        font-weight: 700;
    }

    .podium-name {
        font-size: 1.18rem;
        font-weight: 700;
        color: var(--text-main);
        margin-bottom: 0.3rem;
    }

    .podium-score-label {
        font-size: 0.82rem;
        color: var(--text-soft);
        margin-top: 0.3rem;
    }

    .podium-score {
        font-size: 1.85rem;
        font-weight: 800;
        color: var(--uchicago-brick);
        line-height: 1.1;
        margin-top: 0.08rem;
    }

    div[data-testid="stMetric"] {
        background: var(--card-bg);
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 0.9rem 1rem 0.8rem 1rem;
        box-shadow: 0 6px 18px rgba(0,0,0,0.04);
    }

    [data-testid="stMetricLabel"] {
        color: var(--text-soft);
        font-weight: 600;
    }

    [data-testid="stMetricValue"] {
        color: var(--uchicago-maroon);
    }

    .gap-caption {
        color: var(--text-soft);
        font-size: 0.93rem;
        margin-top: 0.5rem;
        margin-bottom: 0.4rem;
        text-align: center;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid var(--border);
        border-radius: 18px;
        overflow: hidden;
        box-shadow: 0 6px 18px rgba(0,0,0,0.04);
        background: white;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }

    .stTabs [data-baseweb="tab"] {
        color: #6b7280;
        font-weight: 600;
    }

    .stTabs [aria-selected="true"] {
        color: var(--uchicago-maroon) !important;
        border-bottom-color: var(--uchicago-maroon) !important;
    }
</style>
"""


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)

    required_cols = {
        "week",
        "student_name",
        "score",
        "metric_name",
        "higher_is_better",
        "model_family",
        "status",
        "notes",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns in leaderboard_master.csv: {sorted(missing)}"
        )

    df["week"] = df["week"].astype(str)

    df["higher_is_better"] = (
        df["higher_is_better"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False})
    )

    if df["higher_is_better"].isna().any():
        raise ValueError("Column 'higher_is_better' must contain only True/False values.")

    df["student_name"] = (
        df["student_name"]
        .astype(str)
        .str.strip()
        .str.title()
    )

    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df["status"] = df["status"].fillna("Valid").astype(str).str.strip()
    df["model_family"] = df["model_family"].fillna("").astype(str).str.strip()
    df["notes"] = df["notes"].fillna("").astype(str).str.strip()
    df["metric_name"] = df["metric_name"].fillna("").astype(str).str.strip()

    return df


@st.cache_data
def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def sort_week_df(dfw: pd.DataFrame, higher_is_better: bool) -> pd.DataFrame:
    valid_df = dfw[dfw["status"].str.lower() == "valid"].copy()
    invalid_df = dfw[dfw["status"].str.lower() != "valid"].copy()

    valid_df = valid_df.sort_values("score", ascending=not higher_is_better)
    invalid_df = invalid_df.sort_values("student_name", ascending=True)

    return pd.concat([valid_df, invalid_df], ignore_index=True)


def format_score(value):
    if pd.isna(value):
        return "N/A"
    return f"{value:.6f}"


def render_header():
    logo_b64 = get_logo_base64(LOGO_PATH)
    if logo_b64:
        st.markdown(
            f"""
            <div class="hero-wrap">
                <img class="hero-logo" src="data:image/svg+xml;base64,{logo_b64}" />
                <div>
                    <div class="hero-title">MLII Challenger Dashboard</div>
                </div>
            </div>
            <div class="hero-subtitle">Weekly competition results</div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.title("🏆 MLII Challenger Dashboard")
        st.caption("Weekly competition results")


def render_podium(valid_df: pd.DataFrame, metric_name: str):
    st.markdown('<div class="section-title">Top Performers</div>', unsafe_allow_html=True)

    top3 = valid_df.head(3).copy()
    medal_classes = ["podium-gold", "podium-silver", "podium-bronze"]
    medals = ["🥇", "🥈", "🥉"]
    places = ["1st Place", "2nd Place", "3rd Place"]

    cols = st.columns(3)
    for i in range(3):
        with cols[i]:
            if i < len(top3):
                row = top3.iloc[i]
                st.markdown(
                    f"""
                    <div class="podium-card {medal_classes[i]}">
                        <div class="podium-medal">{medals[i]}</div>
                        <div class="podium-place">{places[i]}</div>
                        <div class="podium-name">{row["student_name"]}</div>
                        <div class="podium-score-label">{metric_name}</div>
                        <div class="podium-score">{row["score"]:.6f}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )


def render_week_tab(week, cfg, df):
    st.markdown(f'<div class="section-title">Week {week}: {cfg["title"]}</div>', unsafe_allow_html=True)

    topics_str = " • ".join(cfg["topics"])
    st.markdown(
        f"""
        <div class="summary-strip">
            <div class="mini-kicker">Challenge Focus</div>
            <div class="focus-line">{cfg['focus']}</div>
            <div class="topic-line"><strong>Topics:</strong> {topics_str}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    dfw = df[df["week"] == str(week)].copy()

    if dfw.empty:
        st.info("No submissions yet.")
        return

    higher_is_better = bool(cfg["higher_is_better"])
    metric_name = cfg.get("metric_name", "Score")
    benchmark_score = cfg.get("wa_score", None)
    benchmark_label = cfg.get("benchmark_label", "Benchmark")

    dfw = sort_week_df(dfw, higher_is_better)
    dfw["rank"] = range(1, len(dfw) + 1)

    valid_df = dfw[dfw["status"].str.lower() == "valid"].copy()

    if not valid_df.empty:
        render_podium(valid_df, metric_name)

    st.markdown('<div class="section-title">Current Cohort vs Historical Best</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)

    if not valid_df.empty:
        top_score = valid_df.iloc[0]["score"]
        top_student = valid_df.iloc[0]["student_name"]

        c1.metric(f"Top {metric_name}", f"{top_score:.4f}")
        c2.metric("Current Leader", top_student)

        if benchmark_score is None:
            c3.metric(benchmark_label, "N/A")
        else:
            c3.metric(benchmark_label, f"{benchmark_score:.4f}")
            gap = top_score - benchmark_score if higher_is_better else benchmark_score - top_score
            st.markdown(
                f'<div class="gap-caption">Gap to {benchmark_label}: {gap:.6f}</div>',
                unsafe_allow_html=True
            )
            st.markdown(
                f'''
                <div class="gap-caption" style="margin-top:6px;">
                    Historical Best held by 🏆 <strong>Mo Kahn</strong>
                </div>
                ''',  # noqa: F541
                unsafe_allow_html=True
            )

    st.markdown('<div class="section-title">Full Leaderboard</div>', unsafe_allow_html=True)

    display_df = dfw.copy()
    display_df["score"] = display_df["score"].apply(format_score)
    display_df = display_df[["rank", "student_name", "score", "status"]]
    display_df.columns = ["Rank", "Student", "Accuracy", "Status"]
    st.dataframe(display_df, use_container_width=True, hide_index=True)


def main():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    df = load_data()
    config = load_config()

    render_header()

    weeks = list(config.keys())
    tabs = st.tabs([f"Week {week}" for week in weeks])

    for tab, week in zip(tabs, weeks):
        with tab:
            render_week_tab(week, config[week], df)


if __name__ == "__main__":
    main()