import json
from pathlib import Path
import pandas as pd
import streamlit as st

st.set_page_config(page_title="MLII Challenger Dashboard", page_icon="🏆", layout="wide")

DATA_PATH = Path("leaderboard_master.csv")
CONFIG_PATH = Path("week_config.json")


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["week"] = df["week"].astype(str)
    df["higher_is_better"] = df["higher_is_better"].astype(str).str.lower().map(
        {"true": True, "false": False}
    )
    return df


@st.cache_data
def load_config():
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)


def render_week_tab(week, cfg, df):
    st.subheader(f"Week {week}: {cfg['title']}")

    st.markdown(f"**Challenge Focus:** {cfg['focus']}")
    st.markdown("**Topics:**")
    for t in cfg["topics"]:
        st.write(f"- {t}")

    dfw = df[df["week"] == week].copy()

    if dfw.empty:
        st.info("No submissions yet.")
        return

    dfw = dfw.sort_values("score", ascending=not cfg["higher_is_better"])
    dfw["rank"] = range(1, len(dfw) + 1)

    st.markdown("### 🏆 Leaderboard")
    st.dataframe(dfw[["rank", "student_name", "score", "model_family", "notes"]])

    st.markdown("### 📊 Benchmark Comparison")
    top_score = dfw.iloc[0]["score"]
    wa_score = cfg["wa_score"]

    col1, col2 = st.columns(2)
    col1.metric("Top Student Score", round(top_score, 4))
    col2.metric("WA Benchmark", wa_score)


def main():
    st.title("🏆 MLII Challenger Dashboard")
    st.caption("Weekly competition results")

    df = load_data()
    config = load_config()

    tabs = st.tabs([f"Week {w}" for w in config.keys()])

    for tab, week in zip(tabs, config.keys()):
        with tab:
            render_week_tab(week, config[week], df)


if __name__ == "__main__":
    main()