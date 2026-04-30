import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from orchestrator.router import route

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="FPL-Intel PRO", layout="wide")
st.title("⚽ FPL-Intel PRO: Premier League AI Platform")

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    url = "https://fantasy.premierleague.com/api/bootstrap-static/"
    return requests.get(url).json()

@st.cache_data
def get_player_history(player_id):
    url = f"https://fantasy.premierleague.com/api/element-summary/{player_id}/"
    data = requests.get(url).json()
    return pd.DataFrame(data["history"])

data = load_data()

players = pd.DataFrame(data["elements"])
teams = {t["id"]: t["name"] for t in data["teams"]}

players["team_name"] = players["team"].map(teams)
players["name"] = players["web_name"]

# -----------------------------
# TABS
# -----------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "💬 Chat",
    "🔍 Search",
    "⚔️ Compare",
    "📊 Dashboard"
])

# =====================================================
# 💬 CHAT
# =====================================================
with tab1:
    st.header("AI Assistant")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                st.markdown(msg["content"])
            else:
                st.write(msg["content"])

    user_input = st.chat_input("Ask about players, injuries, predictions...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.write(user_input)

        try:
            result = route(user_input, chat_history=st.session_state.messages[:-1])
        except Exception as e:
            result = {
                "player": "Error",
                "predicted_points": 0.0,
                "news": "Error connecting to pipeline",
                "advice": str(e),
                "_type": "error",
            }

        with st.chat_message("assistant"):
            if "ml" in result and "rag" in result:
                # Combined response — show both side-by-side
                ml_res = result["ml"]
                rag_res = result["rag"]
                col_ml, col_rag = st.columns(2)
                with col_ml:
                    st.subheader(f"📊 {ml_res.get('player', user_input)}")
                    if ml_res.get("predicted_points") is not None:
                        st.metric("Predicted Points", ml_res["predicted_points"])
                    if ml_res.get("advice"):
                        st.success(ml_res["advice"])
                with col_rag:
                    st.subheader("📰 Latest News & Analysis")
                    st.markdown(rag_res.get("news", ""))
                display_text = (
                    f"**ML Prediction** — {ml_res.get('player', user_input)}: "
                    f"{ml_res.get('predicted_points', '?')} pts\n\n"
                    f"**Analysis**\n{rag_res.get('news', '')}"
                )
            elif result.get("_type") == "rag" or (
                result.get("predicted_points") is None and result.get("news")
            ):
                # Pure RAG answer — render as markdown
                st.markdown(result["news"])
                display_text = result["news"]
            else:
                # ML-only answer — structured card
                st.subheader(result.get("player", user_input))
                if result.get("predicted_points") is not None:
                    st.metric("Predicted Points", result["predicted_points"])
                if result.get("news"):
                    st.info(result["news"])
                if result.get("advice"):
                    st.success(result["advice"])
                display_text = (
                    f"**{result.get('player', user_input)}** — "
                    f"{result.get('predicted_points', '?')} pts\n"
                    f"{result.get('advice', '')}"
                )

        st.session_state.messages.append({
            "role": "assistant",
            "content": display_text,
        })

# =====================================================
# 🔍 SEARCH
# =====================================================
with tab2:
    st.header("Search Players / Teams")

    query = st.text_input("Search")

    if query:
        results = players[
            players["name"].str.lower().str.contains(query.lower()) |
            players["team_name"].str.lower().str.contains(query.lower())
        ]

        for _, p in results.head(10).iterrows():
            st.subheader(f"{p['name']} ({p['team_name']})")

            c1, c2, c3 = st.columns(3)
            c1.metric("Goals", p["goals_scored"])
            c2.metric("Assists", p["assists"])
            c3.metric("Points", p["total_points"])

            st.progress(float(p["form"]) / 10 if p["form"] else 0)
            st.divider()

# =====================================================
# ⚔️ COMPARE
# =====================================================
with tab3:
    st.header("Compare Players")

    names = players["name"].unique()

    p1_name = st.selectbox("Player 1", names)
    p2_name = st.selectbox("Player 2", names)

    if p1_name and p2_name:
        p1 = players[players["name"] == p1_name].iloc[0]
        p2 = players[players["name"] == p2_name].iloc[0]

        df = pd.DataFrame({
            "Metric": ["Goals", "Assists", "Points", "Form"],
            p1_name: [p1["goals_scored"], p1["assists"], p1["total_points"], float(p1["form"])],
            p2_name: [p2["goals_scored"], p2["assists"], p2["total_points"], float(p2["form"])]
        })

        st.table(df)

        fig = px.bar(df, x="Metric", y=[p1_name, p2_name], barmode="group",
                     title="Player Comparison")
        st.plotly_chart(fig, use_container_width=True)

# =====================================================
# 📊 PRO DASHBOARD
# =====================================================
with tab4:
    st.header("📊 Player Analytics Dashboard")

    selected = st.selectbox("Select Player", players["name"].unique())

    if selected:
        player = players[players["name"] == selected].iloc[0]
        history = get_player_history(player["id"])

        if not history.empty:
            st.subheader(f"{selected} Performance Analysis")

            # -----------------------------
            # Points Over Time
            # -----------------------------
            fig1 = px.line(
                history,
                x="round",
                y="total_points",
                title="Points per Gameweek",
                markers=True
            )
            st.plotly_chart(fig1, use_container_width=True)

            # -----------------------------
            # Goals & Assists
            # -----------------------------
            fig2 = px.line(
                history,
                x="round",
                y=["goals_scored", "assists"],
                title="Goals & Assists Trend"
            )
            st.plotly_chart(fig2, use_container_width=True)

            # -----------------------------
            # Minutes Played
            # -----------------------------
            fig3 = px.bar(
                history,
                x="round",
                y="minutes",
                title="Minutes Played"
            )
            st.plotly_chart(fig3, use_container_width=True)

            # -----------------------------
            # Form (Moving Average)
            # -----------------------------
            history["form"] = history["total_points"].rolling(3).mean()

            fig4 = px.line(
                history,
                x="round",
                y="form",
                title="Form Trend (3 GW Avg)"
            )
            st.plotly_chart(fig4, use_container_width=True)

            # -----------------------------
            # Best Performance
            # -----------------------------
            best = history.loc[history["total_points"].idxmax()]
            st.success(f"🔥 Best Gameweek: GW {best['round']} ({best['total_points']} pts)")

            # -----------------------------
            # Summary Stats
            # -----------------------------
            col1, col2, col3 = st.columns(3)
            col1.metric("Avg Points", round(history["total_points"].mean(), 2))
            col2.metric("Total Goals", history["goals_scored"].sum())
            col3.metric("Total Assists", history["assists"].sum())

        else:
            st.warning("No data available")