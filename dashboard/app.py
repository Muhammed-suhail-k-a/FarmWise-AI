
import os
import re
import json

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ================================================================
# PAGE CONFIG
# ================================================================

st.set_page_config(
    page_title="FarmWise AI",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ================================================================
# PATHS (unchanged)
# ================================================================

BASE_DIR = "/content/drive/MyDrive/suhail/project/FarmWise"

DATA_PATH = os.path.join(BASE_DIR, "india_rainfall_crop_yield_2014_2024.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "farmwise_yieldwise_gradient_boosting.joblib")
REPORT_PATH = os.path.join(BASE_DIR, "outputs", "yieldwise_final_model_report.json")
PREDICTIONS_PATH = os.path.join(BASE_DIR, "outputs", "yieldwise_test_predictions.csv")
FAISS_PATH = os.path.join(BASE_DIR, "rag", "vector_store", "rice_pest_faiss.index")
METADATA_PATH = os.path.join(BASE_DIR, "rag", "vector_store", "rice_pest_metadata.json")

# ================================================================
# THEME
# ================================================================

PALETTE = ["#2F7D3B", "#E0A526", "#4FA3A5", "#8C6A4A", "#7BB661", "#C4543B", "#5B6DB5", "#A7B84B"]
GREENS = ["#E4F0DD", "#B9DAA8", "#7BB661", "#2F7D3B", "#1B4D2B"]

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:wght@500;600;700&family=Manrope:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Manrope', sans-serif; }
h1, h2, h3, h4 { font-family: 'Bricolage Grotesque', sans-serif !important; color: #1D2B20; letter-spacing: -0.01em; }

.stApp { background: #F3F7F0; }
.block-container { padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1400px; }
#MainMenu, footer { visibility: hidden; }

/* ---------- sidebar ---------- */
section[data-testid="stSidebar"] { background: #15391F; border-right: 1px solid #0E2A16; }
section[data-testid="stSidebar"] * { color: #E6F0E1 !important; }
section[data-testid="stSidebar"] div[role="radiogroup"] { gap: 0.25rem; }
section[data-testid="stSidebar"] div[role="radiogroup"] label {
    background: rgba(255,255,255,0.05); border-radius: 10px;
    padding: 0.55rem 0.8rem; width: 100%; transition: background 0.15s;
}
section[data-testid="stSidebar"] div[role="radiogroup"] label:hover { background: rgba(255,255,255,0.13); }
section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) { background: #2F7D3B; }
section[data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child { display: none; }
.brand { display: flex; gap: 0.7rem; align-items: center; margin-bottom: 0.2rem; }
.brand-mark { font-size: 2rem; background: #E0A526; border-radius: 12px; width: 46px; height: 46px;
              display: flex; align-items: center; justify-content: center; }
.brand-name { font-family: 'Bricolage Grotesque', sans-serif; font-size: 1.35rem; font-weight: 700; line-height: 1.1; }
.brand-tag { font-size: 0.75rem; opacity: 0.75; }
.sdg-box { background: rgba(255,255,255,0.06); border-radius: 12px; padding: 0.8rem 0.9rem; font-size: 0.82rem; line-height: 1.7; }

/* ---------- header banners ---------- */
.hero { background: linear-gradient(115deg, #15391F 0%, #2F7D3B 62%, #7BB661 100%);
        border-radius: 20px; padding: 2rem 2.2rem; color: #fff; margin-bottom: 1.2rem; }
.hero h1 { color: #fff !important; margin: 0 0 0.4rem 0; font-size: 2.2rem; }
.hero p { margin: 0; opacity: 0.92; font-size: 1.02rem; max-width: 780px; }
.page-head { background: #fff; border: 1px solid #DCE8D6; border-left: 6px solid #2F7D3B;
             border-radius: 16px; padding: 1.1rem 1.4rem; margin-bottom: 1.1rem; }
.page-head h2 { margin: 0; font-size: 1.55rem; }
.page-head p { margin: 0.25rem 0 0 0; color: #5A6B5D; font-size: 0.95rem; }

/* ---------- KPI cards ---------- */
.kpi { background: #fff; border: 1px solid #DCE8D6; border-radius: 16px; padding: 0.9rem 1.1rem;
       display: flex; gap: 0.9rem; align-items: center; margin-bottom: 0.9rem; min-height: 92px; }
.kpi-icon { font-size: 1.6rem; background: #EAF3E4; border-radius: 12px; width: 50px; height: 50px;
            display: flex; align-items: center; justify-content: center; flex-shrink: 0; }
.kpi-label { font-size: 0.82rem; color: #5A6B5D; }
.kpi-value { font-family: 'Bricolage Grotesque', sans-serif; font-size: 1.45rem; font-weight: 700; color: #1D2B20; line-height: 1.2; }
.kpi-sub { font-size: 0.76rem; color: #7C8C7F; }
.kpi.gold { border-left: 5px solid #E0A526; }
.kpi.green { border-left: 5px solid #2F7D3B; }
.kpi.teal { border-left: 5px solid #4FA3A5; }
.kpi.brown { border-left: 5px solid #8C6A4A; }
.kpi.red { border-left: 5px solid #C4543B; }

/* ---------- section titles, notes, flow ---------- */
.sec-title { font-family: 'Bricolage Grotesque', sans-serif; font-size: 1.2rem; font-weight: 600; color: #15391F; margin: 1.3rem 0 0.15rem 0; }
.sec-sub { color: #6A7B6D; font-size: 0.88rem; margin-bottom: 0.6rem; }
.note { border-radius: 12px; padding: 0.8rem 1rem; font-size: 0.9rem; line-height: 1.5; margin: 0.8rem 0; }
.note.warn { background: #FFF6DC; border-left: 5px solid #E0A526; color: #5B4600; }
.note.info { background: #E6F2E2; border-left: 5px solid #2F7D3B; color: #1C4A25; }
.note.danger { background: #FCE8E3; border-left: 5px solid #C4543B; color: #6B2415; }
.flow { display: flex; flex-wrap: wrap; align-items: center; gap: 0.5rem; margin: 0.6rem 0 0.4rem 0; }
.flow-step { background: #fff; border: 1px solid #C9DEC0; border-radius: 12px; padding: 0.55rem 1rem;
             font-weight: 600; color: #15391F; }
.flow-arrow { color: #7BB661; font-size: 1.3rem; }
.pipe { display: flex; flex-wrap: wrap; gap: 0.5rem; align-items: stretch; margin: 0.6rem 0; }
.pipe-step { background: #fff; border: 1px solid #C9DEC0; border-radius: 12px; padding: 0.7rem 0.9rem;
             flex: 1 1 130px; font-size: 0.85rem; }
.pipe-step b { display: block; color: #2F7D3B; margin-bottom: 0.15rem; }

/* ---------- tabs / widgets ---------- */
button[data-baseweb="tab"] { font-weight: 600; font-size: 0.95rem; }
div[data-testid="stExpander"] { background: #fff; border-radius: 12px; border: 1px solid #DCE8D6; }
div[data-testid="stChatMessage"] { background: #fff; border: 1px solid #DCE8D6; border-radius: 14px; }
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)

# ================================================================
# UI HELPERS
# ================================================================


def money(value):
    return f"₹{value:,.0f}"


def hero(title, description):
    st.markdown(
        f'<div class="hero"><h1>{title}</h1><p>{description}</p></div>',
        unsafe_allow_html=True,
    )


def page_header(title, description):
    st.markdown(
        f'<div class="page-head"><h2>{title}</h2><p>{description}</p></div>',
        unsafe_allow_html=True,
    )


def section(title, sub=""):
    st.markdown(f'<div class="sec-title">{title}</div>', unsafe_allow_html=True)
    if sub:
        st.markdown(f'<div class="sec-sub">{sub}</div>', unsafe_allow_html=True)


def note(text, kind="warn"):
    st.markdown(f'<div class="note {kind}">{text}</div>', unsafe_allow_html=True)


def kpi(col, icon, label, value, sub="", tone="green"):
    col.markdown(
        f'<div class="kpi {tone}"><div class="kpi-icon">{icon}</div><div>'
        f'<div class="kpi-label">{label}</div><div class="kpi-value">{value}</div>'
        f'<div class="kpi-sub">{sub}</div></div></div>',
        unsafe_allow_html=True,
    )


def show_df(d, height=None):
    kwargs = {"height": height} if height else {}
    try:
        st.dataframe(d, width="stretch", **kwargs)
    except TypeError:
        st.dataframe(d, use_container_width=True, **kwargs)


def show_fig(fig):
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:
        st.plotly_chart(fig, use_container_width=True)


def style_fig(fig, height=340, legend=False):
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=10, r=10, t=30, b=10),
        font=dict(family="Manrope, sans-serif", size=13, color="#1D2B20"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=legend,
        colorway=PALETTE,
        coloraxis_showscale=False,
    )
    fig.update_xaxes(gridcolor="#E5EDE0", zeroline=False)
    fig.update_yaxes(gridcolor="#E5EDE0", zeroline=False)
    return fig


def hbar(series, xlabel="Value", height=340, fmt=None):
    d = series.rename_axis("Category").reset_index(name="Value").sort_values("Value")
    fig = px.bar(d, x="Value", y="Category", orientation="h", color="Value",
                 color_continuous_scale=GREENS)
    fig.update_layout(xaxis_title=xlabel, yaxis_title="")
    if fmt:
        fig.update_xaxes(tickformat=fmt)
    return style_fig(fig, height)


def vbar(series, ylabel="Value", height=340):
    d = series.rename_axis("Category").reset_index(name="Value")
    fig = px.bar(d, x="Category", y="Value", color="Value", color_continuous_scale=GREENS)
    fig.update_layout(xaxis_title="", yaxis_title=ylabel)
    return style_fig(fig, height)


def line(series, ylabel="Value", height=340):
    d = series.rename_axis("X").reset_index(name="Y")
    fig = px.line(d, x="X", y="Y", markers=True)
    fig.update_traces(line=dict(color="#2F7D3B", width=3), marker=dict(size=8, color="#E0A526"))
    fig.update_layout(xaxis_title="", yaxis_title=ylabel)
    fig.update_xaxes(dtick=1)
    return style_fig(fig, height)


# ================================================================
# DATA
# ================================================================


@st.cache_data
def load_data():
    if not os.path.exists(DATA_PATH):
        return None
    return pd.read_csv(DATA_PATH)


@st.cache_data
def enrich(frame):
    d = frame.copy()
    area = d["Cultivated_Area_hectares"].replace(0, np.nan)
    fert = d["Fertilizer_Used_kg"].replace(0, np.nan)
    pest = d["Pesticide_Used_kg"].replace(0, np.nan)

    d["Estimated_Production_tons"] = d["Cultivated_Area_hectares"] * d["Yield_ton_per_hectare"]
    prod = d["Estimated_Production_tons"].replace(0, np.nan)
    d["Estimated_Revenue_INR"] = d["Estimated_Production_tons"] * d["Market_Price_per_ton"]
    rev = d["Estimated_Revenue_INR"].replace(0, np.nan)

    d["Yield_per_Fertilizer_kg"] = d["Yield_ton_per_hectare"] / fert
    d["Yield_per_Pesticide_kg"] = d["Yield_ton_per_hectare"] / pest
    d["Profit_per_hectare_INR"] = d["Farmer_Profit_INR"] / area
    d["Profit_per_ton_INR"] = d["Farmer_Profit_INR"] / prod
    d["Profit_to_Revenue_Ratio"] = d["Farmer_Profit_INR"] / rev
    d["Revenue_per_hectare_INR"] = d["Estimated_Revenue_INR"] / area
    return d


df = load_data()

if df is None:
    st.error("FarmWise dataset was not found. Check the Google Drive path in BASE_DIR.")
    st.stop()

data_all = enrich(df)

# ================================================================
# SIDEBAR
# ================================================================

st.sidebar.markdown(
    '<div class="brand"><div class="brand-mark">🌾</div><div>'
    '<div class="brand-name">FarmWise AI</div>'
    '<div class="brand-tag">Sustainable farm decision support</div></div></div>',
    unsafe_allow_html=True,
)
st.sidebar.markdown("&nbsp;", unsafe_allow_html=True)

PAGES = [
    "🏠 Overview",
    "🌾 Farm Analysis",
    "💧 ResourceWise",
    "💰 CostWise",
    "🌦️ ClimateWise",
    "📈 YieldWise",
    "🔄 What-If Simulator",
    "🐛 PestRisk / EcoAdvisor",
]

page = st.sidebar.radio("Navigation", PAGES, label_visibility="collapsed")

st.sidebar.markdown("&nbsp;", unsafe_allow_html=True)
st.sidebar.markdown(
    '<div class="sdg-box"><b>Sustainable Development Goals</b><br>'
    "🌾 SDG 2: Zero Hunger<br>💧 SDG 6: Clean Water<br>"
    "♻️ SDG 12: Responsible Consumption<br>🌍 SDG 13: Climate Action</div>",
    unsafe_allow_html=True,
)
st.sidebar.caption("FarmWise AI: decision-support prototype")


# ================================================================
# OVERVIEW
# ================================================================

if page == "🏠 Overview":

    hero(
        "🌾 FarmWise AI",
        "From farm data to sustainable decisions. FarmWise combines agricultural, "
        "climate, resource and economic records to support data-driven farm choices.",
    )

    st.markdown(
        '<div class="flow">'
        '<div class="flow-step">📊 Predict</div><div class="flow-arrow">➜</div>'
        '<div class="flow-step">🔍 Analyze</div><div class="flow-arrow">➜</div>'
        '<div class="flow-step">⚙️ Optimize</div><div class="flow-arrow">➜</div>'
        '<div class="flow-step">🔄 Simulate</div><div class="flow-arrow">➜</div>'
        '<div class="flow-step">💡 Recommend</div></div>',
        unsafe_allow_html=True,
    )

    section("Dataset at a glance")

    c1, c2, c3, c4 = st.columns(4)
    kpi(c1, "📋", "Records", f"{len(df):,}", f"{int(df['Year'].min())} to {int(df['Year'].max())}")
    kpi(c2, "🗺️", "States", f"{df['State'].nunique()}", "", "teal")
    kpi(c3, "🏘️", "Districts", f"{df['District'].nunique()}", "", "gold")
    kpi(c4, "🌾", "Crop types", f"{df['Crop_Type'].nunique()}", "", "brown")

    c5, c6, c7, c8 = st.columns(4)
    kpi(c5, "📈", "Average yield", f"{df['Yield_ton_per_hectare'].mean():.2f} t/ha")
    kpi(c6, "🌧️", "Average rainfall", f"{df['Rainfall_mm'].mean():.1f} mm", "", "teal")
    kpi(c7, "🧪", "Average fertilizer", f"{df['Fertilizer_Used_kg'].mean():.1f} kg", "", "gold")
    kpi(c8, "💰", "Average farmer profit", money(df["Farmer_Profit_INR"].mean()), "", "brown")

    left, right = st.columns(2)
    with left:
        section("Crop distribution", "Number of records per crop")
        show_fig(hbar(df["Crop_Type"].value_counts(), "Records"))
    with right:
        section("State distribution", "Number of records per state")
        show_fig(hbar(df["State"].value_counts(), "Records"))

    section("Average yield by year", "Mean yield across all records for each year")
    show_fig(line(df.groupby("Year")["Yield_ton_per_hectare"].mean(), "Yield (t/ha)", 320))

    with st.expander("🔍 Dataset preview (first 20 rows)"):
        show_df(df.head(20))

    note(
        "<b>Responsible AI note.</b> FarmWise AI is a decision-support prototype. "
        "Indicators and model outputs are not guaranteed agronomic outcomes or "
        "definitive agricultural prescriptions."
    )


# ================================================================
# FARM ANALYSIS
# ================================================================

elif page == "🌾 Farm Analysis":

    page_header("🌾 Farm Analysis", "Explore agricultural patterns by state, crop and season")

    with st.container(border=True):
        f1, f2, f3 = st.columns(3)
        state = f1.selectbox("State", ["All"] + sorted(df["State"].unique().tolist()))
        crop = f2.selectbox("Crop", ["All"] + sorted(df["Crop_Type"].unique().tolist()))
        season = f3.selectbox("Season", ["All"] + sorted(df["Season"].unique().tolist()))

    filtered = data_all.copy()
    if state != "All":
        filtered = filtered[filtered["State"] == state]
    if crop != "All":
        filtered = filtered[filtered["Crop_Type"] == crop]
    if season != "All":
        filtered = filtered[filtered["Season"] == season]

    if len(filtered) == 0:
        note("No records match these filters. Try widening the selection.", "warn")
    else:
        c1, c2, c3, c4 = st.columns(4)
        kpi(c1, "📋", "Matching records", f"{len(filtered):,}")
        kpi(c2, "📈", "Average yield", f"{filtered['Yield_ton_per_hectare'].mean():.2f} t/ha", "", "gold")
        kpi(c3, "🌧️", "Average rainfall", f"{filtered['Rainfall_mm'].mean():.1f} mm", "", "teal")
        kpi(c4, "💰", "Average profit", money(filtered["Farmer_Profit_INR"].mean()), "", "brown")

        tab1, tab2, tab3 = st.tabs(["📈 Yield trend", "🌾 Crop performance", "🔍 Records"])

        with tab1:
            show_fig(line(filtered.groupby("Year")["Yield_ton_per_hectare"].mean(), "Yield (t/ha)"))
        with tab2:
            show_fig(hbar(filtered.groupby("Crop_Type")["Yield_ton_per_hectare"].mean(), "Average yield (t/ha)"))
        with tab3:
            show_df(filtered, height=420)
            st.download_button(
                "⬇️ Download filtered data (CSV)",
                filtered.to_csv(index=False).encode("utf-8"),
                file_name="farmwise_filtered.csv",
                mime="text/csv",
            )


# ================================================================
# RESOURCEWISE
# ================================================================

elif page == "💧 ResourceWise":

    page_header("💧 ResourceWise", "Resource use and agricultural efficiency analysis")

    with st.container(border=True):
        f1, f2 = st.columns(2)
        selected_crop = f1.selectbox("Crop", ["All"] + sorted(data_all["Crop_Type"].unique()), key="resource_crop")
        selected_state = f2.selectbox("State", ["All"] + sorted(data_all["State"].unique()), key="resource_state")

    filtered = data_all.copy()
    if selected_crop != "All":
        filtered = filtered[filtered["Crop_Type"] == selected_crop]
    if selected_state != "All":
        filtered = filtered[filtered["State"] == selected_state]

    if len(filtered) == 0:
        note("No records match these filters.", "warn")
    else:
        c1, c2, c3, c4 = st.columns(4)
        kpi(c1, "🧪", "Average fertilizer", f"{filtered['Fertilizer_Used_kg'].mean():.2f} kg", "", "gold")
        kpi(c2, "🧴", "Average pesticide", f"{filtered['Pesticide_Used_kg'].mean():.2f} kg", "", "brown")
        kpi(c3, "🌱", "Yield per fertilizer", f"{filtered['Yield_per_Fertilizer_kg'].mean():.4f}", "t/ha per kg")
        kpi(c4, "🐛", "Yield per pesticide", f"{filtered['Yield_per_Pesticide_kg'].mean():.4f}", "t/ha per kg", "teal")

        tab1, tab2, tab3 = st.tabs(["📊 Usage by crop", "⚖️ Efficiency", "⚠️ Screening"])

        with tab1:
            a, b = st.columns(2)
            with a:
                section("Fertilizer usage", "Average kg per record")
                show_fig(hbar(filtered.groupby("Crop_Type")["Fertilizer_Used_kg"].mean(), "kg"))
            with b:
                section("Pesticide usage", "Average kg per record")
                show_fig(hbar(filtered.groupby("Crop_Type")["Pesticide_Used_kg"].mean(), "kg"))

        with tab2:
            eff = (
                filtered.groupby("Crop_Type")
                .agg(
                    Yield=("Yield_ton_per_hectare", "mean"),
                    Fertilizer=("Fertilizer_Used_kg", "mean"),
                    Pesticide=("Pesticide_Used_kg", "mean"),
                    Yield_per_Fertilizer=("Yield_per_Fertilizer_kg", "mean"),
                    Yield_per_Pesticide=("Yield_per_Pesticide_kg", "mean"),
                )
                .round(4)
            )
            show_fig(hbar(eff["Yield_per_Fertilizer"], "Yield per fertilizer kg"))
            show_df(eff)

        with tab3:
            fert_q75 = data_all["Fertilizer_Used_kg"].quantile(0.75)
            pest_q75 = data_all["Pesticide_Used_kg"].quantile(0.75)
            yield_q25 = data_all["Yield_ton_per_hectare"].quantile(0.25)

            flagged = data_all[
                ((data_all["Fertilizer_Used_kg"] >= fert_q75) | (data_all["Pesticide_Used_kg"] >= pest_q75))
                & (data_all["Yield_ton_per_hectare"] <= yield_q25)
            ]

            k1, k2 = st.columns(2)
            kpi(k1, "⚠️", "Flagged records", f"{len(flagged):,}", "High resource use with low yield", "red")
            kpi(k2, "📐", "Share of dataset", f"{len(flagged) / len(data_all) * 100:.2f}%", "", "gold")

            note(
                "This is a statistical screening indicator (top 25% resource use with bottom 25% yield). "
                "It is not a fertilizer or pesticide prescription.",
                "info",
            )
            if len(flagged) > 0:
                show_df(
                    flagged[["State", "District", "Crop_Type", "Season", "Fertilizer_Used_kg",
                             "Pesticide_Used_kg", "Yield_ton_per_hectare"]].head(50),
                    height=380,
                )


# ================================================================
# COSTWISE
# ================================================================

elif page == "💰 CostWise":

    page_header("💰 CostWise", "Economic and farm profitability analysis")

    data = data_all

    c1, c2, c3, c4 = st.columns(4)
    kpi(c1, "🌾", "Average production", f"{data['Estimated_Production_tons'].mean():,.0f} tons")
    kpi(c2, "🏷️", "Average market price", money(data["Market_Price_per_ton"].mean()), "per ton", "gold")
    kpi(c3, "💰", "Average farmer profit", money(data["Farmer_Profit_INR"].mean()), "", "brown")
    kpi(c4, "📐", "Profit per hectare", money(data["Profit_per_hectare_INR"].mean()), "", "teal")

    with st.container(border=True):
        selected_crop = st.selectbox("Crop", ["All"] + sorted(data["Crop_Type"].unique()), key="cost_crop")

    filtered = data if selected_crop == "All" else data[data["Crop_Type"] == selected_crop]

    tab1, tab2, tab3 = st.tabs(["💰 Profit and revenue", "📊 Profit distribution", "📋 Summary table"])

    with tab1:
        a, b = st.columns(2)
        with a:
            section("Average profit by crop")
            show_fig(hbar(filtered.groupby("Crop_Type")["Farmer_Profit_INR"].mean(), "INR"))
        with b:
            section("Revenue per hectare")
            show_fig(hbar(filtered.groupby("Crop_Type")["Revenue_per_hectare_INR"].mean(), "INR per ha"))

    with tab2:
        q25 = data["Farmer_Profit_INR"].quantile(0.25)
        q75 = data["Farmer_Profit_INR"].quantile(0.75)
        k1, k2 = st.columns(2)
        kpi(k1, "📉", "Lower profit quartile", f"{(data['Farmer_Profit_INR'] <= q25).sum():,} records", f"At or below {money(q25)}", "red")
        kpi(k2, "📈", "Higher profit quartile", f"{(data['Farmer_Profit_INR'] >= q75).sum():,} records", f"At or above {money(q75)}")

        fig = px.histogram(filtered, x="Farmer_Profit_INR", nbins=40, color_discrete_sequence=["#2F7D3B"])
        fig.update_layout(xaxis_title="Farmer profit (INR)", yaxis_title="Records")
        show_fig(style_fig(fig, 320))

    with tab3:
        summary = (
            filtered.groupby("Crop_Type")
            .agg(
                Yield=("Yield_ton_per_hectare", "mean"),
                Production=("Estimated_Production_tons", "mean"),
                Market_Price=("Market_Price_per_ton", "mean"),
                Farmer_Profit=("Farmer_Profit_INR", "mean"),
                Profit_per_Hectare=("Profit_per_hectare_INR", "mean"),
                Profit_per_Ton=("Profit_per_ton_INR", "mean"),
            )
            .round(2)
        )
        show_df(summary)

    note(
        "The dataset has no explicit farm-cost column, so conventional accounting cost or "
        "profit margin should not be inferred from these indicators.",
        "info",
    )


# ================================================================
# CLIMATEWISE
# ================================================================

elif page == "🌦️ ClimateWise":

    page_header("🌦️ ClimateWise", "Climate variability and dataset-relative stress analysis")

    data = data_all.copy()

    c1, c2, c3 = st.columns(3)
    kpi(c1, "🌧️", "Average rainfall", f"{data['Rainfall_mm'].mean():.1f} mm", "", "teal")
    kpi(c2, "🌡️", "Average temperature", f"{data['Avg_Temperature_C'].mean():.2f} °C", "", "red")
    kpi(c3, "💦", "Average humidity", f"{data['Humidity_percent'].mean():.2f}%", "", "gold")

    r10, r90 = data["Rainfall_mm"].quantile([0.10, 0.90])
    t10, t90 = data["Avg_Temperature_C"].quantile([0.10, 0.90])
    h10, h90 = data["Humidity_percent"].quantile([0.10, 0.90])

    data["Climate_Stress_Score"] = (
        ((data["Rainfall_mm"] < r10) | (data["Rainfall_mm"] > r90)).astype(int)
        + ((data["Avg_Temperature_C"] < t10) | (data["Avg_Temperature_C"] > t90)).astype(int)
        + ((data["Humidity_percent"] < h10) | (data["Humidity_percent"] > h90)).astype(int)
    )
    data["Climate_Stress_Level"] = np.where(
    data["Climate_Stress_Score"] == 0,
    "Low",
    np.where(
        data["Climate_Stress_Score"] == 1,
        "Moderate",
        "High"
    )
)

    tab1, tab2, tab3 = st.tabs(["🔗 Climate and yield", "📐 Thresholds", "⚠️ Stress screening"])

    with tab1:
        corr = data[["Rainfall_mm", "Avg_Temperature_C", "Humidity_percent", "Yield_ton_per_hectare"]] \
            .corr()["Yield_ton_per_hectare"].drop("Yield_ton_per_hectare")
        a, b = st.columns([3, 2])
        with a:
            section("Correlation with yield")
            fig = px.bar(corr.rename_axis("Variable").reset_index(name="Correlation"),
                         x="Variable", y="Correlation", color_discrete_sequence=["#4FA3A5"])
            show_fig(style_fig(fig, 320))
        with b:
            section("Values")
            show_df(corr.rename("Correlation with yield").to_frame().round(4))

    with tab2:
        thresholds = pd.DataFrame({
            "Variable": ["Rainfall (mm)", "Temperature (°C)", "Humidity (%)"],
            "Lower threshold (10th percentile)": [r10, t10, h10],
            "Upper threshold (90th percentile)": [r90, t90, h90],
        })
        show_df(thresholds.round(2))

    with tab3:
        counts = data["Climate_Stress_Level"].value_counts()
        k1, k2, k3 = st.columns(3)
        kpi(k1, "🟢", "Low stress", f"{counts.get('Low', 0):,}")
        kpi(k2, "🟡", "Moderate stress", f"{counts.get('Moderate', 0):,}", "", "gold")
        kpi(k3, "🔴", "High stress", f"{counts.get('High', 0):,}", "", "red")

        a, b = st.columns([2, 3])
        with a:
            section("Stress mix")
            order = ["Low", "Moderate", "High"]
            fig = go.Figure(go.Pie(
                labels=order,
                values=[counts.get(k, 0) for k in order],
                hole=0.6,
                marker=dict(colors=["#7BB661", "#E0A526", "#C4543B"]),
            ))
            show_fig(style_fig(fig, 320, legend=True))
        with b:
            section("High-stress rate by state", "Share of records with high stress")
            state_stress = data.groupby("State")["Climate_Stress_Level"].apply(lambda x: (x == "High").mean() * 100)
            show_fig(hbar(state_stress, "% of records", 320))

    note(
        "Climate stress is defined relative to this dataset's 10th and 90th percentile "
        "thresholds. It is not a biological or agronomic diagnosis.",
        "info",
    )


# ================================================================
# YIELDWISE
# ================================================================

elif page == "📈 YieldWise":

    page_header("📈 YieldWise", "Experimental agricultural yield prediction component")

    report = None
    if os.path.exists(REPORT_PATH):
        try:
            with open(REPORT_PATH, "r") as f:
                report = json.load(f)
        except Exception:
            report = None

    tab1, tab2, tab3 = st.tabs(["🤖 Model evaluation", "🧪 Test predictions", "🔎 Interpretation"])

    with tab1:
        if report is None:
            note("Model report not found or unreadable.", "warn")
        else:
            model_name = report.get("selected_model") or report.get("final_model") or "Gradient Boosting"
            note(f"Selected model by validation performance: <b>{model_name}</b>", "info")

            results = report.get("model_comparison", report.get("results", None))
            if isinstance(results, list):
                show_df(pd.DataFrame(results))
            elif isinstance(results, dict):
                show_df(pd.DataFrame(results).T)
            else:
                with st.expander("Raw report"):
                    st.json(report)

    with tab2:
        if not os.path.exists(PREDICTIONS_PATH):
            note("Prediction file not found.", "warn")
        else:
            try:
                predictions = pd.read_csv(PREDICTIONS_PATH)

                actual_col, predicted_col = None, None
                for col in predictions.columns:
                    low = col.lower()
                    if "actual" in low and "yield" in low:
                        actual_col = col
                    if "pred" in low and "yield" in low:
                        predicted_col = col

                if actual_col and predicted_col:
                    comp = predictions[[actual_col, predicted_col]].dropna().copy()
                    comp.columns = ["Actual", "Predicted"]

                    err = comp["Predicted"] - comp["Actual"]
                    mae = err.abs().mean()
                    rmse = float(np.sqrt((err ** 2).mean()))
                    ss_res = float((err ** 2).sum())
                    ss_tot = float(((comp["Actual"] - comp["Actual"].mean()) ** 2).sum())
                    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

                    k1, k2, k3, k4 = st.columns(4)
                    kpi(k1, "📋", "Test records", f"{len(comp):,}")
                    kpi(k2, "📏", "MAE", f"{mae:.3f}", "t/ha", "gold")
                    kpi(k3, "📐", "RMSE", f"{rmse:.3f}", "t/ha", "teal")
                    kpi(k4, "📉", "R²", f"{r2:.3f}", "Computed from this file", "red")

                    a, b = st.columns(2)

                    with a:
                        section("Actual vs predicted")
                        lo = float(min(comp.min()))
                        hi = float(max(comp.max()))

                        fig = px.scatter(
                            comp,
                            x="Actual",
                            y="Predicted",
                            opacity=0.55,
                            color_discrete_sequence=["#2F7D3B"]
                        )

                        fig.add_trace(
                            go.Scatter(
                                x=[lo, hi],
                                y=[lo, hi],
                                mode="lines",
                                line=dict(color="#E0A526", dash="dash"),
                                name="Perfect fit"
                            )
                        )

                        show_fig(style_fig(fig, 340))

                    with b:
                        section("First 100 test records")

                        fig = go.Figure()

                        fig.add_trace(
                            go.Scatter(
                                y=comp["Actual"].head(100),
                                name="Actual",
                                line=dict(color="#2F7D3B")
                            )
                        )

                        fig.add_trace(
                            go.Scatter(
                                y=comp["Predicted"].head(100),
                                name="Predicted",
                                line=dict(color="#E0A526")
                            )
                        )

                        show_fig(style_fig(fig, 340, legend=True))

                with st.expander("🔍 Prediction table"):
                    show_df(predictions.head(50))

            except Exception as e:
                note(f"Could not load prediction file: {e}", "danger")

    with tab3:
        section("Feature importance", "Values carried over from the earlier dashboard version")

        fi = pd.DataFrame({
            "Feature": [
                "Rainfall",
                "Cultivated Area",
                "Fertilizer",
                "Pesticide",
                "Temperature",
                "Humidity",
                "Year"
            ],
            "Importance": [
                0.16579,
                0.14564,
                0.12577,
                0.10543,
                0.08912,
                0.07269,
                0.02209
            ],
        }).set_index("Feature")["Importance"]

        show_fig(hbar(fi, "Importance"))

        st.caption(
            "Feature importance describes what the model used, not causal effects on yield."
        )

# ================================================================
# WHAT-IF SIMULATOR
# ================================================================

elif page == "🔄 What-If Simulator":

    page_header("🔄 What-If Simulator", "Model-based historical scenario analysis for resource changes")

    note(
        "This simulator compares a resource scenario against historically similar observations. "
        "It does <b>not</b> claim causal effects or guarantee future yield.",
        "info",
    )

    data = df.copy()

    section("1. Choose the farm context")
    with st.container(border=True):
        c1, c2, c3 = st.columns(3)
        selected_crop = c1.selectbox("Crop", sorted(data["Crop_Type"].unique()), key="whatif_crop")
        selected_state = c2.selectbox("State", sorted(data["State"].unique()), key="whatif_state")
        selected_season = c3.selectbox("Season", sorted(data["Season"].unique()), key="whatif_season")

    reference = data[
        (data["Crop_Type"] == selected_crop) & (data["State"] == selected_state) & (data["Season"] == selected_season)
    ].copy()
    fallback_message = ""

    if len(reference) < 5:
        reference = data[(data["Crop_Type"] == selected_crop) & (data["State"] == selected_state)].copy()
        fallback_message = "Limited crop-state-season history. Using crop-state historical context."
    if len(reference) < 5:
        reference = data[data["Crop_Type"] == selected_crop].copy()
        fallback_message = "Limited crop-state history. Using crop-level historical context."
    if len(reference) < 5:
        reference = data.copy()
        fallback_message = "Limited crop-specific history. Using full-dataset historical context."

    if fallback_message:
        note(fallback_message, "warn")

    baseline_fertilizer = float(reference["Fertilizer_Used_kg"].median())
    baseline_pesticide = float(reference["Pesticide_Used_kg"].median())
    baseline_yield = float(reference["Yield_ton_per_hectare"].median())
    baseline_profit = float(reference["Farmer_Profit_INR"].median())

    section("2. Set the resource scenario")
    with st.container(border=True):
        s1, s2 = st.columns(2)
        fertilizer_reduction = s1.slider("Fertilizer reduction (%)", 0, 50, 15, 5)
        pesticide_reduction = s2.slider("Pesticide reduction (%)", 0, 50, 30, 5)

    scenario_fertilizer = baseline_fertilizer * (1 - fertilizer_reduction / 100)
    scenario_pesticide = baseline_pesticide * (1 - pesticide_reduction / 100)

    section("3. Baseline and scenario")
    b1, b2, b3, b4 = st.columns(4)
    kpi(b1, "🧪", "Baseline fertilizer", f"{baseline_fertilizer:.2f} kg", "Historical median", "gold")
    kpi(b2, "🧴", "Baseline pesticide", f"{baseline_pesticide:.2f} kg", "Historical median", "brown")
    kpi(b3, "📈", "Median yield", f"{baseline_yield:.2f} t/ha")
    kpi(b4, "💰", "Median profit", money(baseline_profit), "", "teal")

    d1, d2 = st.columns([3, 2])
    with d1:
        comp = pd.DataFrame({
            "Resource": ["Fertilizer (kg)", "Fertilizer (kg)", "Pesticide (kg)", "Pesticide (kg)"],
            "Case": ["Baseline", "Scenario", "Baseline", "Scenario"],
            "Amount": [baseline_fertilizer, scenario_fertilizer, baseline_pesticide, scenario_pesticide],
        })
        fig = px.bar(comp, x="Resource", y="Amount", color="Case", barmode="group",
                     color_discrete_map={"Baseline": "#8C6A4A", "Scenario": "#2F7D3B"})
        show_fig(style_fig(fig, 320, legend=True))
    with d2:
        kpi(st, "🧪", "Scenario fertilizer", f"{scenario_fertilizer:.2f} kg", f"-{fertilizer_reduction}%")
        kpi(st, "🧴", "Scenario pesticide", f"{scenario_pesticide:.2f} kg", f"-{pesticide_reduction}%", "teal")

    def safe_ratio(numerator, denominator):
        return numerator / denominator if denominator > 0 else np.nan

    base_yf = safe_ratio(baseline_yield, baseline_fertilizer)
    scen_yf = safe_ratio(baseline_yield, scenario_fertilizer)
    base_yp = safe_ratio(baseline_yield, baseline_pesticide)
    scen_yp = safe_ratio(baseline_yield, scenario_pesticide)

    fert_eff_change = (scen_yf / base_yf - 1) * 100 if base_yf and not np.isnan(base_yf) else np.nan
    pest_eff_change = (scen_yp / base_yp - 1) * 100 if base_yp and not np.isnan(base_yp) else np.nan

    section("4. Efficiency comparison", "Yield is held at the historical median, so this is arithmetic, not a forecast")
    e1, e2 = st.columns(2)
    kpi(e1, "🌱", "Yield per fertilizer", f"{scen_yf:.4f}", f"{fert_eff_change:+.2f}% vs baseline")
    kpi(e2, "🐛", "Yield per pesticide", f"{scen_yp:.4f}", f"{pest_eff_change:+.2f}% vs baseline", "teal")

    section("5. Similar historical cases")

    feature_cols = ["Rainfall_mm", "Avg_Temperature_C", "Humidity_percent", "Fertilizer_Used_kg", "Pesticide_Used_kg"]
    scenario_vector = pd.Series(
        [reference["Rainfall_mm"].median(), reference["Avg_Temperature_C"].median(),
         reference["Humidity_percent"].median(), scenario_fertilizer, scenario_pesticide],
        index=feature_cols, dtype=float,
    )

    ref_matrix = reference[feature_cols].astype(float)
    means = ref_matrix.mean()
    stds = ref_matrix.std().replace(0, 1).fillna(1)

    reference_scaled = (ref_matrix - means) / stds
    scenario_scaled = (scenario_vector - means) / stds

    reference = reference.copy()
    reference["Scenario_Distance"] = np.sqrt(((reference_scaled.values - scenario_scaled.values) ** 2).sum(axis=1))
    nearest = reference.sort_values("Scenario_Distance").head(10)

    a1, a2, a3 = st.columns(3)
    kpi(a1, "📈", "Analogue median yield", f"{nearest['Yield_ton_per_hectare'].median():.2f} t/ha")
    kpi(a2, "📊", "Analogue mean yield", f"{nearest['Yield_ton_per_hectare'].mean():.2f} t/ha", "", "gold")
    kpi(a3, "📚", "Cases searched", f"{len(reference):,}", "", "teal")

    show_df(
        nearest[["State", "District", "Crop_Type", "Season", "Rainfall_mm", "Avg_Temperature_C",
                 "Humidity_percent", "Fertilizer_Used_kg", "Pesticide_Used_kg",
                 "Yield_ton_per_hectare", "Farmer_Profit_INR", "Scenario_Distance"]].round(3)
    )

    note(
        "<b>Interpretation limitation.</b> The analogue yield is not a prediction for the simulated "
        "scenario. Efficiency indicators compare selected resource quantities only, and the "
        "simulator does not establish causal effects of reducing fertilizer or pesticide use."
    )


# ================================================================
# PESTRISK / ECOADVISOR
# ================================================================

elif page == "🐛 PestRisk / EcoAdvisor":

    page_header("🐛 Ask FarmWise AI", "Grounded rice-pest question answering with verified evidence")

    st.markdown(
        '<div class="pipe">'
        '<div class="pipe-step"><b>1. Safety guard</b>Blocks dosage and prescription requests</div>'
        '<div class="pipe-step"><b>2. Retrieval</b>FAISS search over verified rice-pest evidence</div>'
        '<div class="pipe-step"><b>3. Generation</b>Qwen writes an answer from that evidence only</div>'
        '<div class="pipe-step"><b>4. Output check</b>Unsafe dosage text is blocked</div>'
        '<div class="pipe-step"><b>5. Source trace</b>Shows which chunks were used</div></div>',
        unsafe_allow_html=True,
    )

    # ------------------------------------------------------------
    # Check files
    # ------------------------------------------------------------

    if not os.path.exists(FAISS_PATH):
        note(f"FAISS index not found: {FAISS_PATH}", "danger")
        st.stop()
    if not os.path.exists(METADATA_PATH):
        note(f"RAG metadata not found: {METADATA_PATH}", "danger")
        st.stop()

    # ------------------------------------------------------------
    # Cached loaders
    # ------------------------------------------------------------

    @st.cache_resource
    def load_rag_system(index_path, metadata_file):
        import faiss
        from sentence_transformers import SentenceTransformer

        index = faiss.read_index(index_path)

        with open(metadata_file, "r", encoding="utf-8") as f:
            obj = json.load(f)

        if isinstance(obj, list):
            records = obj
        elif isinstance(obj, dict):
            records = obj.get("records") or obj.get("data") or [obj]
        else:
            records = []

        encoder = SentenceTransformer("all-MiniLM-L6-v2")
        return index, records, encoder

    @st.cache_resource
    def load_qwen_model():
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        model_name = "Qwen/Qwen2.5-1.5B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )
        return tokenizer, model

    try:
        index, rag_records, encoder = load_rag_system(FAISS_PATH, METADATA_PATH)
    except Exception as e:
        note(f"Could not load RAG system: {e}", "danger")
        st.stop()

    encoder_dim = encoder.get_sentence_embedding_dimension()
    if index.d != encoder_dim:
        note(
            f"Embedding size mismatch: the index uses {index.d} dimensions but "
            f"all-MiniLM-L6-v2 produces {encoder_dim}. Use the same encoder that built the index.",
            "danger",
        )
        st.stop()

    k1, k2, k3 = st.columns(3)
    kpi(k1, "🧠", "Evidence vectors", f"{index.ntotal:,}")
    kpi(k2, "📐", "Embedding size", f"{index.d}", "", "teal")
    kpi(k3, "🛡️", "Dosage guard", "Active", "Input and output checks", "gold")

    # ------------------------------------------------------------
    # Safety patterns
    # ------------------------------------------------------------

    dosage_patterns = [
        r"\bdose\b", r"\bdosage\b", r"\bml\s*/\s*ha\b", r"\bml\s+per\s+ha\b",
        r"\bkg\s*/\s*ha\b", r"\bkg\s+per\s+ha\b", r"\blitre\s*/\s*ha\b", r"\bliter\s*/\s*ha\b",
        r"\bhow much pesticide\b", r"\bhow much insecticide\b",
        r"\brecommended pesticide\b", r"\brecommend pesticide\b",
        r"\bwhich pesticide\b", r"\bwhich insecticide\b",
    ]

    output_patterns = [
        r"\b\d+\s*(ml|l|kg)\s*/?\s*ha\b", r"\b\d+\s*(ml|l|kg)\s+per\s+ha\b",
        r"\bpesticide dose\b", r"\bapplication rate\b", r"\binsecticide dose\b",
    ]

    def matches_any(text, patterns):
        text = text.lower()
        return any(re.search(p, text) for p in patterns)

    def record_text(record):
        return record.get("text") or record.get("content") or record.get("chunk") or record.get("fact") or ""

    def record_source(record):
        return record.get("source_id") or record.get("source") or "UNKNOWN"

    def record_chunk(record):
        return record.get("chunk_id") or record.get("id") or "UNKNOWN"

    # ------------------------------------------------------------
    # Retrieval + generation
    # ------------------------------------------------------------

    def retrieve(question, top_k=5, min_score=0.50):
        query_embedding = encoder.encode([question], normalize_embeddings=True).astype("float32")
        scores, indices = index.search(query_embedding, top_k)

        retrieved = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(rag_records) or float(score) < min_score:
                continue
            record = rag_records[idx]
            verified = record.get("verified", record.get("verification_status", True))
            if isinstance(verified, str):
                verified = verified.lower() in ["true", "verified", "yes", "approved"]
            if not verified:
                continue
            retrieved.append({"score": float(score), "record": record})
        return retrieved

    SYSTEM_PROMPT = """
You are FarmWise AI EcoAdvisor.

You are a grounded agricultural information assistant.

Use ONLY the supplied VERIFIED EVIDENCE.

Do not invent agricultural facts.

Do not claim symptoms prove a diagnosis.

Do not provide pesticide dosage,
chemical-treatment instructions,
application rates or insecticide prescriptions.

If the evidence is insufficient, clearly say so.

Your answer MUST use these sections:

RISK ASSESSMENT

KEY OBSERVATIONS

RECOMMENDED ACTIONS

WHAT TO AVOID

EVIDENCE LIMITATIONS

SOURCES

Keep the answer concise and practical.
"""

    def generate_answer(question, retrieved):
        import torch

        tokenizer, model = load_qwen_model()

        context = "\n".join(
            f"[{record_source(i['record'])} | {record_chunk(i['record'])}]\n\n{record_text(i['record'])}\n"
            for i in retrieved if record_text(i["record"])
        )

        user_prompt = (
            f"USER QUESTION:\n\n{question}\n\nVERIFIED EVIDENCE:\n\n{context}\n\n"
            "Answer the user's question using only the verified evidence above."
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt")

        if torch.cuda.is_available():
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            output = model.generate(
                **inputs, max_new_tokens=500, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated = output[0][inputs["input_ids"].shape[1]:]
        return tokenizer.decode(generated, skip_special_tokens=True).strip()

    # ------------------------------------------------------------
    # Session state
    # ------------------------------------------------------------

    if "farmwise_chat" not in st.session_state:
        st.session_state.farmwise_chat = []
    if "farmwise_pending" not in st.session_state:
        st.session_state.farmwise_pending = None

    # ------------------------------------------------------------
    # Example questions
    # ------------------------------------------------------------

    section("Try an example")

    examples = {
        "🌾 Folded rice leaves": "Rice leaves are folded and damaged. What pest could cause these symptoms?",
        "🐛 Hopper symptoms": "What symptoms are associated with brown plant hopper in rice?",
        "🍃 Leaf damage": "What rice pests can cause leaf damage and drying symptoms?",
        "🛡️ Safety test": "What pesticide dosage should I use?",
    }

    cols = st.columns(len(examples))
    for col, (label, text) in zip(cols, examples.items()):
        try:
            clicked = col.button(label, width="stretch")
        except TypeError:
            clicked = col.button(label, use_container_width=True)
        if clicked:
            st.session_state.farmwise_pending = text
            st.rerun()

    # ------------------------------------------------------------
    # Conversation history
    # ------------------------------------------------------------

    section("Conversation")

    if not st.session_state.farmwise_chat:
        note("No messages yet. Pick an example above or describe a rice-pest symptom below.", "info")

    for message in st.session_state.farmwise_chat:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("trace"):
                with st.expander("🔗 Evidence source trace"):
                    show_df(pd.DataFrame(message["trace"]))
                    st.caption("Similarity shows retrieval relevance. It is not proof of factual entailment.")

    # ------------------------------------------------------------
    # Input
    # ------------------------------------------------------------

    typed = st.chat_input("Ask about rice pest symptoms or identification...")
    question = typed or st.session_state.farmwise_pending
    st.session_state.farmwise_pending = None

    if question and question.strip():
        question = question.strip()
        st.session_state.farmwise_chat.append({"role": "user", "content": question})

        if matches_any(question, dosage_patterns):
            answer = (
                "### 🚫 Safety limitation\n\n"
                "FarmWise AI does not provide pesticide dosage, application-rate, "
                "chemical-treatment or insecticide prescription instructions.\n\n"
                "You can ask about:\n\n"
                "- Pest identification\n- Visible symptoms\n"
                "- Differences between similar pest symptoms\n"
                "- Verified evidence from the FarmWise knowledge base"
            )
            st.session_state.farmwise_chat.append({"role": "assistant", "content": answer})
            st.rerun()

        with st.spinner("🔎 Searching verified evidence and preparing the answer..."):
            retrieved = retrieve(question)

            if not retrieved:
                answer = (
                    "### ⚠️ Insufficient verified evidence\n\n"
                    "I could not retrieve sufficiently similar verified evidence from the FarmWise "
                    "rice-pest knowledge base.\n\n"
                    "Please describe the visible symptoms more specifically, for example whether the "
                    "leaves are folded, yellowing, drying or showing hopper-related damage.\n\n"
                    "**Limitation:** I should not make a pest diagnosis when verified evidence is insufficient."
                )
                st.session_state.farmwise_chat.append({"role": "assistant", "content": answer})
                st.rerun()

            trace = [
                {"source_id": record_source(i["record"]), "chunk_id": record_chunk(i["record"]),
                 "similarity": round(i["score"], 4)}
                for i in retrieved
            ]

            try:
                answer = generate_answer(question, retrieved)
                if matches_any(answer, output_patterns):
                    answer = (
                        "### 🚫 Response blocked\n\n"
                        "The generated response contained potentially unsafe pesticide-treatment "
                        "or dosage information, so FarmWise AI blocked it. "
                        "Please ask about pest symptoms, identification or verified observations."
                    )
                    trace = None
            except Exception as e:
                answer = (
                    "### ⚠️ FarmWise AI could not generate the answer\n\n"
                    "Evidence retrieval completed, but the language model could not run in this session.\n\n"
                    f"Technical detail: `{e}`"
                )

        st.session_state.farmwise_chat.append({"role": "assistant", "content": answer, "trace": trace})
        st.rerun()

    if st.session_state.farmwise_chat:
        if st.button("🗑️ Clear conversation"):
            st.session_state.farmwise_chat = []
            st.rerun()
