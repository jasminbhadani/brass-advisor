import os
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import pickle
from fetch_data import fetch_prices
from sentiment import get_sentiment

st.set_page_config(
    page_title="Brass Purchase Advisor",
    page_icon="🔩",
    layout="wide"
)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🔩 Brass Raw Material Purchase Advisor")
st.caption("Helping Gujarat Brass Manufacturers Buy Smarter · Powered by AI")

# ── Load or Refresh Data ──────────────────────────────────────────────────────
if st.button("🔄 Refresh Prices"):
    df = fetch_prices()
else:
    try:
        df = pd.read_csv("data/prices.csv", index_col=0, parse_dates=True)
    except FileNotFoundError:
        st.warning("No data found. Fetching now...")
        df = fetch_prices()

# ── Load Model ────────────────────────────────────────────────────────────────
try:
    with open("models/copper_model.pkl", "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("Model not found. Please run train_model.py first.")
    st.stop()

# ── Prepare Features & Predict ───────────────────────────────────────────────
df["copper_lag1"]  = df["copper"].shift(1)
df["copper_lag7"]  = df["copper"].shift(7)
df["copper_lag30"] = df["copper"].shift(30)
df["usdinr_lag1"]  = df["usdinr"].shift(1)
df["zinc_lag1"]    = df["zinc"].shift(1)
df.dropna(inplace=True)

features = ["copper_lag1", "copper_lag7", "copper_lag30", "usdinr_lag1", "zinc_lag1"]
latest   = df[features].iloc[[-1]]

# ── Core Values ───────────────────────────────────────────────────────────────
copper_usd       = float(df["copper"].iloc[-1])         # USD/tonne
zinc_usd         = float(df["zinc"].iloc[-1])           # USD/tonne
usd_inr          = float(df["usdinr"].iloc[-1])         # INR per 1 USD
forecast_usd     = float(model.predict(latest)[0])      # USD/tonne forecast

# INR per kg
copper_inr       = (copper_usd   / 1000) * usd_inr
zinc_inr         = (zinc_usd     / 1000) * usd_inr
forecast_inr     = (forecast_usd / 1000) * usd_inr

# Brass = 65% Copper + 35% Zinc
brass_inr_today    = (copper_inr  * 0.65) + (zinc_inr * 0.35)
brass_inr_forecast = (forecast_inr * 0.65) + (zinc_inr * 0.35)  # zinc assumed stable
brass_usd_today    = (copper_usd  * 0.65) + (zinc_usd  * 0.35)
brass_usd_forecast = (forecast_usd * 0.65) + (zinc_usd  * 0.35)

# Change %
copper_change  = ((forecast_usd      - copper_usd)      / copper_usd)      * 100
brass_change   = ((brass_inr_forecast - brass_inr_today) / brass_inr_today) * 100

last_date = df.index[-1].strftime("%d %b %Y")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — RAW MATERIAL PRICES TODAY
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("📦 Raw Material Prices — Today")
st.caption(f"Source: LME (London Metal Exchange) via Yahoo Finance · Last updated: {last_date}")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    label="🔶 Copper (International)",
    value=f"USD {copper_usd:,.0f}",
    delta="per metric tonne"
)
col2.metric(
    label="🔶 Copper (India)",
    value=f"₹ {copper_inr:,.1f}",
    delta="per kg · Indian Rupee"
)
col3.metric(
    label="🔷 Zinc (International)",
    value=f"USD {zinc_usd:,.0f}",
    delta="per metric tonne"
)
col4.metric(
    label="🔷 Zinc (India)",
    value=f"₹ {zinc_inr:,.1f}",
    delta="per kg · Indian Rupee"
)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — BRASS PRICE ESTIMATE
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("🔩 Estimated Brass Price Today")
st.caption("Formula: 65% Copper + 35% Zinc  (standard brass alloy ratio)")

col5, col6, col7 = st.columns(3)

col5.metric(
    label="🔩 Brass Price (International)",
    value=f"USD {brass_usd_today:,.0f}",
    delta="per metric tonne · estimated"
)
col6.metric(
    label="🔩 Brass Price (India)",
    value=f"₹ {brass_inr_today:,.1f}",
    delta="per kg · Indian Rupee · estimated"
)
col7.metric(
    label="💱 Exchange Rate",
    value=f"₹ {usd_inr:.2f}",
    delta="1 USD = INR · live rate"
)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — AI FORECAST
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("🤖 AI Forecast — Next 7 Days")
st.caption("XGBoost model · trained on LME copper prices + USD/INR + zinc data")

col8, col9, col10 = st.columns(3)

col8.metric(
    label="🔶 Copper Forecast",
    value=f"USD {forecast_usd:,.0f}",
    delta=f"{copper_change:+.1f}% from today · per tonne"
)
col9.metric(
    label="🔩 Brass Price Forecast (India)",
    value=f"₹ {brass_inr_forecast:,.1f}",
    delta=f"{brass_change:+.1f}% from today · per kg"
)
col10.metric(
    label="📊 Expected Movement",
    value="Rising 📈" if brass_change > 2 else "Falling 📉" if brass_change < -2 else "Stable ➡️",
    delta=f"{brass_change:+.1f}% brass price change"
)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — PURCHASE RECOMMENDATION
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("💡 Purchase Recommendation")

if brass_change > 2:
    st.error(f"""
🔴  **BUY NOW** — Brass price expected to RISE by {brass_change:+.1f}% in next 7 days

| Material | Today (USD/t) | Today (INR/kg) | Forecast (INR/kg) | Change |
|---|---|---|---|---|
| 🔶 Copper | USD {copper_usd:,.0f} | ₹ {copper_inr:,.1f} | ₹ {forecast_inr:,.1f} | {copper_change:+.1f}% |
| 🔷 Zinc | USD {zinc_usd:,.0f} | ₹ {zinc_inr:,.1f} | ₹ {zinc_inr:,.1f} | stable |
| 🔩 **Brass (Est.)** | USD {brass_usd_today:,.0f} | **₹ {brass_inr_today:,.1f}** | **₹ {brass_inr_forecast:,.1f}** | **{brass_change:+.1f}%** |

**✅ Advice:** Purchase brass scrap this week before prices climb.
""")
elif brass_change < -2:
    st.success(f"""
🟢  **WAIT** — Brass price expected to FALL by {abs(brass_change):.1f}% in next 7 days

| Material | Today (USD/t) | Today (INR/kg) | Forecast (INR/kg) | Change |
|---|---|---|---|---|
| 🔶 Copper | USD {copper_usd:,.0f} | ₹ {copper_inr:,.1f} | ₹ {forecast_inr:,.1f} | {copper_change:+.1f}% |
| 🔷 Zinc | USD {zinc_usd:,.0f} | ₹ {zinc_inr:,.1f} | ₹ {zinc_inr:,.1f} | stable |
| 🔩 **Brass (Est.)** | USD {brass_usd_today:,.0f} | **₹ {brass_inr_today:,.1f}** | **₹ {brass_inr_forecast:,.1f}** | **{brass_change:+.1f}%** |

**✅ Advice:** Hold purchase. Better rates expected soon.
""")
else:
    st.warning(f"""
🟡  **NEUTRAL** — Brass price expected to remain STABLE ({brass_change:+.1f}%)

| Material | Today (USD/t) | Today (INR/kg) | Forecast (INR/kg) | Change |
|---|---|---|---|---|
| 🔶 Copper | USD {copper_usd:,.0f} | ₹ {copper_inr:,.1f} | ₹ {forecast_inr:,.1f} | {copper_change:+.1f}% |
| 🔷 Zinc | USD {zinc_usd:,.0f} | ₹ {zinc_inr:,.1f} | ₹ {zinc_inr:,.1f} | stable |
| 🔩 **Brass (Est.)** | USD {brass_usd_today:,.0f} | **₹ {brass_inr_today:,.1f}** | **₹ {brass_inr_forecast:,.1f}** | **{brass_change:+.1f}%** |

**✅ Advice:** Buy as per your normal schedule. No urgent action needed.
""")

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — PRICE CHARTS
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("📊 Price Trends — Last 6 Months")

tab1, tab2, tab3 = st.tabs([
    "🔩 Brass Price (INR/kg)",
    "🔶 Copper (USD/tonne)",
    "🔷 Zinc (USD/tonne)"
])

# Brass INR chart
with tab1:
    st.caption("Estimated brass price in INR/kg · Formula: 65% copper + 35% zinc")
    brass_series = ((df["copper"].iloc[-180:] * 0.65) + (df["zinc"].iloc[-180:] * 0.35)) / 1000 * usd_inr
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=brass_series.index,
        y=brass_series.values,
        name="Brass (INR/kg)",
        line=dict(color="#b5651d", width=2.5),
        hovertemplate="Date: %{x}<br>Brass: ₹%{y:,.1f}/kg<extra></extra>"
    ))
    fig1.add_hline(
        y=brass_inr_forecast,
        line_dash="dash",
        line_color="red",
        annotation_text=f"7-Day Forecast: ₹{brass_inr_forecast:,.1f}/kg",
        annotation_position="bottom right"
    )
    fig1.update_layout(
        height=380,
        margin=dict(l=0, r=0, t=20, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis_title="INR per kg",
        xaxis_title="Date"
    )
    fig1.update_yaxes(tickprefix="₹ ", tickformat=",")
    st.plotly_chart(fig1, use_container_width=True)

# Copper USD chart
with tab2:
    st.caption("LME Copper Futures · USD (United States Dollar) per metric tonne")
    recent_copper = df["copper"].iloc[-180:]
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=recent_copper.index,
        y=recent_copper.values,
        name="Copper (USD/tonne)",
        line=dict(color="#f0a500", width=2),
        hovertemplate="Date: %{x}<br>Copper: USD %{y:,.0f}/tonne<extra></extra>"
    ))
    fig2.add_hline(
        y=forecast_usd,
        line_dash="dash",
        line_color="red",
        annotation_text=f"7-Day Forecast: USD {forecast_usd:,.0f}",
        annotation_position="bottom right"
    )
    fig2.update_layout(
        height=380,
        margin=dict(l=0, r=0, t=20, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis_title="USD per tonne",
        xaxis_title="Date"
    )
    fig2.update_yaxes(tickprefix="USD ", tickformat=",")
    st.plotly_chart(fig2, use_container_width=True)

# Zinc USD chart
with tab3:
    st.caption("LME Zinc Futures · USD (United States Dollar) per metric tonne")
    recent_zinc = df["zinc"].iloc[-180:]
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=recent_zinc.index,
        y=recent_zinc.values,
        name="Zinc (USD/tonne)",
        line=dict(color="#4a90d9", width=2),
        hovertemplate="Date: %{x}<br>Zinc: USD %{y:,.0f}/tonne<extra></extra>"
    ))
    fig3.update_layout(
        height=380,
        margin=dict(l=0, r=0, t=20, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis_title="USD per tonne",
        xaxis_title="Date"
    )
    fig3.update_yaxes(tickprefix="USD ", tickformat=",")
    st.plotly_chart(fig3, use_container_width=True)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — NEWS SENTIMENT
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("📰 Market News Sentiment")
st.caption("Latest copper & metal market news · keyword-based sentiment scoring")

with st.spinner("Fetching latest news..."):
    signal, headlines, score = get_sentiment()

col11, col12 = st.columns([1, 3])
col11.metric("Market Signal", signal)
col12.write("**Latest Headlines:**")
if headlines:
    for h in headlines:
        col12.write(f"• {h}")
else:
    col12.info("No headlines available. Check your NewsAPI key in sentiment.py")

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════════
st.caption("""
📌 **Currency & Unit Reference:**
USD = United States Dollar · INR = Indian Rupee (₹) · 1 tonne = 1,000 kg

🔩 **Brass Price Calculation:**
Estimated brass price = (Copper price × 65%) + (Zinc price × 35%)
Based on standard brass alloy composition (CuZn35)

⚠️ *AI forecast is for guidance only — not financial advice.
Actual market prices may vary based on local conditions in Jamnagar / Gujarat.*
""")