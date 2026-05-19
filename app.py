import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import pickle
from fetch_data import fetch_prices
from sentiment import get_sentiment
# Auto-setup on first run (for Streamlit Cloud)
import os

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

# Core values
copper_usd_tonne   = float(df["copper"].iloc[-1])       # USD per tonne
forecast_usd_tonne = float(model.predict(latest)[0])    # USD per tonne
usd_inr            = float(df["usdinr"].iloc[-1])       # 1 USD = X INR
change_pct         = ((forecast_usd_tonne - copper_usd_tonne) / copper_usd_tonne) * 100

# INR per kg
copper_inr_kg   = (copper_usd_tonne   / 1000) * usd_inr
forecast_inr_kg = (forecast_usd_tonne / 1000) * usd_inr
change_inr_pct  = ((forecast_inr_kg - copper_inr_kg) / copper_inr_kg) * 100

# ── SECTION 1 — Today's Price ─────────────────────────────────────────────────
st.subheader("📈 Today's Copper Price")
st.caption(f"Source: LME (London Metal Exchange) · Last updated: {df.index[-1].strftime('%d %b %Y')}")

col1, col2, col3 = st.columns(3)

col1.metric(
    label="🔶 Copper Today  (International)",
    value=f"USD {copper_usd_tonne:,.0f}",
    delta="per metric tonne · US Dollar"
)
col2.metric(
    label="🇮🇳 Copper Today  (India)",
    value=f"₹ {copper_inr_kg:,.1f}",
    delta="per kg · Indian Rupee"
)
col3.metric(
    label="💱 Exchange Rate",
    value=f"₹ {usd_inr:.2f}",
    delta="1 USD = INR · Live rate"
)

st.divider()

# ── SECTION 2 — AI Forecast ───────────────────────────────────────────────────
st.subheader("🤖 AI Forecast — Next 7 Days")
st.caption("XGBoost model trained on historical LME copper prices + USD/INR rate")

col4, col5, col6 = st.columns(3)

col4.metric(
    label="🔮 Forecast  (International)",
    value=f"USD {forecast_usd_tonne:,.0f}",
    delta=f"{change_pct:+.1f}% from today · per tonne"
)
col5.metric(
    label="🔮 Forecast  (India)",
    value=f"₹ {forecast_inr_kg:,.1f}",
    delta=f"{change_inr_pct:+.1f}% from today · per kg"
)
col6.metric(
    label="📊 Price Movement",
    value="Rising 📈" if change_pct > 2 else "Falling 📉" if change_pct < -2 else "Stable ➡️",
    delta=f"{change_pct:+.1f}% expected in 7 days"
)

st.divider()

# ── SECTION 3 — Recommendation ───────────────────────────────────────────────
st.subheader("💡 Purchase Recommendation")

if change_pct > 2:
    st.error(f"""
🔴  **BUY NOW** — Copper price expected to RISE by {change_pct:+.1f}% in next 7 days

| | Today | 7-Day Forecast | Change |
|---|---|---|---|
| **International (USD/tonne)** | USD {copper_usd_tonne:,.0f} | USD {forecast_usd_tonne:,.0f} | {change_pct:+.1f}% |
| **India (INR/kg)** | ₹ {copper_inr_kg:,.1f} | ₹ {forecast_inr_kg:,.1f} | {change_inr_pct:+.1f}% |

**Advice:** Purchase brass scrap this week before prices climb.
""")
elif change_pct < -2:
    st.success(f"""
🟢  **WAIT** — Copper price expected to FALL by {change_pct:.1f}% in next 7 days

| | Today | 7-Day Forecast | Change |
|---|---|---|---|
| **International (USD/tonne)** | USD {copper_usd_tonne:,.0f} | USD {forecast_usd_tonne:,.0f} | {change_pct:+.1f}% |
| **India (INR/kg)** | ₹ {copper_inr_kg:,.1f} | ₹ {forecast_inr_kg:,.1f} | {change_inr_pct:+.1f}% |

**Advice:** Hold purchase. Better rates expected soon.
""")
else:
    st.warning(f"""
🟡  **NEUTRAL** — Copper price expected to remain STABLE ({change_pct:+.1f}%)

| | Today | 7-Day Forecast | Change |
|---|---|---|---|
| **International (USD/tonne)** | USD {copper_usd_tonne:,.0f} | USD {forecast_usd_tonne:,.0f} | {change_pct:+.1f}% |
| **India (INR/kg)** | ₹ {copper_inr_kg:,.1f} | ₹ {forecast_inr_kg:,.1f} | {change_inr_pct:+.1f}% |

**Advice:** Buy as per your normal schedule. No urgent action needed.
""")

st.divider()

# ── SECTION 4 — Price Charts (Tabbed) ────────────────────────────────────────
st.subheader("📊 Copper Price Trend — Last 6 Months")

tab1, tab2 = st.tabs([
    "🌍 USD per tonne  (International)",
    "🇮🇳 INR per kg  (India)"
])

with tab1:
    st.caption("LME Copper Futures · Price in USD (United States Dollar) per metric tonne")
    recent_usd = df["copper"].iloc[-180:]
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=recent_usd.index,
        y=recent_usd.values,
        name="Copper (USD/tonne)",
        line=dict(color="#f0a500", width=2),
        hovertemplate="Date: %{x}<br>Price: USD %{y:,.0f}/tonne<extra></extra>"
    ))
    fig1.add_hline(
        y=forecast_usd_tonne,
        line_dash="dash",
        line_color="red",
        annotation_text=f"7-Day Forecast: USD {forecast_usd_tonne:,.0f}",
        annotation_position="bottom right"
    )
    fig1.update_layout(
        height=380,
        margin=dict(l=0, r=0, t=20, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis_title="USD per tonne",
        xaxis_title="Date"
    )
    fig1.update_yaxes(tickprefix="USD ", tickformat=",")
    st.plotly_chart(fig1, use_container_width=True)

with tab2:
    st.caption("Same LME data converted to INR (Indian Rupee ₹) per kg using live USD/INR rate")
    recent_inr = (df["copper"].iloc[-180:] / 1000) * usd_inr
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=recent_inr.index,
        y=recent_inr.values,
        name="Copper (INR/kg)",
        line=dict(color="#138808", width=2),
        hovertemplate="Date: %{x}<br>Price: ₹%{y:,.1f}/kg<extra></extra>"
    ))
    fig2.add_hline(
        y=forecast_inr_kg,
        line_dash="dash",
        line_color="red",
        annotation_text=f"7-Day Forecast: ₹{forecast_inr_kg:,.1f}/kg",
        annotation_position="bottom right"
    )
    fig2.update_layout(
        height=380,
        margin=dict(l=0, r=0, t=20, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis_title="INR per kg",
        xaxis_title="Date"
    )
    fig2.update_yaxes(tickprefix="₹ ", tickformat=",")
    st.plotly_chart(fig2, use_container_width=True)

st.divider()

# ── SECTION 5 — News Sentiment ────────────────────────────────────────────────
st.subheader("📰 Market News Sentiment")
st.caption("Latest copper market news · Sentiment analysis powered by keyword scoring")

with st.spinner("Fetching latest news..."):
    signal, headlines, score = get_sentiment()

col7, col8 = st.columns([1, 3])
col7.metric("Market Signal", signal)
col8.write("**Latest Headlines:**")

if headlines:
    for h in headlines:
        col8.write(f"• {h}")
else:
    col8.info("No headlines available. Check your NewsAPI key in sentiment.py")

st.divider()

# ── Footer ────────────────────────────────────────────────────────────────────
st.caption("""
📌 **Currency Reference:**
USD = United States Dollar · 
INR = Indian Rupee (₹) · 
All LME prices sourced in USD · 
INR prices calculated using live USD/INR forex rate · 
1 tonne = 1,000 kg

⚠️ *AI forecast is for guidance only — not financial advice. 
Actual prices may vary based on local market conditions in Jamnagar / Gujarat.*
""")