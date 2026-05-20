print("Script started...")

import yfinance as yf
import pandas as pd
import os

yf.set_tz_cache_location("cache")


def fetch_ticker(symbol, start="2020-01-01"):
    """Fetch price history using Ticker method"""
    try:
        print(f"  Fetching {symbol}...")
        ticker = yf.Ticker(symbol)
        data = ticker.history(start=start, auto_adjust=True)["Close"]
        if data.empty:
            print(f"  ❌ {symbol} returned empty data")
            return None
        # Remove timezone info from index to allow merging
        data.index = data.index.tz_localize(None)
        print(f"  ✅ {symbol} OK — latest: {float(data.iloc[-1]):.4f} ({len(data)} rows)")
        return data
    except Exception as e:
        print(f"  ❌ {symbol} failed: {e}")
        return None


def get_zinc(copper_tonne):
    """Try multiple zinc tickers, fallback to copper ratio if all fail"""

    for ticker_name in ["ZINC=F", "ZNC=F"]:
        data = fetch_ticker(ticker_name)
        if data is None:
            continue

        latest = float(data.iloc[-1])

        # Valid LME zinc range = USD 1,500 to 5,000 per tonne
        if 1500 < latest < 5000:
            print(f"  ✅ Zinc valid: USD {latest:,.0f}/tonne")
            return data

        # Might be in cents/lb — convert to USD/tonne
        elif 50 < latest < 250:
            converted = data * 2204 / 100
            print(f"  ✅ Zinc converted from cents/lb: USD {float(converted.iloc[-1]):,.0f}/tonne")
            return converted

        # Might be in USD/lb — convert to USD/tonne
        elif 0.5 < latest < 5:
            converted = data * 2204
            print(f"  ✅ Zinc converted from USD/lb: USD {float(converted.iloc[-1]):,.0f}/tonne")
            return converted

        else:
            print(f"  ⚠️ Zinc value {latest} out of expected range — skipping")
            continue

    # All tickers failed — use fallback
    print("  ⚠️ All zinc tickers failed — using fallback (25% of copper)")
    return copper_tonne * 0.25


def fetch_prices():
    print("\n─────────────────────────────────────")
    print("  FETCHING RAW MATERIAL PRICES")
    print("─────────────────────────────────────")

    # ── Copper ───────────────────────────────
    print("\n[1/3] Copper (HG=F):")
    copper_raw = fetch_ticker("HG=F")
    if copper_raw is None:
        print("❌ Copper fetch failed completely!")
        return None
    copper_tonne = copper_raw * 2204  # USD/lb → USD/tonne

    # ── USD/INR ──────────────────────────────
    print("\n[2/3] USD/INR Rate (INR=X):")
    usdinr = fetch_ticker("INR=X")
    if usdinr is None:
        print("⚠️ USD/INR fetch failed — using fallback rate 95.0")
        usdinr = pd.Series(95.0, index=copper_tonne.index)

    # ── Zinc ─────────────────────────────────
    print("\n[3/3] Zinc:")
    zinc = get_zinc(copper_tonne)

    # ── Combine using outer join then forward fill ────────────
    print("\nCombining data...")
    df = pd.DataFrame({
        "copper": copper_tonne,
        "zinc":   zinc,
        "usdinr": usdinr
    })

    # Forward fill to handle missing dates (weekends/holidays)
    df = df.ffill()

    # Drop only if ALL three columns are NaN
    df = df.dropna(how="all")

    # Final cleanup — drop rows where copper is missing
    df = df[df["copper"].notna()]

    print(f"  Rows after combining: {len(df)}")

    if len(df) == 0:
        print("❌ No data after combining — check internet connection")
        return None

    # ── Save ─────────────────────────────────
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/prices.csv")

    # ── Summary ──────────────────────────────
    copper_latest = float(df["copper"].iloc[-1])
    zinc_latest   = float(df["zinc"].iloc[-1])
    usdinr_latest = float(df["usdinr"].iloc[-1])

    copper_inr = (copper_latest / 1000) * usdinr_latest
    zinc_inr   = (zinc_latest   / 1000) * usdinr_latest
    brass_inr  = (copper_inr * 0.65) + (zinc_inr * 0.35)

    print(f"\n✅ SUCCESS! {len(df)} rows saved to data/prices.csv")
    print(f"─────────────────────────────────────────────────────")
    print(f"  Copper  : USD {copper_latest:>10,.0f}/tonne  |  INR {copper_inr:>8,.1f}/kg")
    print(f"  Zinc    : USD {zinc_latest:>10,.0f}/tonne  |  INR {zinc_inr:>8,.1f}/kg")
    print(f"  USD/INR :     {usdinr_latest:>10.2f}")
    print(f"─────────────────────────────────────────────────────")
    print(f"  🔩 Est. Brass Price  :  INR {brass_inr:>8,.1f}/kg")
    print(f"     (65% copper + 35% zinc)")
    print(f"─────────────────────────────────────────────────────")

    return df


if __name__ == "__main__":
    df = fetch_prices()
    if df is not None:
        print("\nLast 3 rows of saved data:")
        print(df.tail(3))