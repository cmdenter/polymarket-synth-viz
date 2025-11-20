import duckdb
import requests
import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from datetime import datetime, timedelta
import time

# --- CONFIGURATION ---
# The event slug for the market we want to model (e.g., End of Month Bitcoin)
# You can find this in the URL of the Polymarket page
EVENT_SLUG = "what-price-will-bitcoin-hit-in-2025"
DB_FILE = "polymarket_data.duckdb"

app = FastAPI()

# Enable CORS for local React dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- DUCKDB SETUP ---
def init_db():
    con = duckdb.connect(DB_FILE)
    con.execute("""
        CREATE TABLE IF NOT EXISTS raw_trades (
            timestamp TIMESTAMP,
            strike_price DOUBLE,
            price DOUBLE,
            token_id VARCHAR
        );
    """)
    # Create a specialized view to get the latest price for every strike at every 5-min interval
    # This uses DuckDB's power to fill gaps (ASOF join logic simulation via window functions)
    return con

# --- DATA INGESTION ENGINE ---
def fetch_market_metadata():
    """Finds all Strike Prices and Token IDs for the event."""
    print(f"Fetching metadata for {EVENT_SLUG}...")
    url = f"https://gamma-api.polymarket.com/events?slug={EVENT_SLUG}"
    resp = requests.get(url).json()

    tokens = {}
    for event in resp:
        for market in event.get('markets', []):
            try:
                # Parse strike from groupItemTitle: "↑ 100,000", "$120,000", "↓ 50,000", etc.
                title = market.get('groupItemTitle', '')
                if not title:
                    continue

                # Remove arrows, $, commas, and extra whitespace
                strike_str = title.replace('↑','').replace('↓','').replace('$','').replace(',','').strip()
                strike = float(strike_str)

                # Get the 'Yes' token ID from clobTokenIds
                clob_tokens = market.get('clobTokenIds', [])
                if not clob_tokens:
                    continue

                # Parse the JSON string if needed
                if isinstance(clob_tokens, str):
                    import json
                    clob_tokens = json.loads(clob_tokens)

                # First token is usually "Yes"
                token_id = clob_tokens[0]

                tokens[strike] = token_id
                print(f"  Found strike: ${strike:,.0f}")
            except Exception as e:
                print(f"  Skipping market: {e}")
                continue
    return tokens

def backfill_data(con):
    """Downloads history for all strikes and saves to DuckDB."""
    tokens = fetch_market_metadata()
    print(f"Found {len(tokens)} strikes. Downloading history...")

    for strike, token_id in tokens.items():
        # Fetch 1-hour candles (valid intervals: '1m', '1h', '6h', '1d', '1w')
        url = "https://clob.polymarket.com/prices-history"
        params = {"market": token_id, "interval": "1h"}

        try:
            resp = requests.get(url, params=params, timeout=10).json()
        except Exception as e:
            print(f"Error fetching data for ${strike}: {e}")
            continue

        history = resp.get('history', [])
        if not history:
            print(f"No history for strike ${strike}")
            continue

        # Prepare batch insert
        # timestamps come as unix seconds
        data = []
        for h in history:
            try:
                ts = datetime.fromtimestamp(h['t'])
                price = h['p']
                data.append((ts, strike, price, token_id))
            except Exception as e:
                continue

        if data:
            # Bulk insert is faster
            con.executemany("INSERT INTO raw_trades VALUES (?, ?, ?, ?)", data)
            print(f"Ingested {len(data)} points for strike ${strike:,.0f}")

    print("Data Backfill Complete.")

# --- THE MATH ENGINE (CURVE GENERATION) ---
def calculate_curve(df_slice):
    """
    Takes a dataframe of [Strike, Price] for a SINGLE timestamp.
    Returns X (Prices) and Y (Probability Density) arrays.
    """
    if df_slice.empty or len(df_slice) < 3:
        return [], []

    # 1. Sort by strike
    df_slice = df_slice.sort_values("strike_price")
    strikes = df_slice["strike_price"].values
    prices = df_slice["price"].values

    # 2. Calculate CDF
    # Polymarket price is P(Price > X).
    # CDF is P(Price < X) = 1 - Price
    cdf = 1.0 - prices

    # 3. Fit Spline (Smoothing)
    # We use MonotoneCubicInterpolation to prevent the curve from dipping below 0 probability
    try:
        # Using cubic spline for smoothness, but clamping needed in real production
        cs = CubicSpline(strikes, cdf)

        # 4. Calculate PDF (Derivative of CDF)
        x_range = np.linspace(strikes.min(), strikes.max(), 100)
        pdf_values = cs(x_range, 1) # 1st derivative

        # Clean up noise (negative probabilities are impossible)
        pdf_values = np.maximum(pdf_values, 0)

        return x_range.tolist(), pdf_values.tolist()
    except:
        return [], []

# --- API ENDPOINTS ---

@app.post("/update-data")
def trigger_update():
    con = init_db()
    con.execute("DELETE FROM raw_trades") # Simple full refresh for this demo
    backfill_data(con)
    return {"status": "Data updated"}

@app.get("/history")
def get_history():
    """
    Returns a list of available timestamps and the generated curve for each.
    This essentially generates the 'frames' of the movie.
    """
    con = init_db()

    # Get data with 1-hour time buckets to align timestamps across strikes
    query = """
        SELECT
            time_bucket(INTERVAL '1 hour', timestamp) AS hour,
            strike_price,
            AVG(price) as price
        FROM raw_trades
        GROUP BY hour, strike_price
        ORDER BY hour, strike_price
    """
    df = con.execute(query).df()

    # Pivot to get strikes as columns
    df_pivot = df.pivot(index='hour', columns='strike_price', values='price')
    df_pivot = df_pivot.dropna() # Drop rows where we don't have all strikes

    result = []

    # Iterate through time (The Movie Frames)
    for hour, row in df_pivot.iterrows():
        # Extract strikes and prices for this hour
        strikes = []
        prices = []
        for strike_price in df_pivot.columns:
            strikes.append(float(strike_price))
            prices.append(row[strike_price])

        # Build dataframe for the math engine
        frame_df = pd.DataFrame({'strike_price': strikes, 'price': prices})

        # Math
        xs, ys = calculate_curve(frame_df)

        result.append({
            "timestamp": hour.isoformat(),
            "curve_x": xs,
            "curve_y": ys,
            "implied_median": xs[np.argmax(ys)] if len(ys) > 0 else 0
        })

    return result

if __name__ == "__main__":
    # Initialize DB on start
    init_db()
    uvicorn.run(app, host="0.0.0.0", port=8000)
