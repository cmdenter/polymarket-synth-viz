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
EVENT_SLUG = "bitcoin-price-december-2024"
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
                # Parse "Bitcoin > $95,000" -> 95000.0
                title = market['groupItemTitle']
                strike = float(title.replace(',','').replace('>','').replace('$','').strip())

                # Get the 'Yes' token ID
                # Outcomes is usually ["Yes", "No"] or ["No", "Yes"]
                yes_idx = 0 if market['outcomes'][0] == "Yes" else 1
                token_id = market['clobTokenIds'][yes_idx]

                tokens[strike] = token_id
            except Exception as e:
                continue
    return tokens

def backfill_data(con):
    """Downloads history for all strikes and saves to DuckDB."""
    tokens = fetch_market_metadata()
    print(f"Found {len(tokens)} strikes. Downloading history...")

    for strike, token_id in tokens.items():
        # Fetch 5-min candles
        url = "https://clob.polymarket.com/prices-history"
        params = {"market": token_id, "interval": "5m", "fidelity": 5}
        resp = requests.get(url, params=params).json()

        history = resp.get('history', [])
        if not history: continue

        # Prepare batch insert
        # timestamps come as unix seconds
        data = []
        for h in history:
            ts = datetime.fromtimestamp(h['t'])
            price = h['p']
            data.append((ts, strike, price, token_id))

        # Bulk insert is faster
        con.executemany("INSERT INTO raw_trades VALUES (?, ?, ?, ?)", data)
        print(f"Ingested {len(data)} points for strike ${strike}")

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

    # DuckDB PIVOT: The secret weapon.
    # Transforms raw rows into a Matrix: Time x Strike1 x Strike2 ...
    query = """
        PIVOT raw_trades
        ON strike_price
        USING last(price)
        GROUP BY timestamp
        ORDER BY timestamp ASC
    """
    df_pivot = con.execute(query).df()
    df_pivot = df_pivot.dropna() # Simple drop for demo, ideally ffill

    result = []

    # Iterate through time (The Movie Frames)
    for index, row in df_pivot.iterrows():
        ts = row['timestamp']

        # Extract strikes and prices for this moment
        # Filter out the timestamp column
        strikes = []
        prices = []
        for col in df_pivot.columns:
            if col != 'timestamp':
                strikes.append(float(col))
                prices.append(row[col])

        # Build dataframe for the math engine
        frame_df = pd.DataFrame({'strike_price': strikes, 'price': prices})

        # Math
        xs, ys = calculate_curve(frame_df)

        result.append({
            "timestamp": ts.isoformat(),
            "curve_x": xs,
            "curve_y": ys,
            "implied_median": xs[np.argmax(ys)] if len(ys) > 0 else 0
        })

    return result

if __name__ == "__main__":
    # Initialize DB on start
    init_db()
    uvicorn.run(app, host="0.0.0.0", port=8000)
