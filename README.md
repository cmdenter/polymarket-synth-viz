# Polymarket Synth Visualizer

Interactive visualization of Polymarket implied probability density functions with a sleek "Synth" style interface.

## Features

- **Simulation Mode**: Generates realistic Gaussian probability curves for demonstration
- **Live API Mode**: Connects to real Polymarket data via Python backend
- **Time-Travel Animation**: Scrub through historical probability distributions like a movie
- **Real-time Curve Rendering**: Uses Chart.js for smooth, gradient-filled probability density plots

## Quick Start (Simulation Mode)

Just open `index.html` in your browser! The app runs in simulation mode by default with no setup required.

```bash
open index.html
```

## Live API Setup

To use real Polymarket data:

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the Backend Server

```bash
python server.py
```

The server will start on `http://localhost:8000`

### 3. Open the Frontend

Open `index.html` in your browser and toggle the "Live API" switch.

### 4. Load Data

Click "Refresh" or use the `/update-data` endpoint to backfill historical data:

```bash
curl -X POST http://localhost:8000/update-data
```

## How It Works

### Frontend (`index.html`)
- React-based single-page application
- TailwindCSS for styling
- Chart.js for probability curve rendering
- Supports both simulation and live data modes

### Backend (`server.py`)
- FastAPI server for data ingestion and processing
- DuckDB for efficient time-series storage and pivoting
- Fetches Polymarket strike prices and historical data
- Calculates probability density functions using cubic spline interpolation

## Configuration

Edit `server.py` to change the market:

```python
EVENT_SLUG = "bitcoin-price-december-2024"  # Change to any Polymarket event
```

Find event slugs in Polymarket URLs, e.g.:
- `https://polymarket.com/event/bitcoin-price-december-2024`

## Architecture

```
┌─────────────┐
│  index.html │  (React UI with Chart.js)
└──────┬──────┘
       │ HTTP
       ↓
┌─────────────┐
│  server.py  │  (FastAPI + DuckDB)
└──────┬──────┘
       │
       ↓
┌─────────────────┐
│ Polymarket API  │
└─────────────────┘
```

## API Endpoints

- `GET /history` - Returns array of timestamped probability curves
- `POST /update-data` - Triggers data backfill from Polymarket

## Tech Stack

- **Frontend**: React 18, TailwindCSS, Chart.js, Babel (in-browser transpilation)
- **Backend**: FastAPI, DuckDB, Pandas, NumPy, SciPy
- **Data Source**: Polymarket CLOB and Gamma APIs

## License

MIT

## Contributing

PRs welcome! This project demonstrates:
- Time-series probability modeling
- DuckDB pivoting for financial data
- Cubic spline interpolation for probability density estimation
- Modern React patterns with hooks
