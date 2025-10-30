# Crypto Trading Research & Execution System

This repository contains my work on developing, testing, and running algorithmic trading strategies for cryptocurrency markets on Binance Futures.
The goal is to build robust, data-driven momentum/VWAP-based systems using Python and deploy them efficiently.

# Features

- Download and convert large historical datasets (JSON → Feather)
- Custom backtester with position management logic
- Walk-forward simulation and parameter optimization
- Live execution bot (async + multiprocessing)
- Hedge mode support for Binance Futures
- Spread- and fee-adjusted PnL models
- Google Sheets integration for logging results

# Tech Stack

Language & Core

Python (NumPy, Pandas)

## Data

Feather format (Apache Arrow)

JSON line format parsing

## Concurrency

asyncio

multiprocessing

WebSockets

## Exchange APIs

Binance Futures API

## Logging / Output

- Google Sheets API (for analysis & tracking)
- Local output:
  - JSON files (detailed trade/backtest data)
  - `.log` files (live trading logs)

# Running the Project

Install dependencies

```bash
pip install -r requirements.txt

# Set environment variables

Make a .env file containing:
BINANCE_API_KEY=your_key_here
BINANCE_API_SECRET=your_secret_here

# Project Structure
src/
 ├─ google_sheets/        # Writes backtesting results to google sheets
 ├─ data/                 # Raw JSON market data
 ├─ data_feather/         # Converted high-performance feather files
 ├─ momentum_strategy/    # Backtesting and live bot for momentum-based strategies
 ├─ vwap_strategy/        # Backtesting and live bot for VWAP-based strategies
 ├─ helpers/              # Shared utilities & logging
 └─ scripts/               # Utility scripts to fetch market data from Binance (e.g., trading pairs, symbol info)

## About Me

Self-taught developer building automated crypto systems.  
8 years competitive esports experience → applied to systematic trading discipline.  
Currently exploring market micro-structure and high-frequency execution bottlenecks.





