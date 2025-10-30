import pandas as pd
import numpy as np
import os
import sys
import json

# Resolve project paths for importing shared backtesting utilities
base_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(base_dir, "..", ".."))
sys.path.append(PROJECT_ROOT)
from helpers.back_test.bt_single import summarize_results

# === Load symbol spread config (execution friction per instrument) ===
spread_json_path = os.path.join(base_dir, PROJECT_ROOT, "symbol_quantities10.json")
with open(spread_json_path, "r") as f:
    spread_data = json.load(f)

symbol = 'ETCUSDT'
timeframe = '30m'

# Pull symbol-specific spread; fallback to zero if not available
if symbol in spread_data and 'spread' in spread_data[symbol]:
    spread = float(spread_data[symbol]['spread'])
else:
    print(f"Warning: Spread for {symbol} not found in JSON. Using 0.")
    spread = 0.0

# === Strategy hyperparams (picked from prior batch tests) ===
momentum_window = 34
volatility_window = 380
volume_factor = 2.4
max_hold_candles = 85

# Futures taker fee + estimated slippage
BASE_FEE_RATE = 0.00045 + spread

# === Load candle history ===
file_path = os.path.join(base_dir, PROJECT_ROOT, "data", symbol, f"{symbol}_{timeframe}_last1000_data.json")
file_path = os.path.abspath(file_path)
df = pd.read_json(file_path, lines=True)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.set_index('timestamp', inplace=True)
df = df.sort_index()  # ensure chronological order

# === Feature engineering ===
df['return'] = df['close'].pct_change()
df['momentum'] = df['close'].pct_change(periods=momentum_window)
df['volatility'] = df['return'].rolling(window=volatility_window).std()
df['avg_volume'] = df['volume'].rolling(window=volatility_window).mean()

# Convert to numpy (significant perf win in Python loops)
momentum = df['momentum'].values
volatility = df['volatility'].values
volume = df['volume'].values
avg_volume = df['avg_volume'].values
open_ = df['open'].values
close = df['close'].values
timestamps = df.index.to_numpy()

# === Portfolio + trade tracking ===
cash = 1000.0
btc = 0.0
hold_candles = 0
positions = []
pct_gains = []

# Skip warm-up period required by indicators
start_index = max(momentum_window, volatility_window) + 1

# === Execution loop ===
for i in range(start_index, len(df)):
    mom = momentum[i - 1]
    vol_today = volume[i - 1]
    vol_avg = avg_volume[i - 1]
    price = open_[i]           # execute at candle open
    time = timestamps[i]
    is_green = close[i - 1] > open_[i - 1]

    # --- Entry logic ---
    if btc == 0:
        # Regime filter: momentum + volume expansion + prior bullish candle
        if mom > 0 and vol_today > volume_factor * vol_avg and is_green:
            entry_price = price
            cash_before = cash
            btc = (cash * (1 - BASE_FEE_RATE)) / entry_price
            cash = 0
            hold_candles = 0
            positions.append({
                'type': 'BUY',
                'entry_time': time,
                'entry_price': entry_price,
                'cash_before': cash_before
            })

    # --- Exit logic ---
    else:
        hold_candles += 1
        # Momentum decay or max time-in-trade
        if mom < 0 or hold_candles >= max_hold_candles:
            exit_price = price
            exit_time = time
            cash_after = btc * exit_price * (1 - BASE_FEE_RATE)

            # Realised P&L (double fee since round trip)
            pct_gain = (exit_price * (1 - 2 * BASE_FEE_RATE) - positions[-1]['entry_price']) / positions[-1]['entry_price'] * 100
            pct_gains.append(pct_gain)

            positions[-1].update({
                'type': 'SELL',
                'exit_time': exit_time,
                'exit_price': exit_price,
                'cash_after': cash_after,
                'pct_gain': pct_gain
            })

            cash = cash_after
            btc = 0
            hold_candles = 0

# === Final equity mark-to-market ===
final_value = btc * close[-1] if btc > 0 else cash

# === Summaries (PnL, equity curve, trade list) ===
equity_curve = [pos['cash_after'] if pos.get('cash_after') else cash for pos in positions]
summarize_results(pct_gains, equity_curve, positions, close[-1], BASE_FEE_RATE)
