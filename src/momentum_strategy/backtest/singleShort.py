import pandas as pd
import numpy as np
import os
import sys
import json

# === Path setup (import shared backtest utils) ===
base_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(base_dir, "..", ".."))
sys.path.append(PROJECT_ROOT)

from helpers.back_test.bt_single import summarize_results  # rolled-up analytics

# === Load per-symbol execution friction (spread) ===
spread_json_path = os.path.join(base_dir, PROJECT_ROOT, "symbol_quantities10.json")
with open(spread_json_path, "r") as f:
    spread_data = json.load(f)

symbol = 'CAKEUSDT'
timeframe = '1h'

# Pull configured spread, fallback = 0
spread = float(spread_data.get(symbol, {}).get('spread', 0.0))
if spread == 0.0:
    print(f"Warning: Spread for {symbol} not found. Using 0.")

# === Strategy config ===
momentum_window = 30          # directional filter
volatility_window = 40        # volume regime estimation
volume_factor = 3.2           # breakout threshold
max_hold_candles = 5          # time-in-trade cap

# All-in trading cost: taker + modeled spread
BASE_FEE_RATE = 0.00045 + spread

# === Load historical candles ===
file_path = os.path.join(base_dir, PROJECT_ROOT, "data", symbol, f"{symbol}_{timeframe}_last1000_data.json")
file_path = os.path.abspath(file_path)
df = pd.read_json(file_path, lines=True)

df['timestamp'] = pd.to_datetime(df['timestamp'])
df.set_index('timestamp', inplace=True)
df = df.sort_index()

# === Feature construction ===
df['return'] = df['close'].pct_change()
df['momentum'] = df['close'].pct_change(periods=momentum_window)
df['volatility'] = df['return'].rolling(window=volatility_window).std()
df['avg_volume'] = df['volume'].rolling(window=volatility_window).mean()

# === Vectorized series for fast loop ===
momentum = df['momentum'].values
volatility = df['volatility'].values
volume = df['volume'].values
avg_volume = df['avg_volume'].values
open_ = df['open'].values
close = df['close'].values
timestamps = df.index.to_numpy()

# === Portfolio state ===
cash = 1000.0
short_position = 0.0
entry_price = None
entry_time = None
hold_candles = 0
positions = []
pct_gains = []

# ensure indicators warm up
start_index = max(momentum_window, volatility_window) + 1

# === Execution loop ===
for i in range(start_index, len(df)):
    mom = momentum[i - 1]
    vol_today = volume[i - 1]
    vol_avg = avg_volume[i - 1]
    price = open_[i]            # execute at next-bar open
    time = timestamps[i]
    is_red = close[i - 1] < open_[i - 1]

    # --- Entry (short breakout) ---
    if short_position == 0:
        # Down-momentum, volume expansion, red impulse
        if mom < 0 and vol_today > volume_factor * vol_avg and is_red:
            entry_price = price
            entry_time = time
            short_position = (cash * (1 - BASE_FEE_RATE)) / entry_price
            hold_candles = 0
            cash = 0

    # --- Exit logic ---
    else:
        hold_candles += 1
        # mean-reversion signal or max-hold
        if mom > 0 or hold_candles >= max_hold_candles:
            exit_price = price
            exit_time = time

            # Short PnL: value = entry - exit, apply fee
            gross_cash = short_position * (2 * entry_price - exit_price)
            cash_after = gross_cash * (1 - BASE_FEE_RATE)

            pct_gain = (entry_price - exit_price * (1 + 2 * BASE_FEE_RATE)) / entry_price * 100
            pct_gains.append(pct_gain)

            positions.append({
                'entry_time': entry_time,
                'entry_price': entry_price,
                'exit_time': exit_time,
                'exit_price': exit_price,
                'cash_before': entry_price * short_position,
                'cash_after': cash_after,
                'pct_gain': pct_gain
            })

            cash = cash_after
            short_position = 0
            entry_price = None
            entry_time = None
            hold_candles = 0

# === Mark-to-market (if position left open) ===
open_position_value = short_position * (2 * entry_price - close[-1]) if short_position > 0 else 0
equity_curve = [p['cash_after'] if p.get('cash_after') else cash for p in positions]

# === Performance summary ===
summarize_results(pct_gains, equity_curve, positions, close[-1], BASE_FEE_RATE)
