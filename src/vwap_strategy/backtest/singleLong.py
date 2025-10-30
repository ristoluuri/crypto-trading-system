import os
import pandas as pd
import numpy as np
import sys
import json

# === Setup project paths for imports ===
base_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(base_dir, "..", ".."))
sys.path.append(PROJECT_ROOT)
from helpers.back_test.bt_single import summarize_results

# === Load symbol spreads from JSON ===
spread_json_path = os.path.join(base_dir, PROJECT_ROOT, "symbol_quantities10.json")
with open(spread_json_path, "r") as f:
    spread_data = json.load(f)

symbol = 'SPXUSDT'
timeframe = '1h'

# Retrieve spread for the symbol; fallback to 0 if not found
if symbol in spread_data and 'spread' in spread_data[symbol]:
    spread = float(spread_data[symbol]['spread'])
else:
    print(f"Warning: Spread for {symbol} not found in JSON. Using 0.")
    spread = 0.0

# Combine base taker fee and spread to calculate total trading fee rate
BASE_FREE_RATE = 0.00045 + spread

# === Load historical OHLCV data ===
file_path = os.path.join(base_dir, PROJECT_ROOT, "data", symbol, f"{symbol}_{timeframe}_last1000_data.json")
df = pd.read_json(file_path, lines=True)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.set_index('timestamp', inplace=True)
df = df.sort_index()

# === VWAP strategy parameters ===
vwap_distance_threshold = 0.06  # Entry threshold: how far below VWAP to buy
vwap_exit_threshold = 0.05      # Exit threshold: price relative to VWAP for profit taking
max_hold_candles = 33           # Maximum candles to hold a trade
vwap_window = 25                # VWAP rolling window
stop_loss = 0.005               # Stop-loss percentage
cooldown_candles = 9            # Cooldown period after stop-loss exit

# === Calculate VWAP ===
df['cum_vol'] = df['volume'].rolling(vwap_window).sum()
df['cum_vol_x_price'] = (df['close'] * df['volume']).rolling(vwap_window).sum()
df['vwap'] = df['cum_vol_x_price'] / df['cum_vol']
df.dropna(inplace=True)

# === Prepare numpy arrays for faster iteration ===
open_ = df['open'].values
close = df['close'].values
low = df['low'].values
vwap = df['vwap'].values
timestamps = df.index.to_numpy()

# === Initialize trading variables ===
cash = 1000.0           # Starting capital
btc = 0.0               # Current position size in asset
hold_candles = 0        # Number of candles current trade has been held
positions = []          # Record of all trades
pct_gains = []          # Percent gains per trade
equity_curve = []       # Track equity over time
min_low = None          # Track lowest price during a trade
cooldown_counter = 0    # Cooldown counter after stop-loss exit

# === Main strategy loop ===
for i in range(1, len(open_)):
    price = open_[i]
    time = timestamps[i]
    vw = vwap[i - 1]
    current_low = low[i]

    # === Handle cooldown period after stop-loss exit ===
    if cooldown_counter > 0:
        cooldown_counter -= 1
        current_value = cash if btc == 0 else btc * price
        equity_curve.append(current_value)
        continue

    # === ENTRY LOGIC ===
    if btc == 0:
        # Buy if price is sufficiently below VWAP
        if price < vw * (1 - vwap_distance_threshold):
            entry_price = price
            cash_before = cash
            btc = (cash * (1 - BASE_FREE_RATE)) / entry_price  # Apply fees
            cash = 0
            hold_candles = 0
            min_low = current_low
            min_low_pct = ((min_low - entry_price) / entry_price) * 100

            # Record trade entry
            positions.append({
                'type': 'BUY',
                'entry_time': time,
                'entry_price': entry_price,
                'cash_before': cash_before,
                'entry_vwap': vw,
                'min_low_during_trade': round(min_low_pct, 3)
            })

            # 🔹 Immediate stop-loss check on entry candle
            stop_loss_price = entry_price * (1 - stop_loss)
            if low[i] <= stop_loss_price:
                exit_time = time
                exit_price = stop_loss_price
                cash_after = btc * exit_price * (1 - BASE_FREE_RATE)
                pct_gain = ((exit_price * (1 - 2 * BASE_FREE_RATE) - entry_price) / entry_price) * 100

                pct_gains.append(pct_gain)
                # Update trade record with exit info
                positions[-1].update({
                    'type': 'SELL',
                    'exit_time': exit_time,
                    'exit_price': exit_price,
                    'cash_after': cash_after,
                    'pct_gain': pct_gain,
                    'exit_vwap': vw
                })

                # Reset trading state
                cash = cash_after
                btc = 0
                hold_candles = 0
                min_low = None
                cooldown_counter = cooldown_candles
                equity_curve.append(cash)
                continue  # Skip to next candle

    # === EXIT LOGIC ===
    else:
        hold_candles += 1
        # Track minimum low during trade
        if min_low is None or current_low < min_low:
            min_low = current_low
            min_low_pct = (min_low - entry_price) / entry_price * 100
            positions[-1]['min_low_during_trade'] = round(min_low_pct, 3)

        # Check exit conditions
        stop_loss_price = entry_price * (1 - stop_loss)
        hit_stop_loss = low[i] <= stop_loss_price
        exit_by_vwap = price > vw * (1 - vwap_exit_threshold)
        max_hold = hold_candles >= max_hold_candles

        if hit_stop_loss or exit_by_vwap or max_hold:
            exit_time = time
            exit_price = stop_loss_price if hit_stop_loss else price
            cash_after = btc * exit_price * (1 - BASE_FREE_RATE)
            pct_gain = ((exit_price * (1 - 2 * BASE_FREE_RATE) - entry_price) / entry_price) * 100

            pct_gains.append(pct_gain)
            # Update trade record with exit info
            positions[-1].update({
                'type': 'SELL',
                'exit_time': exit_time,
                'exit_price': exit_price,
                'cash_after': cash_after,
                'pct_gain': pct_gain,
                'exit_vwap': vw
            })

            # Reset trading state
            cash = cash_after
            btc = 0
            hold_candles = 0
            min_low = None

            # Apply cooldown only on stop-loss
            if hit_stop_loss:
                cooldown_counter = cooldown_candles

    # === Update equity curve ===
    current_value = cash if btc == 0 else btc * price * (1 - BASE_FREE_RATE)
    equity_curve.append(current_value)

# === Summarize results ===
summarize_results(pct_gains, equity_curve, positions, close[-1], BASE_FREE_RATE)
