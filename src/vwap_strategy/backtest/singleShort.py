import os
import pandas as pd
import numpy as np
import sys
import json

# === Setup project paths for importing helper functions ===
base_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(base_dir, "..", ".."))
sys.path.append(PROJECT_ROOT)
from helpers.back_test.bt_single import summarize_results

# === Load symbol spreads from JSON ===
spread_json_path = os.path.join(base_dir, PROJECT_ROOT, "symbol_quantities10.json")
with open(spread_json_path, "r") as f:
    spread_data = json.load(f)

symbol = 'FISUSDT'
timeframe = '30m'

# Retrieve spread for the symbol; fallback to 0 if not found
if symbol in spread_data and 'spread' in spread_data[symbol]:
    spread = float(spread_data[symbol]['spread'])
else:
    print(f"Warning: Spread for {symbol} not found in JSON. Using 0.")
    spread = 0.0

# Combine base taker fee and spread for total trading fee rate
BASE_FREE_RATE = 0.00045 + spread

# === Load historical OHLCV price data ===
file_path = os.path.join(base_dir, PROJECT_ROOT, "data", symbol, f"{symbol}_{timeframe}_last1000_data.json")
df = pd.read_json(file_path, lines=True)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.set_index('timestamp', inplace=True)
df = df.sort_index()

# === VWAP strategy parameters ===
vwap_distance_threshold = 0.02  # Price deviation from VWAP to enter a short
vwap_exit_threshold = 0.005     # Price deviation from VWAP to exit a short
max_hold_candles = 15           # Maximum number of candles to hold a short
vwap_window = 15                # Rolling window for VWAP calculation
stop_loss = 0.03                # Stop-loss percentage
cooldown_candles = 5            # Cooldown period after stop-loss triggered

# === Calculate VWAP ===
df['cum_vol'] = df['volume'].rolling(vwap_window).sum()                  # Cumulative volume
df['cum_vol_x_price'] = (df['close'] * df['volume']).rolling(vwap_window).sum()  # VWAP numerator
df['vwap'] = df['cum_vol_x_price'] / df['cum_vol']                        # VWAP
df.dropna(inplace=True)

# === Prepare numpy arrays for faster iteration ===
open_ = df['open'].values
close = df['close'].values
high = df['high'].values
vwap = df['vwap'].values
timestamps = df.index.to_numpy()

# === Initialize trading variables ===
cash = 1000.0           # Starting capital
short_position = 0.0    # Current size of short position
hold_candles = 0        # Number of candles the current trade has been held
positions = []          # List to store details of all trades
pct_gains = []          # Percent gains per trade
equity_curve = []       # Track equity over time
max_high = None         # Track maximum high during a short trade
cooldown_counter = 0    # Cooldown counter after stop-loss exit

# === Main VWAP shorting strategy loop ===
for i in range(1, len(open_)):
    price = open_[i]            # Current open price
    time = timestamps[i]        # Current timestamp
    vw = vwap[i - 1]            # Previous candle VWAP
    current_high = high[i]      # Current high price

    # === Handle cooldown period after stop-loss exit ===
    if cooldown_counter > 0:
        cooldown_counter -= 1
        current_value = cash + short_position * (entry_price - price) if short_position > 0 else cash
        equity_curve.append(current_value)
        continue

    # === ENTRY LOGIC for SHORT ===
    if short_position == 0:
        # Enter short if price is sufficiently above VWAP
        if price > vw * (1 + vwap_distance_threshold):
            entry_price = price
            cash_before = cash
            short_position = cash / entry_price  # Short size in units
            hold_candles = 0
            max_high = current_high
            max_high_pct = ((max_high - entry_price) / entry_price) * 100

            # Record trade entry
            positions.append({
                'type': 'SHORT',
                'entry_time': time,
                'entry_price': entry_price,
                'cash_before': cash_before,
                'entry_vwap': vw,
                'max_high_during_trade': round(max_high_pct, 3)
            })

            # 🔹 Immediate stop-loss check on entry candle
            stop_loss_price = entry_price * (1 + stop_loss)
            if high[i] >= stop_loss_price:
                # Exit immediately if stop-loss hit
                exit_time = time
                exit_price = stop_loss_price
                cash_after = cash + short_position * entry_price - short_position * exit_price * (1 + 2 * BASE_FREE_RATE)
                pct_gain = ((entry_price - exit_price) / entry_price - 2 * BASE_FREE_RATE) * 100

                pct_gains.append(pct_gain)
                # Update trade record with exit info
                positions[-1].update({
                    'type': 'COVER',
                    'exit_time': exit_time,
                    'exit_price': exit_price,
                    'cash_after': cash_after,
                    'pct_gain': pct_gain,
                    'exit_vwap': vw
                })

                # Reset trading state
                cash = cash_after
                short_position = 0
                hold_candles = 0
                max_high = None
                cooldown_counter = cooldown_candles
                equity_curve.append(cash)
                continue

    # === EXIT LOGIC for SHORT ===
    else:
        hold_candles += 1
        # Track maximum high during trade for reference
        if max_high is None or current_high > max_high:
            max_high = current_high
            max_high_pct = (max_high - entry_price) / entry_price * 100
            positions[-1]['max_high_during_trade'] = round(max_high_pct, 3)

        # Determine exit conditions
        stop_loss_price = entry_price * (1 + stop_loss)
        hit_stop_loss = high[i] >= stop_loss_price
        exit_by_vwap = price < vw * (1 + vwap_exit_threshold)
        max_hold = hold_candles >= max_hold_candles

        if hit_stop_loss or exit_by_vwap or max_hold:
            exit_time = time
            exit_price = stop_loss_price if hit_stop_loss else price
            cash_after = cash + short_position * entry_price - short_position * exit_price * (1 + 2 * BASE_FREE_RATE)
            pct_gain = ((entry_price - exit_price) / entry_price - 2 * BASE_FREE_RATE) * 100

            pct_gains.append(pct_gain)
            # Update trade record with exit info
            positions[-1].update({
                'type': 'COVER',
                'exit_time': exit_time,
                'exit_price': exit_price,
                'cash_after': cash_after,
                'pct_gain': pct_gain,
                'exit_vwap': vw
            })

            # Reset trading state
            cash = cash_after
            short_position = 0
            hold_candles = 0
            max_high = None

            # Apply cooldown only if stop-loss triggered
            if hit_stop_loss:
                cooldown_counter = cooldown_candles

    # === Update equity curve at the end of each candle ===
    current_value = cash + short_position * (entry_price - price) if short_position > 0 else cash
    equity_curve.append(current_value)

# === Summarize trade results ===
summarize_results(pct_gains, equity_curve, positions, close[-1], BASE_FREE_RATE)
