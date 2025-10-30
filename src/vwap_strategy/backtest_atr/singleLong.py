import os
import pandas as pd
import numpy as np
import sys
import json

# === Setup project paths for helper functions ===
base_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(base_dir, "..", ".."))
sys.path.append(PROJECT_ROOT)
from helpers.back_test.bt_single import summarize_results

# === Load spread data from JSON ===
spread_json_path = os.path.join(base_dir, PROJECT_ROOT, "symbol_quantities10.json")
with open(spread_json_path, "r") as f:
    spread_data = json.load(f)

# Symbol and timeframe to trade
symbol = 'FIOUSDT'
timeframe = '1h'

# Get spread for the symbol, default to 0 if not found
spread = float(spread_data.get(symbol, {}).get('spread', 0.0))

# === Fee rate (taker fee + spread) ===
BASE_FREE_RATE = 0.00045 + spread

# === Load historical OHLCV data ===
file_path = os.path.join(base_dir, PROJECT_ROOT, "data", symbol, f"{symbol}_{timeframe}_last1000_data.json")
df = pd.read_json(file_path, lines=True)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.set_index('timestamp', inplace=True)
df = df.sort_index()

# === Strategy parameters ===
vwap_exit_threshold = 0.01   # VWAP threshold to exit
max_hold_candles = 15        # Max candles to hold a trade
vwap_window = 25             # Rolling window for VWAP calculation
stop_loss = 0.02             # Stop-loss percentage
cooldown_candles = 15        # Candles to wait after stop-loss hit

# === ATR parameters for dynamic VWAP distance threshold ===
atr_length = 14              # EMA length for ATR calculation
lookback = 1000              # Lookback period for ATR min/max
threshold_min = 0.005        # Minimum VWAP distance threshold
threshold_max = 0.055        # Maximum VWAP distance threshold

# === VWAP Calculation ===
df['cum_vol'] = df['volume'].rolling(vwap_window).sum()
df['cum_vol_x_price'] = (df['close'] * df['volume']).rolling(vwap_window).sum()
df['vwap'] = df['cum_vol_x_price'] / df['cum_vol']

# === ATR Calculation (True Range based) ===
df['h-l'] = df['high'] - df['low']
df['h-c'] = (df['high'] - df['close'].shift()).abs()
df['l-c'] = (df['low'] - df['close'].shift()).abs()
df['tr'] = df[['h-l', 'h-c', 'l-c']].max(axis=1)
df['atr'] = df['tr'].ewm(alpha=1/atr_length, adjust=False).mean()

# Calculate ATR min/max over lookback period
df['atr_min'] = df['atr'].rolling(lookback).min()
df['atr_max'] = df['atr'].rolling(lookback).max()

# Dynamic VWAP distance threshold based on ATR
df['vwap_distance_threshold'] = threshold_min + ((df['atr'] - df['atr_min']) / 
                                                  np.maximum(df['atr_max'] - df['atr_min'], 1e-8)) * (threshold_max - threshold_min)
df.dropna(inplace=True)

# === Prepare numpy arrays for faster iteration ===
open_, close, low = df['open'].values, df['close'].values, df['low'].values
vwap = df['vwap'].values
vw_dist_thresh_arr = df['vwap_distance_threshold'].values
timestamps = df.index.to_numpy()

# === Strategy state variables ===
cash = 1000.0           # Starting capital
asset = 0.0             # Asset amount held
hold_candles = 0        # Number of candles position has been held
cooldown_counter = 0    # Counter for cooldown after stop-loss
entry_price = 0         # Price at entry
min_low = None          # Minimum low during current trade

pct_gains = []          # List of percent gains per trade
equity_curve = []       # Portfolio value over time
all_hold_candles = []   # Track hold duration of each trade
positions = []          # Track trade details

# === Main Strategy Loop ===
for i in range(1, len(df)):
    price = open_[i]
    vw = vwap[i-1]
    vw_dist_thresh = vw_dist_thresh_arr[i-1]
    current_low = low[i]
    time = timestamps[i]

    # --- Cooldown handling ---
    if cooldown_counter > 0:
        cooldown_counter -= 1
        equity_curve.append(asset*price if asset > 0 else cash)
        continue

    # --- ENTRY LOGIC ---
    if asset == 0:
        if price < vw*(1 - vw_dist_thresh):  # Price below VWAP threshold triggers buy
            entry_price = price
            cash_before = cash
            asset = (cash*(1 - BASE_FREE_RATE)) / entry_price
            cash = 0
            hold_candles = 0
            min_low = current_low

            # Log position details
            positions.append({
                'type': 'BUY',
                'entry_time': time,
                'entry_price': entry_price,
                'cash_before': cash_before,
                'entry_vwap': vw,
                'entry_threshold': vw_dist_thresh,
                'min_low_during_trade': 0.0
            })

            # Immediate stop-loss check on entry candle
            stop_loss_price = entry_price*(1 - stop_loss)
            if current_low <= stop_loss_price:
                exit_price = stop_loss_price
                cash_after = asset*exit_price*(1 - BASE_FREE_RATE)
                pct_gain = ((exit_price*(1 - 2*BASE_FREE_RATE) - entry_price)/entry_price)*100
                pct_gains.append(pct_gain)
                all_hold_candles.append(0)
                # Update position with exit info
                positions[-1].update({
                    'type': 'SELL',
                    'exit_time': time,
                    'exit_price': exit_price,
                    'cash_after': cash_after,
                    'pct_gain': pct_gain,
                    'exit_vwap': vw
                })
                cash, asset, hold_candles = cash_after, 0, 0
                cooldown_counter = cooldown_candles
                equity_curve.append(cash)
                min_low = None
                continue

    # --- EXIT LOGIC ---
    else:
        hold_candles += 1
        min_low = current_low if min_low is None else min(min_low, current_low)
        min_low_pct = (min_low - entry_price)/entry_price*100
        positions[-1]['min_low_during_trade'] = round(min_low_pct, 3)

        # Conditions to exit: stop-loss, VWAP exit, max hold
        stop_loss_price = entry_price*(1 - stop_loss)
        hit_stop_loss = current_low <= stop_loss_price
        exit_by_vwap = price > vw*(1 - vwap_exit_threshold)
        max_hold = hold_candles >= max_hold_candles

        if hit_stop_loss or exit_by_vwap or max_hold:
            exit_price = stop_loss_price if hit_stop_loss else price
            cash_after = asset*exit_price*(1 - BASE_FREE_RATE)
            pct_gain = ((exit_price*(1 - 2*BASE_FREE_RATE) - entry_price)/entry_price)*100
            pct_gains.append(pct_gain)
            all_hold_candles.append(hold_candles)
            # Update position with exit info
            positions[-1].update({
                'type': 'SELL',
                'exit_time': time,
                'exit_price': exit_price,
                'cash_after': cash_after,
                'pct_gain': pct_gain,
                'exit_vwap': vw
            })
            cash, asset, hold_candles = cash_after, 0, 0
            min_low = None
            if hit_stop_loss:
                cooldown_counter = cooldown_candles

    # --- Update equity curve ---
    equity_curve.append(asset*price if asset > 0 else cash)

# === Summarize results ===
summarize_results(pct_gains, equity_curve, positions, close[-1], BASE_FREE_RATE)
