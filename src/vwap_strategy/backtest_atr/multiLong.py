import os
import json
import pandas as pd
import numpy as np
import random
from multiprocessing import Pool
import sys

# === Setup project paths for importing helper functions ===
base_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(base_dir, "..", ".."))
sys.path.append(PROJECT_ROOT)
from helpers.back_test.bt_multi import run_backtest_for

# === Constants ===
BASE_FEE_RATE = 0.00045           # Base taker fee
NUM_RANDOM_COMBINATIONS = 5       # Number of random parameter sets to generate

# === Parameter ranges for random combinations ===
vwap_exit_thresholds = np.arange(0.01, 0.06, 0.02).round(3).tolist()
max_hold_candles_list = list(range(5, 36, 10))
vwap_windows = list(range(5, 36, 10))
fixed_sl_values = [0.01, 0.02, 0.03]
cooldown_candle_options = list(range(3, 16, 6))

# === Generate random parameter combinations ===
def generate_random_param_combinations():
    """
    Generates a list of dictionaries containing random parameter sets
    for the VWAP trading strategy.
    """
    return [dict(
        vwap_exit_threshold=random.choice(vwap_exit_thresholds),
        max_hold_candles=random.choice(max_hold_candles_list),
        vwap_window=random.choice(vwap_windows),
        stop_loss=random.choice(fixed_sl_values),
        cooldown_candles=random.choice(cooldown_candle_options),
    ) for _ in range(NUM_RANDOM_COMBINATIONS)]

# === Precompute indicators (VWAP + ATR) ===
def precompute_indicators(df, atr_length=14, lookback=1000, threshold_min=0.005, threshold_max=0.055, vwap_window=14):
    """
    Precomputes VWAP, ATR, and dynamic VWAP distance threshold based on ATR.
    """
    # VWAP calculation
    df['cum_vol'] = df['volume'].rolling(vwap_window).sum()
    df['cum_vol_x_price'] = (df['close'] * df['volume']).rolling(vwap_window).sum()
    df['vwap'] = df['cum_vol_x_price'] / df['cum_vol']

    # ATR calculation
    df['h-l'] = df['high'] - df['low']
    df['h-c'] = (df['high'] - df['close'].shift()).abs()
    df['l-c'] = (df['low'] - df['close'].shift()).abs()
    df['tr'] = df[['h-l','h-c','l-c']].max(axis=1)
    df['atr'] = df['tr'].ewm(alpha=1/atr_length, adjust=False).mean()

    # ATR min/max over lookback period
    df['atr_min'] = df['atr'].rolling(lookback).min()
    df['atr_max'] = df['atr'].rolling(lookback).max()

    # Dynamic VWAP distance threshold based on ATR
    df['vwap_distance_threshold'] = threshold_min + ((df['atr'] - df['atr_min']) /
                                                      np.maximum(df['atr_max'] - df['atr_min'], 1e-8)) * (threshold_max - threshold_min)
    df.dropna(inplace=True)
    return df

# === VWAP trading strategy implementation ===
def vwap_strategy(df, vwap_exit_threshold, max_hold_candles, stop_loss, cooldown_candles, interval_minutes, fee_rate):
    """
    Executes the VWAP-based trading strategy on a precomputed DataFrame.
    Returns percent gains per trade, max drawdown, average hold time, and number of trades.
    """
    # Initialize trading state
    cash, asset = 1000.0, 0.0
    hold_candles = 0
    cooldown_counter = 0
    entry_price = 0
    min_low = None
    pct_gains, equity_curve, all_hold_candles = [], [], []

    # Extract numpy arrays for faster iteration
    open_, close, low = df['open'].values, df['close'].values, df['low'].values
    vwap = df['vwap'].values
    vw_dist_thresh_arr = df['vwap_distance_threshold'].values

    for i in range(1, len(df)):
        price = open_[i]
        vw = vwap[i-1]
        vw_dist_thresh = vw_dist_thresh_arr[i-1]
        current_low = low[i]

        # === Cooldown logic ===
        if cooldown_counter > 0:
            cooldown_counter -= 1
            equity_curve.append(asset*price if asset > 0 else cash)
            continue

        # === ENTRY condition ===
        if asset == 0 and price < vw*(1 - vw_dist_thresh):
            entry_price = price
            asset = (cash*(1 - fee_rate)) / entry_price
            cash = 0
            hold_candles = 0
            min_low = current_low

            # Check stop-loss immediately on entry candle
            stop_loss_price = entry_price*(1 - stop_loss)
            if current_low <= stop_loss_price:
                exit_price = stop_loss_price
                cash_after = asset*exit_price*(1 - fee_rate)
                pct_gain = ((exit_price*(1 - 2*fee_rate) - entry_price)/entry_price)*100
                pct_gains.append(pct_gain)
                all_hold_candles.append(0)
                cash, asset, hold_candles = cash_after, 0, 0
                cooldown_counter = cooldown_candles
                min_low = None
                equity_curve.append(cash)
                continue

        # === EXIT conditions ===
        elif asset > 0:
            hold_candles += 1
            min_low = current_low if min_low is None else min(min_low, current_low)
            stop_loss_price = entry_price*(1 - stop_loss)
            hit_stop_loss = current_low <= stop_loss_price
            exit_by_vwap = price > vw*(1 - vwap_exit_threshold)
            max_hold = hold_candles >= max_hold_candles

            if hit_stop_loss or exit_by_vwap or max_hold:
                exit_price = stop_loss_price if hit_stop_loss else price
                cash_after = asset*exit_price*(1 - fee_rate)
                pct_gain = ((exit_price*(1 - 2*fee_rate) - entry_price)/entry_price)*100
                pct_gains.append(pct_gain)
                all_hold_candles.append(hold_candles)
                cash, asset, hold_candles = cash_after, 0, 0
                min_low = None
                if hit_stop_loss:
                    cooldown_counter = cooldown_candles

        # === Update equity curve ===
        equity_curve.append(asset*price if asset > 0 else cash)

    # Calculate average hold time in hours
    avg_hold = np.mean(all_hold_candles) * interval_minutes / 60 if all_hold_candles else 0

    # Return stats: gains, max drawdown, avg hold time, number of trades
    return pct_gains, np.max((np.maximum.accumulate(np.array(equity_curve)) - np.array(equity_curve)) / np.maximum.accumulate(np.array(equity_curve)) * 100 if equity_curve else 0), avg_hold, len(pct_gains)

# === Helper to round floats in dicts/lists ===
def round_floats(obj, decimals=3):
    if isinstance(obj, float):
        return round(obj, decimals)
    elif isinstance(obj, dict):
        return {k: round_floats(v, decimals) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [round_floats(i, decimals) for i in obj]
    else:
        return obj

# === Wrapper for backtest with fee rate ===
def run_backtest_for_with_fee_rate(combo, base_folder, param_combinations, spreads_dict):
    symbol, interval = combo
    spread = spreads_dict.get(symbol, 0)
    fee_rate = BASE_FEE_RATE + spread

    def strategy_with_fee_rate(df, *args, **kwargs):
        # Precompute indicators for the strategy
        vwap_window = kwargs.pop("vwap_window", 14)
        df_pre = precompute_indicators(df, vwap_window=vwap_window)
        return vwap_strategy(df_pre, *args, fee_rate=fee_rate, **kwargs)

    # Call shared multi-backtest function
    return run_backtest_for(combo, base_folder, strategy_with_fee_rate, param_combinations)

# === Main execution ===
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, PROJECT_ROOT, "symbols.csv")
    base_folder = os.path.abspath(os.path.join(script_dir, PROJECT_ROOT, "data"))

    # Read symbols
    symbols = pd.read_csv(csv_path)['Symbol'].tolist()
    intervals = ['3m', '5m', '15m', '30m', '1h']
    combos = [(s, i) for s in symbols for i in intervals]

    # Generate random parameter sets
    param_combinations = generate_random_param_combinations()

    # Load spreads from JSON
    json_path = os.path.join(script_dir, PROJECT_ROOT, "symbol_quantities10.json")
    with open(json_path, "r") as f:
        symbol_data = json.load(f)

    spreads_dict = {}
    for symbol, data in symbol_data.items():
        try:
            spreads_dict[symbol] = data["tick_size"] / data["price"]
        except Exception:
            spreads_dict[symbol] = 0

    # === Run backtests in parallel ===
    with Pool(processes=5) as pool:
        results = pool.starmap(run_backtest_for_with_fee_rate, [
            (combo, base_folder, param_combinations, spreads_dict) for combo in combos
        ])

    # Extract successful results
    best_results = [result for error, result in results if result]
    output_path = os.path.join(script_dir, "longParams.json")

    # Sort by capital efficiency and round values
    best_results.sort(key=lambda r: r["daily_capital_efficiency"], reverse=True)
    rounded_results = [round_floats(result) for result in best_results]

    # Save results to JSON
    with open(output_path, "w") as f:
        json.dump(rounded_results, f, indent=2)

    print(f"Saved {len(best_results)} results to {output_path}")

if __name__ == '__main__':
    main()
