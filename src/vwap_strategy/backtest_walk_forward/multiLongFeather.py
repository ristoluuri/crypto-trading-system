import os
import sys
import json
import pandas as pd
import numpy as np
import random
from multiprocessing import Pool

# === Base paths setup ===
base_dir = os.path.dirname(os.path.abspath(__file__))  # Directory of this script
PROJECT_ROOT = os.path.abspath(os.path.join(base_dir, "..", ".."))  # Project root
sys.path.append(PROJECT_ROOT)

# Import walk-forward backtesting function (using feather-format data)
from helpers.back_test.walk_forward.bt_wf_feather import run_walk_forward

# === Constants ===
BASE_FEE_RATE = 0.00045           # Base trading fee
NUM_RANDOM_COMBINATIONS = 5       # Number of random parameter sets per run

# === Parameter ranges for randomization ===
vwap_distance_thresholds = np.arange(0.02, 0.07, 0.02).round(3).tolist()  # Entry thresholds
vwap_exit_thresholds = np.arange(0.01, 0.06, 0.02).round(3).tolist()      # Exit thresholds
max_hold_candles_list = list(range(5, 36, 10))                              # Max number of candles to hold
vwap_windows = list(range(5, 36, 10))                                       # VWAP rolling windows
fixed_sl_values = [0.01, 0.02, 0.03]                                        # Stop-loss percentages
cooldown_candle_options = list(range(3, 16, 6))                              # Cooldown after stop-loss in candles

# === Generate random strategy parameter combinations ===
def generate_random_param_combinations():
    """
    Returns a list of random strategy parameter dictionaries.
    """
    return [dict(
        vwap_distance_threshold=random.choice(vwap_distance_thresholds),
        vwap_exit_threshold=random.choice(vwap_exit_thresholds),
        max_hold_candles=random.choice(max_hold_candles_list),
        vwap_window=random.choice(vwap_windows),
        stop_loss=random.choice(fixed_sl_values),
        cooldown_candles=random.choice(cooldown_candle_options),
    ) for _ in range(NUM_RANDOM_COMBINATIONS)]


# === VWAP-based strategy implementation ===
def vwap_strategy(df, vwap_distance_threshold, vwap_exit_threshold, max_hold_candles,
                  vwap_window, stop_loss, cooldown_candles, interval_minutes, fee_rate):
    """
    Simulates trades based on VWAP strategy with stop-loss and cooldown logic.
    Returns:
        pct_gains: list of percent gains per trade
        max_drawdown: max equity drawdown in percent
        avg_hold: average holding time in hours
        number of trades executed
    """
    # Ensure numeric types and drop rows with missing values
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    df["open"] = pd.to_numeric(df["open"], errors="coerce")
    df["low"] = pd.to_numeric(df["low"], errors="coerce")
    df["high"] = pd.to_numeric(df["high"], errors="coerce")
    df.dropna(subset=["open", "high", "low", "close", "volume"], inplace=True)

    # Initialize variables
    cash, asset = 1000.0, 0.0          # Cash and asset holdings
    hold_candles = 0                    # How many candles the current position has been held
    cooldown_counter = 0                # Cooldown counter after stop-loss exit
    entry_price = 0                     # Price at which we entered the trade
    pct_gains, equity_curve, all_hold_candles = [], [], []

    # --- Compute rolling VWAP ---
    df['cum_vol'] = df['volume'].rolling(vwap_window).sum()
    df['cum_vol_x_price'] = (df['close'] * df['volume']).rolling(vwap_window).sum()
    df['vwap'] = df['cum_vol_x_price'] / df['cum_vol']
    df.dropna(inplace=True)

    open_, close, low = df['open'].values, df['close'].values, df['low'].values
    vwap = df['vwap'].values

    # --- Main loop over candles ---
    for i in range(1, len(df)):
        price = open_[i]        # Current open price
        vw = vwap[i - 1]        # Previous candle's VWAP

        # --- Handle cooldown period after stop-loss ---
        if cooldown_counter > 0:
            cooldown_counter -= 1
            current_value = asset * price if asset > 0 else cash
            equity_curve.append(current_value)
            continue

        # --- ENTRY LOGIC ---
        if asset == 0 and price < vw * (1 - vwap_distance_threshold):
            entry_price = price
            asset = (cash * (1 - fee_rate)) / entry_price  # Buy assets
            cash = 0
            hold_candles = 0

            # Immediate stop-loss check on entry candle
            stop_loss_price = entry_price * (1 - stop_loss)
            if low[i] <= stop_loss_price:
                exit_price = stop_loss_price
                cash_after = asset * exit_price * (1 - fee_rate)
                pct_gain = (exit_price * (1 - 2 * fee_rate) - entry_price) / entry_price * 100
                pct_gains.append(pct_gain)
                cash, asset, hold_candles = cash_after, 0, 0
                cooldown_counter = cooldown_candles
                equity_curve.append(cash)
                continue

        # --- EXIT LOGIC ---
        elif asset > 0:
            hold_candles += 1
            stop_loss_price = entry_price * (1 - stop_loss)

            # Determine exit price based on stop-loss, VWAP exit, or max hold
            if low[i] <= stop_loss_price:
                exit_price = stop_loss_price
            elif price > vw * (1 - vwap_exit_threshold) or hold_candles >= max_hold_candles:
                exit_price = price
            else:
                current_value = asset * price
                equity_curve.append(current_value)
                continue

            # Execute exit
            cash_after = asset * exit_price * (1 - fee_rate)
            pct_gain = (exit_price * (1 - 2 * fee_rate) - entry_price) / entry_price * 100
            pct_gains.append(pct_gain)
            all_hold_candles.append(hold_candles)
            cash, asset, hold_candles = cash_after, 0, 0
            cooldown_counter = cooldown_candles

        # --- Update equity curve ---
        current_value = asset * price if asset > 0 else cash
        equity_curve.append(current_value)

    # --- Compute max drawdown and average holding time ---
    equity_curve = np.array(equity_curve)
    max_drawdown = (
        np.max((np.maximum.accumulate(equity_curve) - equity_curve) / np.maximum.accumulate(equity_curve)) * 100
        if equity_curve.size > 0 else 0
    )
    avg_hold = np.mean(all_hold_candles) * interval_minutes / 60 if all_hold_candles else 0

    return pct_gains, max_drawdown, avg_hold, len(pct_gains)


# === Helper function: round floats in nested structures ===
def round_floats(obj, decimals=3):
    if isinstance(obj, float):
        return round(obj, decimals)
    elif isinstance(obj, dict):
        return {k: round_floats(v, decimals) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [round_floats(i, decimals) for i in obj]
    else:
        return obj


# === Wrapper to pass fee_rate to strategy for walk-forward ===
def strategy_wrapper(df, fee_rate, **kwargs):
    interval_minutes = kwargs.pop('interval_minutes', 1)
    return vwap_strategy(df, interval_minutes=interval_minutes, fee_rate=fee_rate, **kwargs)


# === Worker function for multiprocessing ===
def worker_run_walk_forward(combo, base_folder, fee_rate, param_combinations):
    """
    Calls the walk-forward backtesting function on a single symbol/interval combo.
    """
    return run_walk_forward(combo, base_folder, lambda df, **kwargs: strategy_wrapper(df, fee_rate=fee_rate, **kwargs), param_combinations)


# === Main function ===
def main():
    # Load symbol list
    csv_path = os.path.join(base_dir, PROJECT_ROOT, "symbols.csv")
    base_folder = os.path.abspath(os.path.join(base_dir, PROJECT_ROOT, "data_feather"))
    symbols = pd.read_csv(csv_path)['Symbol'].tolist()
    intervals = ['15m', '30m', '1h']
    combos = [(s, i) for s in symbols for i in intervals]

    # Generate random strategy parameters
    param_combinations = generate_random_param_combinations()

    # Load spreads for each symbol
    json_path = os.path.join(base_dir, PROJECT_ROOT, "symbol_quantities10.json")
    with open(json_path, "r") as f:
        symbol_data = json.load(f)

    spreads_dict = {s: data.get("tick_size", 0) / data.get("price", 1) for s, data in symbol_data.items()}

    # Prepare arguments for multiprocessing
    args_list = [(combo, base_folder, BASE_FEE_RATE + spreads_dict.get(combo[0], 0), param_combinations) for combo in combos]

    # Run walk-forward backtests in parallel
    with Pool(processes=5) as pool:
        results = pool.starmap(worker_run_walk_forward, args_list)

    # Process and filter results
    averaged_results = []
    for error, wf_data in results:
        if error or not wf_data:
            continue

        avg_result = wf_data["avg_result"]

        # Filter low-gain trades and insufficient fold coverage
        if avg_result.get("avg_gain_per_trade") <= 0.3:
            continue
        if avg_result.get("pct_of_folds", 0) <= 0.5:
            continue

        # Reorder results for JSON output
        reordered = {
            "symbol": avg_result.get("symbol"),
            "interval": avg_result.get("interval"),
            "num_folds": avg_result.get("num_folds")
        }
        for k in avg_result:
            if k not in ["symbol", "interval", "num_folds"]:
                reordered[k] = avg_result[k]

        averaged_results.append(round_floats(reordered))

    # Sort by average gain per trade
    averaged_results.sort(key=lambda r: r.get("avg_gain_per_trade", 0), reverse=True)

    # Save results to JSON
    output_path = os.path.join(base_dir, "longParamsFeather.json")
    with open(output_path, "w") as f:
        json.dump(averaged_results, f, indent=2)

    print(f"Saved {len(averaged_results)} averaged results to {output_path}")


if __name__ == '__main__':
    main()
