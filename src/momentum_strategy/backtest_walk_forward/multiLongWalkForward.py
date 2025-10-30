import os
import sys
import json
import pandas as pd
import numpy as np
import random
from multiprocessing import Pool, cpu_count

# --- Optional numba acceleration for the trade simulation ---
try:
    from numba import njit
    from numba.typed import List as NumbaList
    NUMBA_AVAILABLE = True
except Exception:
    NUMBA_AVAILABLE = False

# --- Project path setup so we can import internal modules ---
base_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(base_dir, "..", ".."))
sys.path.append(PROJECT_ROOT)
from helpers.back_test.walk_forward.bt_wf import run_walk_forward

# --- Strategy config ---
BASE_FEE_RATE = 0.00045       # baseline exchange fee (we adjust per symbol later)
NUM_RANDOM_COMBINATIONS = 1   # number of random param sets per run

# Parameter search ranges
momentum_windows = list(range(2, 42, 4))
volatility_windows = list(range(20, 420, 40))
max_hold_candles_list = list(range(5, 200, 10))
volume_factors = np.arange(2.4, 4.1, 0.2).round(2).tolist()

# --- Randomized hyperparameter sampler ---
def generate_random_param_combinations():
    return [dict(
        momentum_window=random.choice(momentum_windows),
        volatility_window=random.choice(volatility_windows),
        max_hold_candles=random.choice(max_hold_candles_list),
        volume_factor=random.choice(volume_factors)
    ) for _ in range(NUM_RANDOM_COMBINATIONS)]

# --- Fast rolling functions (NumPy) ---
def rolling_mean(arr, window):
    cumsum = np.cumsum(np.insert(arr, 0, 0))
    return (cumsum[window:] - cumsum[:-window]) / window

def rolling_std(arr, window):
    cumsum = np.cumsum(arr, dtype=float)
    cumsum_sq = np.cumsum(arr**2, dtype=float)
    mean = (cumsum[window:] - cumsum[:-window]) / window
    variance = (cumsum_sq[window:] - cumsum_sq[:-window]) / window - mean**2
    variance = np.maximum(variance, 0.0)
    return np.sqrt(variance)

# --- Trade simulation (numba accelerated if available) ---
if NUMBA_AVAILABLE:
    @njit
    def _simulate_trades_numba(open_arr, momentum_arr, can_enter_entry, fee_rate, max_hold_candles):
        pct_gains = NumbaList()
        hold_candles_list = NumbaList()
        n = open_arr.shape[0]
        i = 0

        # Walk through candles and simulate trades
        while i < n:
            if can_enter_entry[i] == 1:
                entry_price = open_arr[i]
                exit_idx = min(i + max_hold_candles, n - 1)

                # exit early if momentum flips negative
                j = 1
                while i + j <= exit_idx:
                    if momentum_arr[i + j - 1] < 0:
                        exit_idx = i + j
                        break
                    j += 1

                exit_price = open_arr[exit_idx]
                pct_gain = (exit_price * (1 - 2 * fee_rate) - entry_price) / entry_price * 100.0
                hold_len = exit_idx - i

                pct_gains.append(pct_gain)
                hold_candles_list.append(hold_len)
                i = exit_idx + 1
            else:
                i += 1

        return pct_gains, hold_candles_list

else:
    # Python fallback if numba isn't installed
    def _simulate_trades_py(open_arr, momentum_arr, can_enter_entry, fee_rate, max_hold_candles):
        pct_gains = []
        hold_candles_list = []
        n = open_arr.shape[0]
        i = 0

        while i < n:
            if can_enter_entry[i]:
                entry_price = open_arr[i]
                exit_idx = min(i + max_hold_candles, n - 1)

                neg_idx = np.where(momentum_arr[i:exit_idx] < 0)[0]
                if neg_idx.size > 0:
                    exit_idx = i + neg_idx[0] + 1

                exit_price = open_arr[exit_idx]
                pct_gain = (exit_price * (1 - 2 * fee_rate) - entry_price) / entry_price * 100
                hold_len = exit_idx - i

                pct_gains.append(pct_gain)
                hold_candles_list.append(hold_len)
                i = exit_idx + 1
            else:
                i += 1

        return pct_gains, hold_candles_list

# --- Core momentum strategy ---
def momentum_strategy_fast(open_arr, close_arr, volume_arr,
    momentum_window, volatility_window, max_hold_candles, volume_factor,
    interval_minutes, fee_rate,
    quiet_period_candles=20,
    volume_contraction_ratio=0.7,
    volatility_contraction_ratio=0.7):

    # Convert to float & Series for rolling functions
    open_arr = open_arr.astype(float)
    close_arr = close_arr.astype(float)
    volume_arr = volume_arr.astype(float)

    close_s = pd.Series(close_arr)
    volume_s = pd.Series(volume_arr)

    # Indicators
    momentum_arr = close_s.pct_change(periods=momentum_window)
    ret_arr = close_s.pct_change()
    volatility_arr = ret_arr.rolling(volatility_window, min_periods=volatility_window).std()
    avg_volume_arr = volume_s.rolling(volatility_window, min_periods=volatility_window).mean()

    recent_avg_vol_arr = volume_s.rolling(quiet_period_candles, min_periods=quiet_period_candles).mean()
    recent_volatility_arr = close_s.rolling(quiet_period_candles, min_periods=quiet_period_candles).std()

    # Filter valid rows
    valid_idx = ~(momentum_arr.isna() | volatility_arr.isna() | avg_volume_arr.isna() | recent_avg_vol_arr.isna())
    open_arr = open_arr[valid_idx]
    close_arr = close_arr[valid_idx]
    volume_arr = volume_arr[valid_idx]
    momentum_arr = momentum_arr[valid_idx].values
    avg_volume_arr = avg_volume_arr[valid_idx].values
    recent_avg_vol_arr = recent_avg_vol_arr[valid_idx].values
    recent_volatility_arr = recent_volatility_arr[valid_idx].values

    if len(open_arr) == 0:
        return [], 0, 0, 0

    # Entry logic: volume surge + green candle + momentum positive + quiet regime breakout
    is_green_prev = close_arr[:-1] > open_arr[:-1]
    is_volume_quiet_prev = recent_avg_vol_arr[:-1] < volume_contraction_ratio * avg_volume_arr[:-1]
    is_volatility_quiet_prev = recent_volatility_arr[:-1] < volatility_contraction_ratio * ret_arr.rolling(volatility_window).std().dropna()

    can_enter_prev = (momentum_arr[:-1] > 0) & \
                     (volume_arr[:-1] > volume_factor * avg_volume_arr[:-1]) & \
                     is_green_prev & \
                     (is_volume_quiet_prev | is_volatility_quiet_prev)

    # Align signals to next candle entry
    can_enter_entry = np.zeros_like(open_arr, dtype=np.int8)
    can_enter_entry[1:len(can_enter_prev)+1] = can_enter_prev.astype(np.int8)

    # Run trade simulation
    if NUMBA_AVAILABLE:
        pct_gains_nb, hold_list_nb = _simulate_trades_numba(open_arr, momentum_arr, can_enter_entry, fee_rate, max_hold_candles)
        pct_gains = np.array(pct_gains_nb)
        hold_candles = np.array(hold_list_nb)
    else:
        pct_gains_list, hold_candles_list = _simulate_trades_py(open_arr, momentum_arr, can_enter_entry, fee_rate, max_hold_candles)
        pct_gains = np.array(pct_gains_list)
        hold_candles = np.array(hold_candles_list)

    if len(pct_gains) == 0:
        return [], 0, 0, 0

    valid_trades = hold_candles > 0
    pct_gains = pct_gains[valid_trades]
    hold_candles = hold_candles[valid_trades]

    if len(pct_gains) == 0:
        return [], 0, 0, 0

    avg_hold_hours = np.mean(hold_candles) * interval_minutes / 60
    return pct_gains.tolist(), np.sum(pct_gains), avg_hold_hours, len(pct_gains)

# --- Wrapper for walk-forward module ---
def strategy_wrapper(df, fee_rate, **kwargs):
    interval_minutes = kwargs.pop('interval_minutes', 1)
    return momentum_strategy_fast(
        df['open'].values, df['close'].values, df['volume'].values,
        interval_minutes=interval_minutes,
        fee_rate=fee_rate,
        **kwargs
    )

def worker_run_walk_forward(combo, base_folder, fee_rate, param_combinations):
    return run_walk_forward(combo, base_folder,
                            lambda df, **kwargs: strategy_wrapper(df, fee_rate=fee_rate, **kwargs),
                            param_combinations)

# --- Main execution ---
def main():
    # Load symbols
    csv_path = os.path.join(PROJECT_ROOT, "symbols.csv")
    base_folder = os.path.join(PROJECT_ROOT, "data")
    symbols = pd.read_csv(csv_path)['Symbol'].tolist()
    intervals = ['15m', '30m', '1h']
    combos = [(s, i) for s in symbols for i in intervals]

    param_combinations = generate_random_param_combinations()

    # Load spreads to adjust fee per symbol
    json_path = os.path.join(PROJECT_ROOT, "symbol_quantities10.json")
    with open(json_path, "r") as f:
        symbol_data = json.load(f)

    spreads_dict = {s: data.get("tick_size", 0) / data.get("price", 1) for s, data in symbol_data.items()}

    # Create job arguments
    args_list = [
        (combo, base_folder, BASE_FEE_RATE + spreads_dict.get(combo[0], 0), param_combinations)
        for combo in combos
    ]

    # Run parallel backtests
    processes = max(1, min(cpu_count() - 1, 6))
    with Pool(processes=processes) as pool:
        results = pool.starmap(worker_run_walk_forward, args_list)

    # Collect successful strategies
    averaged_results = []
    for error, wf_data in results:
        if error or not wf_data:
            continue
        avg_result = wf_data["avg_result"]

        # Minimum performance filter
        if avg_result.get("avg_gain_per_trade", 0) <= 0.3:
            continue

        # Move symbol/interval keys to end for readability
        reordered = {k: avg_result[k] for k in avg_result if k not in ["symbol", "interval", "num_folds"]}
        reordered.update({
            "symbol": avg_result.get("symbol"),
            "interval": avg_result.get("interval"),
            "num_folds": avg_result.get("num_folds")
        })
        averaged_results.append(reordered)

    averaged_results.sort(key=lambda r: r.get("avg_gain_per_trade", 0), reverse=True)

    output_path = os.path.join(base_dir, "2910.json")
    with open(output_path, "w") as f:
        json.dump(averaged_results, f, indent=2)

    print(f"Saved {len(averaged_results)} averaged results to {output_path}")

if __name__ == "__main__":
    main()
