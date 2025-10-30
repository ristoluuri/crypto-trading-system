import os
import json
import pandas as pd
import numpy as np
import random
from multiprocessing import Pool
import sys

# === Project paths ===
base_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(base_dir, "..", ".."))
sys.path.append(PROJECT_ROOT)

from helpers.back_test.bt_multi import run_backtest_for

# Base Binance futures fee + per-symbol spread adjustments later
BASE_FEE_RATE = 0.00045

# Number of random parameter sets to test per symbol/interval combo
NUM_RANDOM_COMBINATIONS = 5

# Parameter search space for VWAP strategy tuning
vwap_distance_thresholds = np.arange(0.005, 0.05, 0.01).round(3).tolist()
vwap_exit_thresholds     = np.arange(0.01, 0.06, 0.02).round(3).tolist()
max_hold_candles_list    = list(range(5, 36, 10))
vwap_windows             = list(range(5, 36, 10))
fixed_sl_values          = [0.01, 0.02, 0.03]
cooldown_candle_options  = list(range(3, 16, 6))

# Generate random hyperparameter sets for testing
def generate_random_param_combinations():
    return [
        dict(
            vwap_distance_threshold=random.choice(vwap_distance_thresholds),
            vwap_exit_threshold=random.choice(vwap_exit_thresholds),
            max_hold_candles=random.choice(max_hold_candles_list),
            vwap_window=random.choice(vwap_windows),
            stop_loss=random.choice(fixed_sl_values),
            cooldown_candles=random.choice(cooldown_candle_options),
        )
        for _ in range(NUM_RANDOM_COMBINATIONS)
    ]

# === VWAP strategy backtester ===
def vwap_strategy(df, vwap_distance_threshold, vwap_exit_threshold, max_hold_candles,
                  vwap_window, stop_loss, cooldown_candles, interval_minutes, fee_rate):

    # Portfolio state
    cash, asset = 1000.0, 0.0
    hold_candles = 0
    cooldown_counter = 0
    entry_price = 0

    pct_gains, equity_curve, all_hold_candles = [], [], []

    # Rolling VWAP calc
    df['cum_vol'] = df['volume'].rolling(vwap_window).sum()
    df['cum_vol_x_price'] = (df['close'] * df['volume']).rolling(vwap_window).sum()
    df['vwap'] = df['cum_vol_x_price'] / df['cum_vol']
    df.dropna(inplace=True)

    open_, close, low = df['open'].values, df['close'].values, df['low'].values
    vwap = df['vwap'].values

    for i in range(1, len(df)):
        price = open_[i]
        vw = vwap[i - 1]  # use previous candle's VWAP signal

        # Cooldown after closing trade
        if cooldown_counter > 0:
            cooldown_counter -= 1
            equity_curve.append(asset * price if asset > 0 else cash)
            continue

        # --- Entry logic ---
        if asset == 0:
            # Mean-reversion trigger: price significantly below VWAP
            if price < vw * (1 - vwap_distance_threshold):
                entry_price = price
                asset = (cash * (1 - fee_rate)) / entry_price
                cash = 0
                hold_candles = 0

                # Stop-loss check on entry candle
                stop_loss_price = entry_price * (1 - stop_loss)
                if low[i] <= stop_loss_price:
                    exit_price = stop_loss_price
                    cash_after = asset * exit_price * (1 - fee_rate)
                    pct_gain = (exit_price * (1 - 2 * fee_rate) - entry_price) / entry_price * 100
                    pct_gains.append(pct_gain)
                    cash, asset = cash_after, 0
                    cooldown_counter = cooldown_candles
                    equity_curve.append(cash)
                    continue

        # --- Exit logic ---
        else:
            hold_candles += 1
            stop_loss_price = entry_price * (1 - stop_loss)

            if low[i] <= stop_loss_price:
                exit_price = stop_loss_price
            elif price > vw * (1 - vwap_exit_threshold) or hold_candles >= max_hold_candles:
                exit_price = price
            else:
                equity_curve.append(asset * price)
                continue

            cash_after = asset * exit_price * (1 - fee_rate)
            pct_gain = (exit_price * (1 - 2 * fee_rate) - entry_price) / entry_price * 100
            pct_gains.append(pct_gain)
            all_hold_candles.append(hold_candles)

            cash, asset = cash_after, 0
            cooldown_counter = cooldown_candles

        # Track value for equity curve
        equity_curve.append(asset * price if asset > 0 else cash)

    # Risk metrics
    equity_curve = np.array(equity_curve)
    max_drawdown = (
        np.max((np.maximum.accumulate(equity_curve) - equity_curve) / np.maximum.accumulate(equity_curve)) * 100
        if equity_curve.size > 0 else 0
    )
    avg_hold = np.mean(all_hold_candles) * interval_minutes / 60 if all_hold_candles else 0

    return pct_gains, max_drawdown, avg_hold, len(pct_gains)

# Round floats for nice JSON dumps
def round_floats(obj, decimals=3):
    if isinstance(obj, float):
        return round(obj, decimals)
    elif isinstance(obj, dict):
        return {k: round_floats(v, decimals) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [round_floats(i, decimals) for i in obj]
    return obj

# Wrap original backtest call to inject symbol-specific fee
def run_backtest_for_with_fee_rate(combo, base_folder, param_combinations, spreads_dict):
    symbol, interval = combo
    spread = spreads_dict.get(symbol, 0)
    fee_rate = BASE_FEE_RATE + spread

    def strategy_with_fee_rate(df, *args, **kwargs):
        return vwap_strategy(df, *args, fee_rate=fee_rate, **kwargs)

    return run_backtest_for(combo, base_folder, strategy_with_fee_rate, param_combinations)

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, PROJECT_ROOT, "symbols.csv")
    base_folder = os.path.abspath(os.path.join(script_dir, PROJECT_ROOT, "data"))

    symbols = pd.read_csv(csv_path)['Symbol'].tolist()
    intervals = ['3m', '5m', '15m', '30m', '1h']
    combos = [(s, i) for s in symbols for i in intervals]

    param_combinations = generate_random_param_combinations()

    # Load symbol tick-size spreads
    json_path = os.path.join(script_dir, PROJECT_ROOT, "symbol_quantities10.json")
    with open(json_path, "r") as f:
        symbol_data = json.load(f)

    spreads_dict = {}
    for symbol, data in symbol_data.items():
        try:
            tick_size = data["tick_size"]
            price = data["price"]
            spreads_dict[symbol] = tick_size / price
        except Exception:
            spreads_dict[symbol] = 0  # fallback

    # Parallel backtesting across symbols/timeframes
    with Pool(processes=5) as pool:
        results = pool.starmap(run_backtest_for_with_fee_rate, [
            (combo, base_folder, param_combinations, spreads_dict) for combo in combos
        ])

    # Only keep successful runs
    best_results = [result for error, result in results if result]

    output_path = os.path.join(script_dir, "longParams.json")

    # Rank by capital efficiency
    best_results.sort(key=lambda r: r["daily_capital_efficiency"], reverse=True)
    rounded_results = [round_floats(result) for result in best_results]

    with open(output_path, "w") as f:
        json.dump(rounded_results, f, indent=2)

    print(f"Saved {len(best_results)} results to {output_path}")

if __name__ == '__main__':
    main()
