import os
import sys
import json
import pandas as pd
import numpy as np
import random
from multiprocessing import Pool

# === Add project path so we can import project helpers ===
base_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(base_dir, "..", ".."))
sys.path.append(PROJECT_ROOT)
from helpers.back_test.bt_multi import run_backtest_for

# === Config constants ===
BASE_FEE_RATE = 0.00045  # Base trading fee rate for futures
NUM_RANDOM_COMBINATIONS = 5  # Number of parameter samples per symbol/interval

# Parameter search space (wider for robustness, fewer iterations)
momentum_windows = list(range(2, 42, 10))
volatility_windows = list(range(50, 300, 100))
max_hold_candles_list = list(range(5, 75, 20))
volume_factors = np.arange(2.4, 4.1, 0.2).round(2).tolist()

# === Generate random hyperparameters ===
def generate_random_param_combinations():
    """
    Create random parameter combinations for strategy testing.

    Returns:
        list[dict]: Random combinations of momentum, volatility, hold time, and volume multiplier.
    """
    return [
        dict(
            momentum_window=random.choice(momentum_windows),
            volatility_window=random.choice(volatility_windows),
            max_hold_candles=random.choice(max_hold_candles_list),
            volume_factor=random.choice(volume_factors)
        )
        for _ in range(NUM_RANDOM_COMBINATIONS)
    ]


# === Momentum shorting strategy ===
def momentum_short_strategy(
    df,
    momentum_window,
    volatility_window,
    max_hold_candles,
    volume_factor,
    interval_minutes,
    fee_rate
):
    """
    Short-side momentum strategy:
      • Enter short when momentum turns negative + high volume + red candle
      • Exit when momentum turns positive or max hold duration reached

    Args:
        df (pd.DataFrame): Candle data (open/close/volume required)
        momentum_window (int)
        volatility_window (int)
        max_hold_candles (int)
        volume_factor (float): Volume threshold multiplier
        interval_minutes (int): Candle interval length in minutes
        fee_rate (float): Futures fee rate

    Returns:
        tuple: (pct_gains, total_return, avg_hold_hours, num_trades)
    """
    cash = 1000.0  # Start capital
    short_qty = 0.0
    hold_candles = 0
    pct_gains, all_hold_candles = [], []

    # Indicators
    df['momentum'] = df['close'].pct_change(periods=momentum_window)
    df['return'] = df['close'].pct_change()
    df['volatility'] = df['return'].rolling(volatility_window).std()
    df['avg_volume'] = df['volume'].rolling(volatility_window).mean()
    df.dropna(inplace=True)

    open_, close = df['open'].values, df['close'].values
    momentum = df['momentum'].values
    volume, avg_volume = df['volume'].values, df['avg_volume'].values

    for i in range(max(momentum_window, volatility_window), len(df)):
        mom = momentum[i - 1]
        vol_today = volume[i - 1]
        vol_avg = avg_volume[i - 1]
        price = open_[i]
        is_green = close[i - 1] > open_[i - 1]

        # Entry: negative momentum, volume spike, red candle
        if short_qty == 0 and mom < 0 and vol_today > volume_factor * vol_avg and not is_green:
            entry_price = price
            short_qty = (cash * (1 - fee_rate)) / entry_price
            cash = 0
            hold_candles = 0
            entry_price_recorded = entry_price

        # Exit logic
        elif short_qty > 0:
            hold_candles += 1

            # Exit on positive momentum or time limit
            if mom > 0 or hold_candles >= max_hold_candles:
                exit_price = price

                # Short PnL: qty * (entry - exit)
                pnl = short_qty * (entry_price_recorded - exit_price)
                cash_after = (pnl + short_qty * exit_price) * (1 - fee_rate)

                pct_gain = (
                    (entry_price_recorded - exit_price) - (entry_price_recorded * 2 * fee_rate)
                ) / entry_price_recorded * 100

                pct_gains.append(pct_gain)
                all_hold_candles.append(hold_candles)

                cash, short_qty, hold_candles = cash_after, 0, 0

    avg_hold_hours = np.mean(all_hold_candles) * interval_minutes / 60 if all_hold_candles else 0
    num_trades = len(pct_gains)
    total_return = sum(pct_gains)

    return pct_gains, total_return, avg_hold_hours, num_trades


# === Rounding helper ===
def round_floats(obj, decimals=3):
    """Recursively round floats inside dicts/lists for clean JSON output."""
    if isinstance(obj, float):
        return round(obj, decimals)
    elif isinstance(obj, dict):
        return {k: round_floats(v, decimals) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [round_floats(i, decimals) for i in obj]
    return obj


# === Attach fee rate per symbol based on spread ===
def run_backtest_for_with_fee_rate(combo, base_folder, param_combinations, spreads_dict):
    """
    Wrap `run_backtest_for` and inject fee adjusted per symbol spread.

    Args:
        combo (tuple): (symbol, interval)
        base_folder (str): Folder with OHLCV CSVs
        param_combinations (list)
        spreads_dict (dict): spread adjustments per symbol

    Returns:
        tuple: (error, best_results)
    """
    symbol, interval = combo
    spread = spreads_dict.get(symbol, 0)
    fee_rate = BASE_FEE_RATE + spread

    def strategy_with_fee_rate(df, **kwargs):
        interval_minutes = kwargs.pop('interval_minutes', 1)
        return momentum_short_strategy(df, fee_rate=fee_rate, interval_minutes=interval_minutes, **kwargs)

    return run_backtest_for(combo, base_folder, strategy_with_fee_rate, param_combinations)


# === Main runner ===
def main():
    """
    Run short-side momentum backtests across symbols & timeframes,
    sample random hyperparameters, save results to JSON.
    """
    csv_path = os.path.join(PROJECT_ROOT, "symbols.csv")
    base_folder = os.path.join(PROJECT_ROOT, "data")

    symbols = pd.read_csv(csv_path)['Symbol'].tolist()
    intervals = ['3m', '5m', '15m', '30m', '1h']
    combos = [(s, i) for s in symbols for i in intervals]

    param_combinations = generate_random_param_combinations()

    # Load symbol spread data (tick_size / price)
    spreads_json_path = os.path.join(PROJECT_ROOT, "symbol_quantities10.json")
    with open(spreads_json_path, "r") as f:
        symbol_data = json.load(f)

    spreads_dict = {}
    for symbol, data in symbol_data.items():
        try:
            spreads_dict[symbol] = data["tick_size"] / data["price"]
        except Exception:
            spreads_dict[symbol] = 0

    # Run backtests in parallel
    with Pool(processes=5) as pool:
        results = pool.starmap(
            run_backtest_for_with_fee_rate,
            [(combo, base_folder, param_combinations, spreads_dict) for combo in combos]
        )

    best_results = [result for error, result in results if result]

    output_path = os.path.join(base_dir, "shortParamsRobust.json")
    best_results.sort(key=lambda r: r["daily_capital_efficiency"], reverse=True)

    rounded = [round_floats(r) for r in best_results]
    with open(output_path, "w") as f:
        json.dump(rounded, f, indent=2)

    print(f"Saved {len(best_results)} results to {output_path}")


if __name__ == '__main__':
    main()
