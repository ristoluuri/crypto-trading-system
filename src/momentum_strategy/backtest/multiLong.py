import os
import sys
import json
import pandas as pd
import numpy as np
import random
from multiprocessing import Pool

# === Add project path to import helper ===
base_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(base_dir, "..", ".."))
sys.path.append(PROJECT_ROOT)
from helpers.back_test.bt_multi import run_backtest_for

# === Config constants ===
BASE_FEE_RATE = 0.00045  # Base trading fee rate per trade
NUM_RANDOM_COMBINATIONS = 5  # Number of random parameter combinations to test per symbol/interval

momentum_windows = list(range(2, 42, 4))  # Momentum lookback periods
volatility_windows = list(range(20, 420, 40))  # Volatility lookback periods
max_hold_candles_list = list(range(5, 200, 10))  # Max number of candles to hold a position
volume_factors = np.arange(2.4, 4.1, 0.2).round(2).tolist()  # Volume threshold multipliers

# === Random parameter generator ===
def generate_random_param_combinations():
    """
    Generate a list of random parameter combinations for backtesting.

    Returns:
        List[dict]: Each dict contains momentum_window, volatility_window,
                    max_hold_candles, and volume_factor.
    """
    return [dict(
        momentum_window=random.choice(momentum_windows),
        volatility_window=random.choice(volatility_windows),
        max_hold_candles=random.choice(max_hold_candles_list),
        volume_factor=random.choice(volume_factors)
    ) for _ in range(NUM_RANDOM_COMBINATIONS)]

# === Strategy Implementation ===
def momentum_strategy(df, momentum_window, volatility_window, max_hold_candles, volume_factor, interval_minutes, fee_rate):
    """
    Implements a simple momentum-based trading strategy.

    Args:
        df (pd.DataFrame): Historical OHLCV data with columns 'open', 'close', 'volume'.
        momentum_window (int): Lookback period for momentum calculation.
        volatility_window (int): Lookback period for volatility calculation.
        max_hold_candles (int): Maximum number of candles to hold a position.
        volume_factor (float): Multiplier for volume threshold.
        interval_minutes (int): Candle interval in minutes.
        fee_rate (float): Trading fee rate.

    Returns:
        tuple: (pct_gains, total_return, avg_hold_hours, num_trades)
            - pct_gains (List[float]): % gain per trade
            - total_return (float): Total % gain over period
            - avg_hold_hours (float): Average hold duration per trade in hours
            - num_trades (int): Number of trades executed
    """
    cash, btc = 1000.0, 0.0
    hold_candles = 0
    pct_gains, all_hold_candles = [], []

    # Compute indicators
    df['momentum'] = df['close'].pct_change(periods=momentum_window)
    df['return'] = df['close'].pct_change()
    df['volatility'] = df['return'].rolling(volatility_window).std()
    df['avg_volume'] = df['volume'].rolling(volatility_window).mean()
    df.dropna(inplace=True)

    # Convert series to arrays for faster iteration
    open_, close = df['open'].values, df['close'].values
    momentum = df['momentum'].values
    volume, avg_volume = df['volume'].values, df['avg_volume'].values

    # Trading loop
    for i in range(max(momentum_window, volatility_window), len(df)):
        mom = momentum[i - 1]
        vol_today = volume[i - 1]
        vol_avg = avg_volume[i - 1]
        price = open_[i]
        is_green = close[i - 1] > open_[i - 1]

        # Entry condition: momentum positive, high volume, green candle
        if btc == 0 and mom > 0 and vol_today > volume_factor * vol_avg and is_green:
            entry_price = price
            btc = (cash * (1 - fee_rate)) / entry_price
            cash = 0
            hold_candles = 0
            entry_price_recorded = entry_price
        # Exit condition: momentum negative or max hold exceeded
        elif btc > 0:
            hold_candles += 1
            if mom < 0 or hold_candles >= max_hold_candles:
                exit_price = price
                cash_after = btc * exit_price * (1 - fee_rate)
                pct_gain = (exit_price * (1 - 2 * fee_rate) - entry_price_recorded) / entry_price_recorded * 100
                pct_gains.append(pct_gain)
                all_hold_candles.append(hold_candles)
                cash, btc, hold_candles = cash_after, 0, 0

    avg_hold_hours = np.mean(all_hold_candles) * interval_minutes / 60 if all_hold_candles else 0
    num_trades = len(pct_gains)
    total_return = sum(pct_gains)

    return pct_gains, total_return, avg_hold_hours, num_trades

# === Optional rounding helper ===
def round_floats(obj, decimals=3):
    """
    Recursively round float values in dicts, lists, or standalone floats.

    Args:
        obj: Object containing floats (float, dict, list)
        decimals (int): Number of decimal places

    Returns:
        Rounded object
    """
    if isinstance(obj, float):
        return round(obj, decimals)
    elif isinstance(obj, dict):
        return {k: round_floats(v, decimals) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [round_floats(i, decimals) for i in obj]
    else:
        return obj

# === Wrapper to include fee_rate per symbol ===
def run_backtest_for_with_fee_rate(combo, base_folder, param_combinations, spreads_dict):
    """
    Wrapper around `run_backtest_for` that adjusts the fee rate per symbol
    using the symbol's spread.

    Args:
        combo (tuple): (symbol, interval)
        base_folder (str): Path to historical data folder
        param_combinations (List[dict]): List of parameter sets
        spreads_dict (dict): Mapping from symbol -> spread

    Returns:
        result from `run_backtest_for` with adjusted fee rate
    """
    symbol, interval = combo
    spread = spreads_dict.get(symbol, 0)
    fee_rate = BASE_FEE_RATE + spread

    def strategy_with_fee_rate(df, **kwargs):
        interval_minutes = kwargs.pop('interval_minutes', 1)
        return momentum_strategy(df, fee_rate=fee_rate, interval_minutes=interval_minutes, **kwargs)

    return run_backtest_for(combo, base_folder, strategy_with_fee_rate, param_combinations)

# === Main script entry ===
def main():
    """
    Run backtests for all symbol/interval combinations, apply random parameters,
    adjust fee rates per symbol, and save best results to JSON.
    """
    csv_path = os.path.join(PROJECT_ROOT, "symbols.csv")
    base_folder = os.path.join(PROJECT_ROOT, "data")

    symbols = pd.read_csv(csv_path)['Symbol'].tolist()
    intervals = ['3m', '5m', '15m', '30m', '1h']
    combos = [(s, i) for s in symbols for i in intervals]

    # Generate random parameter combinations
    param_combinations = generate_random_param_combinations()

    # Load symbol spreads from JSON
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
        results = pool.starmap(run_backtest_for_with_fee_rate, [
            (combo, base_folder, param_combinations, spreads_dict) for combo in combos
        ])

    # Collect best results and sort by daily capital efficiency
    best_results = [result for error, result in results if result]
    output_path = os.path.join(base_dir, "longParams.json")
    best_results.sort(key=lambda r: r["daily_capital_efficiency"], reverse=True)
    rounded_results = [round_floats(r) for r in best_results]

    with open(output_path, "w") as f:
        json.dump(rounded_results, f, indent=2)

    print(f"Saved {len(best_results)} results to {output_path}")

if __name__ == '__main__':
    main()
