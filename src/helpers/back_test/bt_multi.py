import os
import pandas as pd
import numpy as np
import warnings
from dateutil.relativedelta import relativedelta

warnings.simplefilter(action='ignore', category=FutureWarning)

# 23.09 added the rule that the last 2 splits must have 40% of total trade volume and avg_gain_per_trade > 0.3
NUM_SPLITS = 5
MIN_TRADES_REQUIRED = {
    "1h": 100,
    "30m": 120,
    "15m": 150,
    "5m": 200,
    "3m": 300
}

def interval_to_minutes(interval):
    if interval.endswith('m'):
        return int(interval[:-1])
    elif interval.endswith('h'):
        return int(interval[:-1]) * 60
    elif interval.endswith('d'):
        return int(interval[:-1]) * 1440
    else:
        raise ValueError(f"Unknown interval format: {interval}")

def compute_sharpe(returns, periods_per_year=252 * 24 * 4):
    if len(returns) < 2:
        return 0.0
    returns = np.array(returns) / 100
    mean_ret = np.mean(returns)
    std_ret = np.std(returns)
    if std_ret == 0:
        return 0.0

    sharpe = (mean_ret / std_ret) * np.sqrt(len(returns))

    # --- Sanitize Sharpe values ---
    if not np.isfinite(sharpe) or sharpe < 0 or sharpe > 10:
        return 0.0

    return sharpe

def load_data(symbol, interval, base_folder):
    filepath = os.path.join(base_folder, symbol.lower(), f"{symbol}_{interval}_last1000_data.json")
    if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
        return None, f"Missing or empty file: {filepath}"

    df = pd.read_json(filepath, lines=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)
    return df, None

def run_backtest_for(symbol_interval, base_folder, strategy_func, param_combinations, df=None):
    symbol, interval = symbol_interval
    if df is None:
        df, error = load_data(symbol, interval, base_folder)
        if error:
            return error, None
    else:
        error = None

    interval_minutes = interval_to_minutes(interval)
    candles_per_day = 1440 / interval_minutes
    all_results = []

    for param_set in param_combinations:
        test_gains, test_trades, test_drawdowns = [], [], []
        test_avg_holds, test_avg_trade_gains, test_sharpes = [], [], []

        splits = np.array_split(df, NUM_SPLITS)
        for df_test in splits:
            pct_gains, _, avg_hold, num_trades = strategy_func(
                df_test.copy(), interval_minutes=interval_minutes, **param_set
            )

            # Ensure gains are a list
            if isinstance(pct_gains, (float, np.floating)):
                pct_gains = [pct_gains]

            if not pct_gains or not all((1 + g / 100) > 0 for g in pct_gains):
                continue

            # Compute equity curve
            equity_curve = np.cumprod([1 + g / 100 for g in pct_gains])
            equity_arr = np.array(equity_curve)
            running_max = np.maximum.accumulate(equity_arr)
            drawdowns = (running_max - equity_arr) / running_max
            max_dd = np.max(drawdowns) * 100 if drawdowns.size > 0 else 0

            test_days = len(df_test) / candles_per_day
            geo_avg_gain = (np.prod([1 + g / 100 for g in pct_gains]) ** (1 / len(pct_gains)) - 1) * 100
            gain_per_day = geo_avg_gain * (num_trades / test_days)
            sharpe = compute_sharpe(pct_gains)

            test_gains.append(gain_per_day)
            test_trades.append(num_trades)
            test_drawdowns.append(max_dd)
            test_avg_holds.append(avg_hold)
            test_avg_trade_gains.append(geo_avg_gain)
            test_sharpes.append(sharpe)

        if len(test_gains) == NUM_SPLITS:
            weights = np.array(test_trades) / np.sum(test_trades)
            avg_gain_per_day = np.average(test_gains, weights=weights)
            avg_gain_per_trade = np.average(test_avg_trade_gains, weights=weights)
            avg_hold_hours = np.average(test_avg_holds, weights=weights)
            avg_sharpe = np.average(test_sharpes, weights=weights)
            avg_drawdown = np.average(test_drawdowns, weights=weights)
            total_trades = int(np.sum(test_trades))
            total_days = len(df) / candles_per_day
            avg_trades_per_day = total_trades / total_days
            capital_efficiency = avg_gain_per_trade / avg_hold_hours if avg_hold_hours > 0 else 0
            daily_capital_efficiency = capital_efficiency * avg_trades_per_day

            all_results.append({
                'symbol': symbol,
                'interval': interval,
                'params': param_set,
                'avg_gain_per_day': avg_gain_per_day,
                'avg_gain_per_trade': avg_gain_per_trade,
                'avg_hold_hours': avg_hold_hours,
                'capital_efficiency': capital_efficiency,
                'daily_capital_efficiency': daily_capital_efficiency,
                'avg_sharpe': avg_sharpe,
                'avg_trades_per_day': avg_trades_per_day,
                'total_trades': total_trades,
                'avg_drawdown': avg_drawdown,
                'split_details': [
                    {
                        'split_index': i + 1,
                        'gain_per_day': test_gains[i],
                        'gain_per_trade': test_avg_trade_gains[i],
                        'sharpe': test_sharpes[i],
                        'drawdown': test_drawdowns[i],
                        'avg_hold_hours': test_avg_holds[i],
                        'trades': test_trades[i],
                        'trades_per_day': test_trades[i] / (len(splits[i]) / candles_per_day),
                    }
                    for i in range(NUM_SPLITS)
                ]
            })

    filtered_results_all = []
    for r in all_results:
        total_trades = sum(x['trades'] for x in r['split_details'])
        last2_trades = sum(x['trades'] for x in r['split_details'][-2:])
        last2_gain = np.mean([x['gain_per_trade'] for x in r['split_details'][-2:]])

        # basic requirements
        if r['avg_gain_per_trade'] <= 0.3:
            continue
        if total_trades <= MIN_TRADES_REQUIRED.get(r['interval'], 100):
            continue
        if r['capital_efficiency'] <= 0.001:
            continue

        # new requirements: last 2 splits
        if last2_trades < 0.4 * total_trades:
            continue
        if last2_gain <= 0.3:
            continue

        filtered_results_all.append(r)

    all_results = filtered_results_all

    if not all_results:
        return f"No strong performers for {symbol} {interval}", None

    threshold_sharpe = np.percentile([r['avg_sharpe'] for r in all_results], 80)
    filtered_results = [r for r in all_results if r['avg_sharpe'] >= threshold_sharpe]

    if not filtered_results:
        return f"No parameters meet top 10% Sharpe ratio for {symbol} {interval}", None

    best_result = max(filtered_results, key=lambda r: r['avg_sharpe'])
    return None, best_result