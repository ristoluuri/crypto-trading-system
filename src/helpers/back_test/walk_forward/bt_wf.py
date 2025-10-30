import os
import pandas as pd
import numpy as np
import warnings
from dateutil.relativedelta import relativedelta

warnings.simplefilter(action='ignore', category=FutureWarning)

MIN_TRADES = 15

def interval_to_minutes(interval):
    if interval.endswith('m'):
        return int(interval[:-1])
    elif interval.endswith('h'):
        return int(interval[:-1]) * 60
    elif interval.endswith('d'):
        return int(interval[:-1]) * 1440
    else:
        raise ValueError(f"Unknown interval format: {interval}")

def compute_sharpe(returns):
    if len(returns) < 2:
        return 0.0
    returns = np.array(returns) / 100
    mean_ret = np.mean(returns)
    std_ret = np.std(returns)
    if std_ret == 0:
        return 0.0
    sharpe = (mean_ret / std_ret) * np.sqrt(len(returns))
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

    interval_minutes = interval_to_minutes(interval)
    candles_per_day = 1440 / interval_minutes
    all_results = []

    for param_set in param_combinations:
        pct_gains, _, avg_hold, num_trades = strategy_func(df.copy(), interval_minutes=interval_minutes, **param_set)

        if isinstance(pct_gains, (float, np.floating)):
            pct_gains = [pct_gains]

        if not pct_gains or not all((1 + g / 100) > 0 for g in pct_gains):
            continue

        equity_curve = np.cumprod([1 + g / 100 for g in pct_gains])
        equity_arr = np.array(equity_curve)
        running_max = np.maximum.accumulate(equity_arr)
        drawdowns = (running_max - equity_arr) / running_max
        max_dd = np.max(drawdowns) * 100 if drawdowns.size > 0 else 0

        avg_trades_per_day = num_trades / (len(df) / candles_per_day)
        sharpe = compute_sharpe(pct_gains)
        capital_efficiency = np.mean(pct_gains) / avg_hold if avg_hold > 0 else 0
        daily_capital_efficiency = capital_efficiency * avg_trades_per_day

        all_results.append({
            'symbol': symbol,
            'interval': interval,
            'params': param_set,
            'avg_hold_hours': avg_hold,
            'capital_efficiency': capital_efficiency,
            'daily_capital_efficiency': daily_capital_efficiency,
            'avg_sharpe': sharpe,
            'avg_trades_per_day': avg_trades_per_day,
            'total_trades': num_trades,
            'avg_drawdown': max_dd,
            'pct_gains': pct_gains
        })

    if not all_results:
        return f"No strong performers for {symbol} {interval}", None

    best_result = max(all_results, key=lambda r: r['avg_sharpe'])
    return None, best_result

def generate_walk_forward_folds(df, train_months=9, test_months=3, step_months=3):
    folds = []
    start = df.index.min()
    end = df.index.max()
    train_delta = relativedelta(months=train_months)
    test_delta = relativedelta(months=test_months)
    step_delta = relativedelta(months=step_months)
    current_train_start = start

    while True:
        current_train_end = current_train_start + train_delta - pd.Timedelta(seconds=1)
        current_test_start = current_train_end + pd.Timedelta(seconds=1)
        current_test_end = current_test_start + test_delta - pd.Timedelta(seconds=1)
        if current_test_end > end:
            break

        train_df = df[(df.index >= current_train_start) & (df.index <= current_train_end)].copy()
        test_df = df[(df.index >= current_test_start) & (df.index <= current_test_end)].copy()
        if len(train_df) == 0 or len(test_df) == 0:
            break

        folds.append({
            'train_df': train_df,
            'test_df': test_df,
            'train_start': current_train_start,
            'train_end': current_train_end,
            'test_start': current_test_start,
            'test_end': current_test_end
        })

        current_train_start += step_delta

    return folds

def run_walk_forward(symbol_interval, base_folder, strategy_func, param_combinations,
                     train_months=9, test_months=3, step_months=3):
    symbol, interval = symbol_interval
    df, error = load_data(symbol, interval, base_folder)
    if error:
        return error, None

    folds = generate_walk_forward_folds(df, train_months, test_months, step_months)
    total_folds = len(folds)   # ✅ capture all folds before filtering
    all_fold_results = []

    for fold in folds:
        _, best_train_result = run_backtest_for((symbol, interval), None, strategy_func, param_combinations, df=fold['train_df'])
        if best_train_result is None or best_train_result['total_trades'] < MIN_TRADES:
            continue

        _, test_result = run_backtest_for((symbol, interval), None, strategy_func, [best_train_result['params']], df=fold['test_df'])
        if test_result:
            test_result['winrate_per_trade'] = sum(
                1 for g in test_result.get('pct_gains', []) if g > 0
            ) / max(1, len(test_result.get('pct_gains', [])))
            all_fold_results.append({
                'fold': fold,
                'train_result': best_train_result,
                'test_result': test_result
            })

    if not all_fold_results:
        return f"No strong folds for {symbol} {interval}", None

    # --- Combine all trades across folds ---
    all_gains = []
    all_holds = []
    all_drawdowns = []
    all_trades_count = 0
    fold_weights = []

    for fold in all_fold_results:
        gains = fold['test_result'].get('pct_gains', [])
        all_gains.extend(gains)
        all_holds.append(fold['test_result']['avg_hold_hours'])
        all_drawdowns.append(fold['test_result']['avg_drawdown'])
        fold_trades = fold['test_result']['total_trades']
        fold_weights.append(fold_trades)
        all_trades_count += fold_trades

    # Weight fold-level metrics by number of trades
    avg_result = {}
    avg_result['avg_hold_hours'] = np.average(all_holds, weights=fold_weights)
    avg_result['avg_drawdown'] = np.average(all_drawdowns, weights=fold_weights)

    # Compute average trades per day across the whole walk-forward period
    total_days = sum((fold['fold']['test_end'] - fold['fold']['test_start']).days for fold in all_fold_results)
    avg_result['avg_trades_per_day'] = all_trades_count / max(1, total_days)

    avg_result['avg_gain_per_trade'] = np.mean(all_gains) if all_gains else 0
    avg_result['capital_efficiency'] = np.mean(all_gains) / avg_result['avg_hold_hours'] if avg_result['avg_hold_hours'] > 0 else 0
    avg_result['daily_capital_efficiency'] = avg_result['capital_efficiency'] * avg_result['avg_trades_per_day']
    avg_result['avg_sharpe'] = compute_sharpe(all_gains)
    avg_result['total_trades'] = all_trades_count

    # Winrate across all trades
    total_positive_trades = sum(1 for g in all_gains if g > 0)
    avg_result['winrate_per_trade'] = total_positive_trades / max(1, all_trades_count)

    # Winrate across splits/folds
    positive_folds = sum(1 for fold in all_fold_results if np.mean(fold['test_result'].get('pct_gains', [])) > 0)
    avg_result['winrate_per_fold'] = positive_folds / len(all_fold_results)

    # Add metadata
    avg_result['symbol'] = symbol
    avg_result['interval'] = interval
    avg_result['num_folds'] = len(all_fold_results)   # ✅ valid folds
    avg_result['total_folds'] = total_folds           # ✅ all folds
    avg_result['pct_of_folds'] = len(all_fold_results) / max(1, total_folds)  # ✅ %

    return None, {
        "avg_result": avg_result,
        "folds": all_fold_results,
        "num_folds": len(all_fold_results),
        "total_folds": total_folds,
        "pct_of_folds": avg_result['pct_of_folds']
    }
