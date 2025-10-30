import numpy as np
import pandas as pd

def calculate_metrics(pct_gains, equity_curve, final_value):
    stats = {}
    if pct_gains:
        clipped_gains = [max(min(g, 999), -99.9) for g in pct_gains]
        avg_pct_gain = (np.prod([(1 + g / 100) for g in clipped_gains]) ** (1 / len(clipped_gains)) - 1) * 100
        winrate = (sum(g > 0 for g in pct_gains) / len(pct_gains)) * 100
        returns = np.array(pct_gains)
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        sharpe_ratio = mean_return / std_return if std_return > 0 else float('nan')

        equity_arr = np.array(equity_curve)
        running_max = np.maximum.accumulate(equity_arr)
        drawdowns = (running_max - equity_arr) / running_max
        max_drawdown = np.max(drawdowns) * 100 if len(drawdowns) > 0 else 0

        stats = {
            'final_value': final_value,
            'total_trades': len(pct_gains),
            'geometric_avg_gain': avg_pct_gain,
            'winrate': winrate,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
        }
    else:
        stats = {
            'final_value': final_value,
            'total_trades': 0,
            'geometric_avg_gain': 0,
            'winrate': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0,
        }
    return stats

def summarize_results(pct_gains, equity_curve, positions, close_price, fee_rate, open_position_value=0):
    if positions and positions[-1].get('cash_after'):
        final_value = positions[-1]['cash_after']
    elif open_position_value > 0:
        final_value = open_position_value * (1 - fee_rate)
    else:
        final_value = close_price * (1 - fee_rate)

    stats = calculate_metrics(pct_gains, equity_curve, final_value)

    print(f"\nFinal portfolio value: ${stats['final_value']:.2f}")
    print(f"Total trades: {stats['total_trades']}")

    if stats['total_trades'] > 0:
        print(f"Geometric average % gain per trade: {stats['geometric_avg_gain']:.2f}%")
        print(f"Win rate: {stats['winrate']:.2f}%")
        print(f"Sharpe Ratio: {stats['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {stats['max_drawdown']:.2f}%")
    else:
        print("No trades made.")

    print("\nLast 3 trades:")
    for t in positions[-3:]:
        entry_time_str = pd.to_datetime(t['entry_time']).strftime("%Y-%m-%d %H:%M")
        exit_time_str = pd.to_datetime(t['exit_time']).strftime("%Y-%m-%d %H:%M") if 'exit_time' in t else "-"
        print(f"{entry_time_str} to {exit_time_str} | Entry: {t['entry_price']:.5f} | "
              f"Exit: {t.get('exit_price', 0):.5f} | Cash Before: ${t['cash_before']:.2f} | "
              f"Cash After: ${t.get('cash_after', 0):.2f} | % Gain: {t.get('pct_gain', 0):.2f}% | "
              f"... Min Low During Trade (%): {t.get('min_low_during_trade', 0):.2f}% | "
              f"... Max High During Trade (%): {t.get('max_high_during_trade', 0):.2f}%")
        
        

    return stats
