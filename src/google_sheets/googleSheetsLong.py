import os
import io
import pandas as pd
import numpy as np
import random
from multiprocessing import Pool
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.auth.exceptions import RefreshError

# === Google Sheets API setup ===
# SCOPES define the level of access the script has. Here, full access to read/write spreadsheets.
SCOPES = ['https://www.googleapis.com/auth/spreadsheets']
# ID of the Google Sheet where backtest results will be stored
SPREADSHEET_ID = '1Gn8cOJXPO37t-Iyx45suPJv6WaUt6EHS24T_bL10_hc'

# === Backtest configuration ===

# Get the current script directory to locate credential files
credential_folder = os.path.dirname(os.path.abspath(__file__))
token_path = os.path.join(credential_folder, 'token.json')  # Stores the user's OAuth token
creds_path = os.path.join(credential_folder, 'credentials.json')  # Client credentials for OAuth

# Load symbols from a CSV file
symbols_df = pd.read_csv("symbols6.csv")
symbols = symbols_df['Symbol'].tolist()

# Define intervals to backtest
intervals = ['15m', '30m', '1h']
# Base folder where historical data is stored
base_folder = 'data'

# Trading fee rate applied per trade
fee_rate = 0.0005
# Number of random parameter combinations to test per backtest
NUM_RANDOM_COMBINATIONS = 10
# Number of splits for walk-forward testing
NUM_SPLITS = 5

# Parameter ranges for backtesting
momentum_windows = list(range(2, 42, 4))
volatility_windows = list(range(20, 420, 40))
max_hold_candles_list = list(range(5, 200, 10))
volume_factors = np.arange(2.4, 4.1, 0.2).round(2).tolist()

# === Helper Functions ===

def col_letter(n):
    """
    Converts a number to an Excel-style column letter (1 -> A, 27 -> AA)
    """
    result = ''
    while n > 0:
        n, rem = divmod(n - 1, 26)
        result = chr(65 + rem) + result
    return result

def interval_to_minutes(interval):
    """
    Converts a string interval (e.g., '15m', '1h') to minutes as integer.
    """
    if interval.endswith('m'):
        return int(interval[:-1])
    elif interval.endswith('h'):
        return int(interval[:-1]) * 60
    elif interval.endswith('d'):
        return int(interval[:-1]) * 1440
    else:
        raise ValueError(f"Unknown interval format: {interval}")

def generate_random_param_combinations():
    """
    Generates a list of random backtesting parameter combinations.
    """
    return [{
        'momentum_window': random.choice(momentum_windows),
        'volatility_window': random.choice(volatility_windows),
        'max_hold_candles': random.choice(max_hold_candles_list),
        'volume_factor': random.choice(volume_factors),
    } for _ in range(NUM_RANDOM_COMBINATIONS)]

def compute_sharpe(returns, periods_per_year=252*24*4):
    """
    Compute the Sharpe ratio for a list of returns.
    Assumes returns are in percentages and converts to decimal.
    Annualizes based on the number of periods per year.
    """
    if len(returns) < 2:
        return 0.0
    returns = np.array(returns) / 100  # convert % to decimal
    mean_ret = np.mean(returns)
    std_ret = np.std(returns)
    if std_ret == 0:
        return 0.0
    # Simple annualized Sharpe ratio approximation
    sharpe = (mean_ret / std_ret) * np.sqrt(len(returns))
    return sharpe

def backtest(df, volume_factor, momentum_window, volatility_window, max_hold_candles, interval_minutes):
    """
    Runs the momentum-based trading strategy on a dataframe of historical price data.
    Returns the list of trade percentage gains, maximum drawdown, and average hold time in hours.
    """
    cash = 1000.0  # initial capital
    btc = 0.0  # amount of asset held
    hold_candles = 0
    pct_gains = []
    equity_curve = []
    all_hold_candles = []

    start_index = max(momentum_window, volatility_window) + 1

    # Extract data arrays for faster access
    open_ = df['open'].values
    close = df['close'].values
    momentum = df['momentum'].values
    volume = df['volume'].values
    avg_volume = df['avg_volume'].values

    for i in range(start_index, len(df)):
        mom = momentum[i - 1]
        vol_today = volume[i - 1]
        vol_avg = avg_volume[i - 1]
        price = open_[i]
        is_green = close[i - 1] > open_[i - 1]

        # Entry condition
        if btc == 0 and mom > 0 and vol_today > volume_factor * vol_avg and is_green:
            entry_price = price
            btc = (cash * (1 - fee_rate)) / entry_price
            cash = 0
            hold_candles = 0
            entry_price_recorded = entry_price
        # Exit condition
        elif btc > 0:
            hold_candles += 1
            if mom < 0 or hold_candles >= max_hold_candles:
                exit_price = price
                cash_after = btc * exit_price * (1 - fee_rate)
                pct_gain = (exit_price * (1 - 2 * fee_rate) - entry_price_recorded) / entry_price_recorded * 100
                pct_gains.append(pct_gain)
                all_hold_candles.append(hold_candles)
                cash = cash_after
                btc = 0
                hold_candles = 0

        current_value = btc * price if btc > 0 else cash
        equity_curve.append(current_value)

    # Compute max drawdown
    equity_curve = np.array(equity_curve)
    if equity_curve.size > 0:
        peak = np.maximum.accumulate(equity_curve)
        drawdowns = (peak - equity_curve) / peak
        max_drawdown = np.max(drawdowns) * 100
    else:
        max_drawdown = 0

    # Compute average hold time in hours
    avg_hold = np.mean(all_hold_candles) * interval_minutes / 60 if all_hold_candles else 0

    return pct_gains, max_drawdown, avg_hold

def run_backtest_for(symbol_interval):
    """
    Performs walk-forward backtesting for a given symbol and interval.
    Returns the best parameter set based on Sharpe ratio and capital efficiency.
    """
    symbol, interval = symbol_interval

    # Construct path to historical data
    filepath = os.path.join(base_folder, symbol.lower(), f"{symbol}_{interval}_last1000_data.json")
    if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
        return f"Skipping missing or empty file: {filepath}", None

    # Load historical data
    df = pd.read_json(filepath, lines=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    df = df.sort_index()

    interval_minutes = interval_to_minutes(interval)
    candles_per_day = 1440 / interval_minutes

    all_results = []

    # Test multiple random parameter combinations
    for param_set in generate_random_param_combinations():
        test_gains, test_trades, test_drawdowns = [], [], []
        test_trades_per_day, test_avg_holds, test_avg_trade_gains = [], [], []
        test_sharpes = []

        split_size = len(df) // (NUM_SPLITS + 1)
        # Walk-forward testing: split the dataset into sequential chunks
        for i in range(NUM_SPLITS):
            df_test = df.iloc[split_size * (i + 1): split_size * (i + 2)].copy()
            df_test['return'] = df_test['close'].pct_change()
            df_test['momentum'] = df_test['close'].pct_change(periods=param_set['momentum_window'])
            df_test['volatility'] = df_test['return'].rolling(window=param_set['volatility_window']).std()
            df_test['avg_volume'] = df_test['volume'].rolling(window=param_set['volatility_window']).mean()

            pct_gains, max_dd, avg_hold = backtest(df_test, interval_minutes=interval_minutes, **param_set)
            if not pct_gains:
                continue

            # Compute metrics
            num_trades = len(pct_gains)
            test_days = len(df_test) / candles_per_day
            geo_avg_gain = (np.prod([(1 + g / 100) for g in pct_gains]) ** (1 / num_trades) - 1) * 100
            gain_per_day = geo_avg_gain * (num_trades / test_days)
            sharpe = compute_sharpe(pct_gains)

            test_gains.append(gain_per_day)
            test_trades.append(num_trades)
            test_drawdowns.append(max_dd)
            test_trades_per_day.append(num_trades / test_days)
            test_avg_holds.append(avg_hold)
            test_avg_trade_gains.append(geo_avg_gain)
            test_sharpes.append(sharpe)

        # Aggregate metrics across splits
        if len(test_gains) == NUM_SPLITS:
            sorted_trade_gains = sorted(test_avg_trade_gains, reverse=True)
            if sorted_trade_gains[0] > 5 * sorted_trade_gains[1]:
                continue  # Ignore anomalous runs

            avg_gain_per_day = np.mean(test_gains)
            avg_gain_per_trade = np.mean(test_avg_trade_gains)
            avg_hold_hours = np.mean(test_avg_holds)
            avg_sharpe = np.mean(test_sharpes)
            capital_efficiency_score = avg_gain_per_trade / avg_hold_hours if avg_hold_hours > 0 else 0

            all_results.append({
                'params': param_set,
                'avg_gain_per_day': avg_gain_per_day,
                'avg_gain_per_trade': avg_gain_per_trade,
                'avg_hold_hours': avg_hold_hours,
                'capital_efficiency_score': capital_efficiency_score,
                'avg_sharpe': avg_sharpe,
                'split_metrics': list(zip(
                    test_gains, test_trades, test_drawdowns,
                    test_trades_per_day, test_avg_holds, test_avg_trade_gains, test_sharpes))
            })

    # Filter out poor performing parameter sets
    all_results = [
        r for r in all_results
        if r['avg_gain_per_trade'] > 0.25 and
           sum(x[1] for x in r['split_metrics']) > 100 and
           r['capital_efficiency_score'] > 0.05
    ]

    if not all_results:
        return f"No strong performers for {symbol} {interval}", None

    # Keep only top 10% by Sharpe ratio
    threshold_sharpe = np.percentile([r['avg_sharpe'] for r in all_results], 90)
    filtered_results = [r for r in all_results if r['avg_sharpe'] >= threshold_sharpe]

    if not filtered_results:
        return f"No parameters meet top 10% Sharpe ratio for {symbol} {interval}", None

    best_result = max(filtered_results, key=lambda r: r['avg_sharpe'])
    return None, (symbol, interval, best_result)

# === Google Sheets Helper Functions ===

def create_sheet_if_missing(service, spreadsheet_id, sheet_name):
    """
    Creates a new sheet in the spreadsheet if it doesn't exist.
    """
    sheets_metadata = service.spreadsheets().get(spreadsheetId=spreadsheet_id).execute()
    sheets = sheets_metadata.get('sheets', [])
    sheet_names = [s['properties']['title'] for s in sheets]

    if sheet_name not in sheet_names:
        requests = [{
            'addSheet': {
                'properties': {'title': sheet_name}
            }
        }]
        body = {'requests': requests}
        service.spreadsheets().batchUpdate(spreadsheetId=spreadsheet_id, body=body).execute()

def write_results_to_sheet(service, results_by_symbol):
    """
    Writes all backtest results to a single Google Sheet tab, sorted by capital efficiency.
    """
    all_rows = []

    def make_values_block(symbol, interval, res):
        """
        Converts a backtest result into rows suitable for Google Sheets.
        """
        params = res['params']
        metrics = res['split_metrics']
        capital_efficiency_score = res.get('capital_efficiency_score', 0.0)
        avg_sharpe = res.get('avg_sharpe', 0.0)

        rows = [[f"Backtest Result for {symbol} {interval}"]]
        for k, v in params.items():
            suffix = " candles" if k == 'max_hold_candles' else ""
            rows.append([k, f"{v}{suffix}"])

        rows.append([""])
        rows.append(["-- Averages Over All Splits --"])
        rows.append(["Avg Gain/Day", f"{np.mean([x[0] for x in metrics]):.2f}%"])
        rows.append(["Avg Gain/Trade", f"{np.mean([x[5] for x in metrics]):.2f}%"])
        rows.append(["Avg Trades", f"{np.mean([x[1] for x in metrics]):.2f}"])
        rows.append(["Avg Trades/Day", f"{np.mean([x[3] for x in metrics]):.2f}"])
        rows.append(["Avg Max DD", f"{np.mean([x[2] for x in metrics]):.2f}%"])
        rows.append(["Avg Hold (hrs)", f"{np.mean([x[4] for x in metrics]):.2f}"])
        rows.append(["Capital Efficiency", f"{capital_efficiency_score:.4f}%"])
        rows.append(["Sharpe Ratio", f"{avg_sharpe:.4f}"])

        rows.append([""])
        rows.append(["-- Walk-Forward Test Splits --"])
        rows.append(["Split #", "Gain/Day", "Gain/Trade", "Trades", "Trades/Day", "Max DD", "Hold (hrs)", "Sharpe"])

        for i, m in enumerate(metrics):
            rows.append([i+1, f"{m[0]:.2f}%", f"{m[5]:.2f}%", m[1], f"{m[3]:.2f}", f"{m[2]:.2f}%", f"{m[4]:.2f}", f"{m[6]:.4f}"])

        rows.append([""])
        return capital_efficiency_score, rows

    # Flatten all results and sort by capital efficiency
    flat_results = []
    for _, result in results_by_symbol:
        if result is None:
            continue
        symbol, interval, best_result = result
        score, rows = make_values_block(symbol, interval, best_result)
        flat_results.append((score, rows))
    flat_results.sort(key=lambda x: x[0], reverse=True)

    for _, rows in flat_results:
        all_rows.extend(rows)

    # Clear existing sheet range and write new data
    clear_range = "'v4'!A:Z"
    service.spreadsheets().values().clear(
        spreadsheetId=SPREADSHEET_ID, range=clear_range).execute()

    write_range = "'v4'!A1"
    body = {"values": all_rows}
    service.spreadsheets().values().update(
        spreadsheetId=SPREADSHEET_ID,
        range=write_range,
        valueInputOption="RAW",
        body=body
    ).execute()

# === Authentication ===

def authenticate_google_sheets():
    """
    Handles authentication with Google Sheets API, including token refresh.
    """
    creds = None
    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)
    else:
        flow = InstalledAppFlow.from_client_secrets_file(creds_path, SCOPES)
        creds = flow.run_local_server(port=0)
        with open(token_path, 'w') as token:
            token.write(creds.to_json())

    try:
        service = build('sheets', 'v4', credentials=creds)
        _ = service.spreadsheets().get(spreadsheetId=SPREADSHEET_ID).execute()
    except RefreshError:
        print("Token expired or revoked. Re-authenticating...")
        if os.path.exists(token_path):
            os.remove(token_path)
        flow = InstalledAppFlow.from_client_secrets_file(creds_path, SCOPES)
        creds = flow.run_local_server(port=0)
        with open(token_path, 'w') as token:
            token.write(creds.to_json())
        service = build('sheets', 'v4', credentials=creds)

    return service

# === Main ===

def main():
    """
    Main script: authenticates with Google Sheets, runs backtests in parallel, 
    selects best results per symbol, and writes them to the sheet.
    """
    service = authenticate_google_sheets()

    # Create all symbol/interval combinations
    combos = [(symbol, interval) for symbol in symbols for interval in intervals]

    # Run backtests in parallel using 3 processes
    with Pool(processes=3) as pool:
        results = pool.map(run_backtest_for, combos)

    # Print errors if any
    for error, _ in results:
        if error:
            print(error)

    # Keep only the best backtest per symbol
    best_per_symbol = {}
    for error, result in results:
        if error or result is None:
            continue
        symbol, interval, best_result = result
        existing = best_per_symbol.get(symbol)
        if existing is None or best_result['capital_efficiency_score'] > existing[2]['capital_efficiency_score']:
            best_per_symbol[symbol] = (symbol, interval, best_result)

    filtered_results = [(None, best_per_symbol[s]) for s in best_per_symbol]

    # Write results to Google Sheet
    write_results_to_sheet(service, filtered_results)

if __name__ == '__main__':
    main()
