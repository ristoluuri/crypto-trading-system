import requests
import pandas as pd
import numpy as np
import time
import os
from datetime import datetime, timezone

script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "..", "symbols.csv")

def get_binance_futures_data(ticker='BNBUSDT', interval='1d', start_ms=None, end_ms=None):
    columns = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time',
               'qav', 'num_trades', 'taker_base_vol', 'taker_quote_vol', 'ignore']
    usecols = ['open', 'high', 'low', 'close', 'volume', 'qav',
               'num_trades', 'taker_base_vol', 'taker_quote_vol', 'timestamp']

    df = pd.DataFrame()
    print(f'Downloading {interval} {ticker} OHLCV data...', end=' ')

    while True:
        url = f'https://fapi.binance.com/fapi/v1/klines?symbol={ticker}&interval={interval}&limit=1000&startTime={start_ms}&endTime={end_ms}'
        response = requests.get(url, headers={'Cache-Control': 'no-cache', "Pragma": "no-cache"})
        data = response.json()

        if 'code' in data:
            print(f"\nError from Binance API: {data['msg']}")
            break

        if data:
            data = pd.DataFrame(data, columns=columns, dtype=np.float64)
            # update start time to continue from last candle
            start_ms = int(data.open_time.iloc[-1]) + 1
            data['timestamp'] = pd.to_datetime(data.open_time, unit='ms').dt.strftime("%Y-%m-%d %H:%M:%S")
            data.index = data['timestamp']
            data = data[usecols]
            df = pd.concat([df, data], axis=0)
        else:
            print("\nNo more data.")
            break

        # avoid rate limit bans
        time.sleep(0.05)

    print('Done.')
    df.index = pd.to_datetime(df.index)
    return df


def interval_to_ms(interval):
    unit = interval[-1]
    amount = int(interval[:-1])
    if unit == 'm':
        return amount * 60 * 1000
    elif unit == 'h':
        return amount * 60 * 60 * 1000
    elif unit == 'd':
        return amount * 24 * 60 * 60 * 1000
    else:
        raise ValueError(f"Unsupported interval unit: {unit}")

def save_data_for_symbol_interval(symbol, interval):
    save_dir = f'data/{symbol.lower()}'
    os.makedirs(save_dir, exist_ok=True)

    # end time = now
    end_ms = int(pd.Timestamp.utcnow().timestamp() * 1000)

    # start time = Jan 1, 2021
    start_time = datetime(2021, 1, 1, tzinfo=timezone.utc)
    start_ms = int(start_time.timestamp() * 1000)

    data = get_binance_futures_data(ticker=symbol, interval=interval, start_ms=start_ms, end_ms=end_ms)

    filename = f'{symbol}_{interval}_last1000_data.json'
    filepath = os.path.join(save_dir, filename)
    data.to_json(filepath, orient='records', lines=True)
    print(f'Data saved to {filepath}')


if __name__ == "__main__":
    symbols_df = pd.read_csv(csv_path)
    symbols = symbols_df['Symbol'].tolist()
    intervals = ['15m', '30m', '1h', '4h']

    from concurrent.futures import ThreadPoolExecutor

    def fetch_symbol_interval(symbol, interval):
        save_data_for_symbol_interval(symbol, interval)

    # Use multiple threads to fetch data concurrently
    max_threads = 3  # you can increase if you want
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        for symbol in symbols:
            for interval in intervals:
                executor.submit(fetch_symbol_interval, symbol, interval)
