import asyncio
import sys
import os
import aiohttp
import csv
import requests
from datetime import datetime, timedelta, timezone

# Windows asyncio fix
if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# -----------------------------
# Helper: Check first candle asynchronously
# -----------------------------
async def has_two_years_data(session, symbol, interval="1d"):
    url = "https://fapi.binance.com/fapi/v1/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": 1,
        "startTime": 0
    }
    try:
        async with session.get(url, params=params, timeout=10) as resp:
            data = await resp.json()
            if not data:
                return None
            first_candle_ts = data[0][0]
            first_candle_date = datetime.fromtimestamp(first_candle_ts / 1000, tz=timezone.utc)
            two_years_ago = datetime.now(timezone.utc) - timedelta(days=365*2)
            if first_candle_date <= two_years_ago:
                return symbol
    except Exception as e:
        print(f"Error checking {symbol}: {e}")
    return None

# -----------------------------
# Step 1: Get exchange info
# -----------------------------
exchange_info_url = "https://fapi.binance.com/fapi/v1/exchangeInfo"
exchange_info = requests.get(exchange_info_url).json()

usdt_symbols = [
    s['symbol'] for s in exchange_info['symbols']
    if s['contractType'] == 'PERPETUAL' and s['quoteAsset'] == 'USDT' and s['status'] == 'TRADING'
]

print(f"\nTotal USDT perpetual pairs found: {len(usdt_symbols)}")

# -----------------------------
# Step 1b: Filter symbols with >=2 years of data concurrently
# -----------------------------
async def filter_symbols(symbols):
    valid_symbols = []
    async with aiohttp.ClientSession() as session:
        tasks = [has_two_years_data(session, sym) for sym in symbols]
        for coro in asyncio.as_completed(tasks):
            result = await coro
            if result:
                valid_symbols.append(result)
    return valid_symbols

print("\nChecking historical data availability (async, faster)...")
valid_symbols = asyncio.run(filter_symbols(usdt_symbols))
print(f"\nSymbols with at least 2 years of data: {len(valid_symbols)}")

# -----------------------------
# Step 2: Get 24h ticker stats
# -----------------------------
ticker_url = "https://fapi.binance.com/fapi/v1/ticker/24hr"
ticker_data = requests.get(ticker_url).json()

# -----------------------------
# Step 3: Filter stats & extract quoteVolume
# -----------------------------
volume_data = []
total_volume = 0.0
for ticker in ticker_data:
    symbol = ticker['symbol']
    if symbol in valid_symbols:
        volume = float(ticker['quoteVolume'])
        volume_data.append((symbol, volume))
        total_volume += volume

# -----------------------------
# Step 4: Sort by volume
# -----------------------------
volume_data.sort(key=lambda x: x[1], reverse=True)

# -----------------------------
# Step 5: Save to CSV one folder up with UTF-8
# -----------------------------
base_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(base_dir, "..", "usdt_futures_2yr.csv")

with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Symbol", "24h Quote Volume (USDT)"])
    writer.writerows(volume_data)

print(f"\nSaved to {csv_path}")
print(f"\nTotal USDT Futures processed: {len(valid_symbols)}")
