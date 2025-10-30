import requests
import csv
import os

# === Base directory ===
base_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(base_dir, ".."))
csv_path = os.path.join(parent_dir, "usdt_futures_symbols.csv")

# Step 1: Get exchange info for filtering
exchange_info_url = "https://fapi.binance.com/fapi/v1/exchangeInfo"
exchange_info = requests.get(exchange_info_url).json()

# Filter for USDT perpetual pairs
usdt_symbols = [
    s['symbol'] for s in exchange_info['symbols']
    if s['contractType'] == 'PERPETUAL' and s['quoteAsset'] == 'USDT' and s['status'] == 'TRADING'
]

print(f"\nTotal USDT perpetual pairs found: {len(usdt_symbols)}")

# Step 2: Get 24h ticker stats
ticker_url = "https://fapi.binance.com/fapi/v1/ticker/24hr"
ticker_data = requests.get(ticker_url).json()

# Step 3: Filter stats to only USDT perpetuals and extract quoteVolume
volume_data = []
total_volume = 0.0
for ticker in ticker_data:
    symbol = ticker['symbol']
    if symbol in usdt_symbols:
        volume = float(ticker['quoteVolume'])  # in USDT
        volume_data.append((symbol, volume))
        total_volume += volume

# Step 4: Sort by volume (highest to lowest)
volume_data.sort(key=lambda x: x[1], reverse=True)

# Step 5: Save to CSV one folder up from script
with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Symbol", "24h Quote Volume (USDT)"])
    writer.writerows(volume_data)
    
print(f"\nSaved to {csv_path}")

# Step 6: Print top 10 and bottom 10
print("\nTop 10 USDT Futures by 24h Volume:")
for symbol, volume in volume_data[:10]:
    print(f"{symbol}: {volume:,.2f} USDT")

print("\nBottom 10 USDT Futures by 24h Volume:")
for symbol, volume in volume_data[-10:]:
    print(f"{symbol}: {volume:,.2f} USDT")

# Step 7: Print total volume sum
print(f"\nTotal 24h Quote Volume for all USDT perpetual pairs: {total_volume:,.2f} USDT")
print(f"\nTotal USDT Futures processed: {len(usdt_symbols)}")
