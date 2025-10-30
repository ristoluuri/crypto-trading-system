import requests
import csv
import os

# Base folder of the script
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "..", "usdc_futures_symbols_usdc.csv")  # save 1 folder up

# Step 1: Get exchange info for filtering
exchange_info_url = "https://fapi.binance.com/fapi/v1/exchangeInfo"
exchange_info = requests.get(exchange_info_url).json()

# Filter for USDC perpetual pairs
usdc_symbols = [
    s['symbol'] for s in exchange_info['symbols']
    if s['contractType'] == 'PERPETUAL' and s['quoteAsset'] == 'USDC' and s['status'] == 'TRADING'
]

print(f"\nTotal USDC perpetual pairs found: {len(usdc_symbols)}")

# Step 2: Get 24h ticker stats
ticker_url = "https://fapi.binance.com/fapi/v1/ticker/24hr"
ticker_data = requests.get(ticker_url).json()

# Step 3: Filter stats to only USDC perpetuals and extract quoteVolume
volume_data = []
total_volume = 0.0
for ticker in ticker_data:
    symbol = ticker['symbol']
    if symbol in usdc_symbols:
        volume = float(ticker['quoteVolume'])  # in USDC
        volume_data.append((symbol, volume))
        total_volume += volume

# Step 4: Sort by volume (highest to lowest)
volume_data.sort(key=lambda x: x[1], reverse=True)

# Step 5: Save to CSV
with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Symbol", "24h Quote Volume (USDC)"])
    writer.writerows(volume_data)

print(f"\nSaved to {csv_path}")

# Step 6: Print top 10 and bottom 10
print("\nTop 10 USDC Futures by 24h Volume:")
for symbol, volume in volume_data[:10]:
    print(f"{symbol}: {volume:,.2f} USDC")

print("\nBottom 10 USDC Futures by 24h Volume:")
for symbol, volume in volume_data[-10:]:
    print(f"{symbol}: {volume:,.2f} USDC")

# Step 7: Print total volume sum
print(f"\nTotal 24h Quote Volume for all USDC perpetual pairs: {total_volume:,.2f} USDC")
print(f"\nTotal USDC Futures processed: {len(usdc_symbols)}")
