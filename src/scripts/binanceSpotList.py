import requests
import csv
import os

# Base folder of the script
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "..", "usdc_spot_pairs_with_volume.csv")  # save 1 folder up

# Step 1: Get spot exchange info
exchange_info_url = "https://api.binance.com/api/v3/exchangeInfo"
exchange_info = requests.get(exchange_info_url).json()

# Step 2: Filter for spot pairs with USDC as quote asset and trading allowed
usdc_spot_pairs = [
    s['symbol'] for s in exchange_info['symbols']
    if s['status'] == 'TRADING' and s['quoteAsset'] == 'USDC' and s['isSpotTradingAllowed']
]

print(f"\nTotal spot pairs with USDC as quote asset: {len(usdc_spot_pairs)}")

# Step 3: Get 24h ticker data for all spot pairs
ticker_url = "https://api.binance.com/api/v3/ticker/24hr"
ticker_data = requests.get(ticker_url).json()

# Step 4: Match USDC spot pairs and get their 24h quote volume
volume_data = []
total_volume = 0.0
for ticker in ticker_data:
    symbol = ticker['symbol']
    if symbol in usdc_spot_pairs:
        volume = float(ticker['quoteVolume'])  # volume in USDC
        volume_data.append((symbol, volume))
        total_volume += volume

# Step 5: Save to CSV
with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Symbol", "24h Quote Volume (USDC)"])
    writer.writerows(volume_data)

print(f"Saved to {csv_path}")

# Step 6: Print summary
print("\nTop 10 USDC Spot Pairs by 24h Volume:")
for symbol, volume in sorted(volume_data, key=lambda x: x[1], reverse=True)[:10]:
    print(f"{symbol}: {volume:,.2f} USDC")

print(f"\nTotal 24h Quote Volume for all USDC spot pairs: {total_volume:,.2f} USDC")
