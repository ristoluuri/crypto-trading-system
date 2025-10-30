import requests
import csv
import os

# === Base directory and parent folder ===
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, ".."))
csv_path = os.path.join(parent_dir, "spot_pairs_non_stable_with_usd.csv")

# Define stablecoins & fiat currencies to exclude
STABLECOINS = {
    "USDT", "USDC", "TUSD", "BUSD", "FDUSD", "DAI", "EUR", "TRY", "AUD", "BRL", "GBP", "NGN", "ZAR", "IDRT",
    "RUB", "UAH", "VAI", "BVND", "CAD", "GHS", "CZK", "PLN", "ARS", "MXN", "COP", "CHF", "JPY", "KRW", "INR",
    "USD", "EURI", "RON"
}

# Step 1: Get all spot trading pairs
exchange_info_url = "https://api.binance.com/api/v3/exchangeInfo"
exchange_info = requests.get(exchange_info_url).json()

non_stable_pairs = []
quote_assets = set()

# Store symbol to quote asset mapping for accurate quote extraction
symbol_to_quote = {}

for symbol_info in exchange_info['symbols']:
    if symbol_info['status'] != 'TRADING':
        continue
    base = symbol_info['baseAsset']
    quote = symbol_info['quoteAsset']
    
    if base not in STABLECOINS and quote not in STABLECOINS:
        non_stable_pairs.append(symbol_info['symbol'])
        quote_assets.add(quote)
        symbol_to_quote[symbol_info['symbol']] = quote

print(f"Found {len(non_stable_pairs)} spot pairs without fiat or stablecoins.")

# Step 2: Get 24h ticker stats
ticker_url = "https://api.binance.com/api/v3/ticker/24hr"
ticker_data = requests.get(ticker_url).json()
ticker_map = {t['symbol']: t for t in ticker_data}

# Step 3: Fetch USDT prices for quote assets (to convert to USD)
price_map = {}
for quote in quote_assets:
    if quote == "USDT":
        price_map[quote] = 1.0
        continue
    usdt_pair = quote + "USDT"
    ticker = ticker_map.get(usdt_pair)
    if ticker:
        price_map[quote] = float(ticker['lastPrice'])

# Step 4: Convert quoteVolume to USD
volume_data = []
total_usd_volume = 0.0

for pair in non_stable_pairs:
    ticker = ticker_map.get(pair)
    if ticker:
        quote_asset = symbol_to_quote.get(pair)
        quote_volume = float(ticker['quoteVolume'])
        usd_price = price_map.get(quote_asset)
        if usd_price is not None:
            usd_volume = quote_volume * usd_price
            volume_data.append((pair, quote_volume, usd_volume))
            total_usd_volume += usd_volume

# Step 5: Sort by USD volume
volume_data.sort(key=lambda x: x[2], reverse=True)

# Step 6: Save to CSV one folder up from script
with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Symbol", "24h Quote Volume", "Estimated USD Volume"])
    writer.writerows(volume_data)

# Step 7: Display summary
print("\nTop 10 non-stablecoin Spot pairs by 24h USD Volume:")
for symbol, qvol, usdvol in volume_data[:10]:
    print(f"{symbol}: {usdvol:,.2f} USD")

print("\nBottom 10 non-stablecoin Spot pairs by 24h USD Volume:")
for symbol, qvol, usdvol in volume_data[-10:]:
    print(f"{symbol}: {usdvol:,.2f} USD")

print(f"\nTotal spot pairs without stablecoins/fiat (with USD conversion): {len(volume_data)}")
print(f"Total estimated 24h USD volume of all pairs: {total_usd_volume:,.2f} USD")
print(f"Saved to {csv_path}")
