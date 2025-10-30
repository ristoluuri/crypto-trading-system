import requests
import math
import pandas as pd
import json
import os

def ceil_to_step(x, step):
    return math.ceil(x / step) * step

def get_true_min_qty_and_step(symbol, exchange_info, prices_dict):
    # Use pre-fetched exchange info
    symbol_info = next((s for s in exchange_info['symbols'] if s['symbol'] == symbol), None)
    if not symbol_info:
        raise Exception(f"Symbol {symbol} not found")

    filters = {f['filterType']: f for f in symbol_info['filters']}
    step = float(filters['LOT_SIZE']['stepSize'])                      # quantity step size
    tick_size = float(filters['PRICE_FILTER']['tickSize'])             # price step size
    notional_min = float(filters['MIN_NOTIONAL']['notional'])          # minimum notional value

    # Get price from pre-fetched prices_dict
    if symbol not in prices_dict:
        raise Exception(f"Price for {symbol} not found")
    price = float(prices_dict[symbol])

    min_qty_by_notional = ceil_to_step(notional_min / price, step)
    true_min_qty = max(step, min_qty_by_notional)

    # Calculate number of decimals allowed for price (tick_size)
    price_precision = abs(int(round(math.log10(tick_size)))) if tick_size < 1 else 0

    return true_min_qty, step, tick_size, price_precision

def adjust_and_round(min_qty, step, multiplier=1.1):
    adjusted = min_qty * multiplier
    precision = abs(int(round(math.log10(step)))) if step < 1 else 0
    return round(adjusted, precision)

# === Base directory and parent folder ===
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, ".."))

# CSV input one folder up
csv_path = os.path.join(parent_dir, "usdt_futures_symbols.csv")
# JSON output one folder up
json_path = os.path.join(parent_dir, "symbol_quantities.json")

# Load symbols from CSV file using absolute path
symbols_df = pd.read_csv(csv_path)
symbols = symbols_df['Symbol'].tolist()

# Fetch exchange info once
print("Fetching exchange info...")
exchange_info = requests.get('https://fapi.binance.com/fapi/v1/exchangeInfo').json()
print("Exchange info fetched.")

# Fetch all mark prices once
print("Fetching all mark prices...")
prices_resp = requests.get('https://fapi.binance.com/fapi/v1/premiumIndex').json()
prices_dict = {item['symbol']: item['markPrice'] for item in prices_resp}
print("Mark prices fetched.")

# Collect results
results = {}

for symbol in symbols:
    try:
        min_qty, step, tick_size, price_precision = get_true_min_qty_and_step(symbol, exchange_info, prices_dict)
        adjusted_qty = adjust_and_round(min_qty, step)
        results[symbol] = {
            "min_qty": min_qty,
            "step_size": step,
            "trade_qty": adjusted_qty,
            "tick_size": tick_size,
            "price_precision": price_precision
        }
    except Exception as e:
        results[symbol] = {"error": str(e)}

# Save results to JSON file using absolute path
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print(f"Saved to {json_path}")
