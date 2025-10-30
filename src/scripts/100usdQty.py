import requests
import math
import pandas as pd
import json
import os
from decimal import Decimal, ROUND_UP

def get_decimal_places(value):
    """Return the number of decimal places for a given step size."""
    if value >= 1:
        return 0
    return abs(int(round(math.log10(value))))

def round_to_step(x, step):
    """Round down to the nearest allowed step size using Decimal to avoid float artifacts."""
    decimals = get_decimal_places(step)
    step_dec = Decimal(str(step))
    qty_dec = (Decimal(str(x)) / step_dec).to_integral_value(rounding=ROUND_UP) * step_dec
    return float(qty_dec.quantize(Decimal(f'1.{"0"*decimals}')))

def get_trade_qty_for_usd(symbol, usd_amount, exchange_info, prices_dict):
    # Find symbol info
    symbol_info = next((s for s in exchange_info['symbols'] if s['symbol'] == symbol), None)
    if not symbol_info:
        raise Exception(f"Symbol {symbol} not found")

    filters = {f['filterType']: f for f in symbol_info['filters']}
    step = float(filters['LOT_SIZE']['stepSize'])
    tick_size = float(filters['PRICE_FILTER']['tickSize'])

    # Safe min_notional extraction
    min_notional = 0
    if 'MIN_NOTIONAL' in filters:
        # Some versions use 'minNotional', some use 'notional'
        if 'minNotional' in filters['MIN_NOTIONAL']:
            min_notional = float(filters['MIN_NOTIONAL']['minNotional'])
        elif 'notional' in filters['MIN_NOTIONAL']:
            min_notional = float(filters['MIN_NOTIONAL']['notional'])

    # Get price
    if symbol not in prices_dict:
        raise Exception(f"Price for {symbol} not found")
    price = float(prices_dict[symbol])

    # Calculate quantity from USD amount
    qty = usd_amount / price
    qty = round_to_step(qty, step)  # round up to step size

    # Ensure minimum trade value is satisfied
    min_qty = max(step, round_to_step(min_notional / price, step))
    if qty < min_qty:
        qty = min_qty

    # Ensure trade qty is at least 1.2 × min_qty
    min_trade_qty = round_to_step(1.2 * min_qty, step)
    if qty < min_trade_qty:
        qty = min_trade_qty

    # Price precision
    price_precision = get_decimal_places(tick_size)
    spread = tick_size / price

    return qty, min_qty, step, tick_size, price_precision, price, spread

# Paths
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "..", "usdt_futures_symbols.csv")
json_path = os.path.join(script_dir, "..", "symbol_quantities100.json")

# Load symbols from CSV
symbols_df = pd.read_csv(csv_path)
symbols = symbols_df['Symbol'].tolist()

# Parameters
usd_trade_amount = 10
results = {}

# Fetch exchangeInfo once
exchange_info = requests.get('https://fapi.binance.com/fapi/v1/exchangeInfo').json()

# Fetch all mark prices once
prices_resp = requests.get('https://fapi.binance.com/fapi/v1/premiumIndex').json()
prices_dict = {item['symbol']: item['markPrice'] for item in prices_resp}

# Process each symbol
for symbol in symbols:
    try:
        qty, min_qty, step, tick_size, price_precision, price, spread = get_trade_qty_for_usd(
            symbol, usd_trade_amount, exchange_info, prices_dict
        )
        results[symbol] = {
            "trade_qty": qty,
            "min_qty": min_qty,
            "step_size": step,
            "tick_size": tick_size,
            "price_precision": price_precision,
            "price": price,
            "spread": spread
        }
    except Exception as e:
        results[symbol] = {"error": str(e)}

# Sort results by spread ascending (excluding errors)
sorted_results = dict(sorted(
    ((k, v) for k, v in results.items() if "spread" in v),
    key=lambda item: item[1]["spread"]
))

# Append error entries at the end
errors = {k: v for k, v in results.items() if "error" in v}
sorted_results.update(errors)

# Save sorted results to JSON
with open(json_path, "w") as f:
    json.dump(sorted_results, f, indent=2)

print(f"Saved sorted trade quantities + spreads for ${usd_trade_amount} positions to {json_path}")
