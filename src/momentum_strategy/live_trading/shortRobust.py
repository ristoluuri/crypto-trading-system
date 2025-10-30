import sys, os, json, asyncio, logging, time
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from binance.client import Client

# === Helper imports ===
base_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(base_dir, "..", ".."))
sys.path.append(PROJECT_ROOT)

from helpers.live_trading.mom_trading_helper import (
    SymbolBot,
    CombinedWebSocketManager,
    OPEN_POSITIONS_LOCK,
    MAX_OPEN_POSITIONS,
)
from helpers.live_trading.mom_trading_helper import OPEN_POSITIONS_COUNT  # shared counter
from helpers.live_trading.futures_utils import load_symbol_quantities

if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# === Env and logging ===
env_path = os.path.join(base_dir, PROJECT_ROOT, ".env")
load_dotenv(env_path)

log_file = "live_trading_robust_short.log"
log_dir = os.path.join(base_dir, "logs")
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(os.path.join(log_dir, log_file)),
              logging.StreamHandler()]
)

# === Binance client ===
api_key = os.getenv("BINANCE_API_KEY_SUB3")
api_secret = os.getenv("BINANCE_API_SECRET_SUB3")
client = Client(api_key, api_secret)

mode = client.futures_get_position_mode()
if not mode["dualSidePosition"]:
    logging.info("You are NOT in hedge mode. Please enable it.")
    sys.exit(1)

# === Load symbols ===
param_path = os.path.join(base_dir, "..", "backtest", "shortParamsRobust.json")
with open(param_path) as f:
    symbol_configs = json.load(f)  # this is a list, not a dict

symbol_quantities = load_symbol_quantities(base_dir)

# === ThreadPoolExecutor for async order submission ===
executor = ThreadPoolExecutor(max_workers=10)

async def submit_order(order_func, **params):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(executor, lambda: order_func(**params))

# === Momentum strategy using the helper SymbolBot ===
async def run_momentum_strategy(bot: SymbolBot):
    global OPEN_POSITIONS_COUNT

    if len(bot.candles_df) < bot.volatility_window + 2:
        return

    df = bot.candles_df.copy()
    df['return'] = df['close'].pct_change()
    df['momentum'] = df['close'].pct_change(periods=bot.momentum_window)
    df['volatility'] = df['return'].rolling(window=bot.volatility_window).std()
    df['avg_volume'] = df['volume'].rolling(window=bot.volatility_window).mean()

    current = df.iloc[-1]
    mom = current['momentum']
    vol_today = current['volume']
    vol_avg = current['avg_volume']
    price = current['close']
    is_red = current['close'] < current['open']  # changed for shorting
    time_str = (current.name + bot.interval_to_timedelta()).strftime('%Y-%m-%d %H:%M:%S UTC')

    if getattr(bot, "last_trade_candle", None) == current.name:
        return

    try:
        server_time = int(time.time() * 1000)
        if not getattr(bot, "in_position", False):
            # === SHORT ENTRY ===
            if mom < 0 and vol_today > bot.volume_factor * vol_avg and is_red:
                async with OPEN_POSITIONS_LOCK:
                    if OPEN_POSITIONS_COUNT >= MAX_OPEN_POSITIONS:
                        logging.info(f"[{time_str}] {bot.upper_symbol} skipped (max positions reached).")
                        return
                    OPEN_POSITIONS_COUNT += 1

                # Open SHORT (SELL first)
                await submit_order(bot.client.futures_create_order,
                                   symbol=bot.upper_symbol,
                                   side="SELL",
                                   type="MARKET",
                                   quantity=bot.quantity,
                                   positionSide="SHORT",  # changed to SHORT
                                   timestamp=server_time,
                                   recvWindow=5000)
                logging.info(f"[{time_str}] {bot.upper_symbol} SHORT SELL at ~{price} | Open positions: {OPEN_POSITIONS_COUNT}")
                bot.entry_time = current.name
                bot.last_trade_candle = current.name
                bot.in_position = True
        else:
            held = (df.index >= bot.entry_time).sum() - 1
            # === SHORT EXIT ===
            if mom > 0 or held >= bot.max_hold_candles:
                # Close SHORT (BUY back)
                await submit_order(bot.client.futures_create_order,
                                   symbol=bot.upper_symbol,
                                   side="BUY",
                                   type="MARKET",
                                   quantity=bot.quantity,
                                   positionSide="SHORT",  # closing the short
                                   timestamp=server_time,
                                   recvWindow=5000)
                async with OPEN_POSITIONS_LOCK:
                    OPEN_POSITIONS_COUNT = max(0, OPEN_POSITIONS_COUNT - 1)
                logging.info(f"[{time_str}] {bot.upper_symbol} SHORT BUY (cover) at ~{price} | Held {held} candles | Open positions: {OPEN_POSITIONS_COUNT}")
                bot.entry_time = None
                bot.last_trade_candle = current.name
                bot.in_position = False
    except Exception as e:
        logging.info(f"[{time_str}] {bot.upper_symbol} Order failed: {e}")

# === Main ===
async def main():
    bots = []
    for config in symbol_configs:  # iterate the list
        symbol = config['symbol']
        interval = config['interval']
        params = config.get("params", {})

        merged_config = {
            "interval": interval,
            "quantity_to_trade": symbol_quantities.get(symbol, {}).get("trade_qty", 0),
            "momentum_window": params.get("momentum_window"),
            "volatility_window": params.get("volatility_window"),
            "volume_factor": params.get("volume_factor"),
            "max_hold_candles": params.get("max_hold_candles"),
        }

        bot = SymbolBot(symbol, merged_config, base_dir, client, data_folder_name="websocketDataRobustShort")
        await bot.download_historical()
        bots.append(bot)

    manager = CombinedWebSocketManager(bots, strategy_fn=run_momentum_strategy)
    await manager.start()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Stopped by user.")
