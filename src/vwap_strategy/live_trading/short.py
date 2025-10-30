import sys, os, json, asyncio, logging, time
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from binance.client import Client

# === Base paths setup ===
base_dir = os.path.dirname(os.path.abspath(__file__))  # Directory of this script
PROJECT_ROOT = os.path.abspath(os.path.join(base_dir, "..", ".."))  # Project root
sys.path.append(PROJECT_ROOT)  # Add project root to Python path for imports

# Import trading helpers and shared variables
from helpers.live_trading.vwap_trading_helper import (
    SymbolBot,  # Trading bot for a single symbol
    CombinedWebSocketManagerBuffered,  # Websocket manager for multiple bots
    OPEN_POSITIONS_COUNT,  # Shared counter of open positions
    OPEN_POSITIONS_LOCK,  # Lock to safely update OPEN_POSITIONS_COUNT
    MAX_OPEN_POSITIONS,  # Maximum concurrent positions allowed
)

from helpers.live_trading.futures_utils import load_symbol_quantities  # Load per-symbol quantity and precision

# Use Windows-compatible event loop if on Windows
if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# === Load environment variables ===
env_path = os.path.join(base_dir, PROJECT_ROOT, ".env")
load_dotenv(env_path)

# === Logging setup ===
log_file = "live_trading_short.log"
log_dir = os.path.join(base_dir, "logs")
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(log_dir, log_file)),  # Log to file
        logging.StreamHandler()  # Log to console
    ]
)

# === Binance API client setup ===
api_key = os.getenv("BINANCE_API_KEY_SUB")
api_secret = os.getenv("BINANCE_API_SECRET_SUB")
client = Client(api_key, api_secret)

# === Load short trading strategy configs from backtest results ===
param_path = os.path.join(PROJECT_ROOT, "vwap_strategy", "backtest", "shortParams.json")
with open(param_path) as f:
    symbol_configs = json.load(f)

# Load per-symbol trading quantities and price precisions
symbol_quantities = load_symbol_quantities(base_dir)

# Ensure account is in hedge mode (required for long/short positions)
mode = client.futures_get_position_mode()
if not mode["dualSidePosition"]:
    logging.info("You are NOT in hedge mode. Please enable it.")
    sys.exit(1)

# === Thread pool executor for running synchronous Binance calls asynchronously ===
executor = ThreadPoolExecutor(max_workers=10)

# === Helper function: round price to correct precision for symbol ===
def round_price_to_precision(price: float, precision: int) -> float:
    return float(f"{price:.{precision}f}")

# === Submit Binance orders asynchronously using executor ===
async def submit_order(order_func, **params):
    loop = asyncio.get_running_loop()
    # Run synchronous Binance API call in thread pool
    return await loop.run_in_executor(executor, lambda: order_func(**params))

# === Core short trading strategy logic per bot ===
async def run_strategy(bot, current=None):
    """
    Executes the VWAP SHORT trading strategy for a single bot.
    Handles entry, exit, stop-loss, cooldown, and open positions count.
    """
    global OPEN_POSITIONS_COUNT

    # Skip if not enough candles to calculate VWAP
    if len(bot.candles_df) < bot.max_hold_candles + bot.vwap_window:
        return

    # Compute rolling VWAP
    df = bot.compute_rolling_vwap(bot.candles_df)
    if df.empty:
        return

    # Current candle info
    current = df.iloc[-1]
    vw = current["vwap"]
    price = current["close"]
    time_str = (current.name + bot.interval_to_timedelta()).strftime('%Y-%m-%d %H:%M UTC')

    # Skip if already processed this candle
    if bot.last_trade_candle == current.name and not bot.in_position:
        return

    # Skip if still in cooldown after last exit
    if bot.last_exit_candle is not None:
        candles_since_exit = (df.index >= bot.last_exit_candle).sum() - 1
        if candles_since_exit < bot.cooldown_candles:
            return

    server_time = int(time.time() * 1000)

    try:
        # === ENTRY LOGIC for SHORT ===
        if not bot.in_position and price > vw * (1 + bot.vwap_distance_threshold):
            async with OPEN_POSITIONS_LOCK:
                # Skip if max open positions reached
                if OPEN_POSITIONS_COUNT >= MAX_OPEN_POSITIONS:
                    logging.info(f"[{time_str}] Max open positions reached, skipping {bot.symbol}")
                    return
                OPEN_POSITIONS_COUNT += 1

            # Place market sell order to open short
            await submit_order(bot.client.futures_create_order,
                symbol=bot.upper_symbol,
                side="SELL",
                type="MARKET",
                quantity=bot.quantity,
                positionSide="SHORT",
                timestamp=server_time,
                recvWindow=5000
            )

            logging.info(f"[{time_str} | {bot.interval}] {bot.upper_symbol} SHORT @ {price:.7f} VWAP: {vw:.7f}")

            # Update bot state
            bot.in_position = True
            bot.entry_time = current.name
            bot.entry_price = price
            bot.last_trade_candle = current.name

            # Place stop-loss order (buy to cover if price rises)
            stop_price = bot.entry_price * (1 + bot.stop_loss)
            rounded_stop_price = round_price_to_precision(stop_price, bot.price_precision)
            stop_loss_order = await submit_order(bot.client.futures_create_order,
                symbol=bot.upper_symbol,
                side="BUY",
                type="STOP_MARKET",
                quantity=bot.quantity,
                stopPrice=f"{rounded_stop_price:.{bot.price_precision}f}",
                positionSide="SHORT",
                timestamp=server_time,
                recvWindow=5000
            )
            bot.stop_loss_order_id = stop_loss_order.get("orderId")

        # === IN-POSITION LOGIC for SHORT ===
        elif bot.in_position:
            stop_price = bot.entry_price * (1 + bot.stop_loss)

            # 1️⃣ Stop-loss hit (price moved against short)
            if current["high"] >= stop_price:
                logging.info(f"[{time_str} | {bot.interval}] {bot.upper_symbol} STOP LOSS HIT @ {current['high']:.7f} >= {stop_price:.7f}")

                # Reset bot state
                bot.in_position = False
                bot.entry_time = None
                bot.entry_price = None
                bot.last_trade_candle = current.name
                bot.stop_loss_order_id = None
                bot.last_exit_candle = current.name

                async with OPEN_POSITIONS_LOCK:
                    OPEN_POSITIONS_COUNT -= 1

                return  # Skip VWAP/time exit if SL hit

            # 2️⃣ VWAP or max-hold exit
            if bot.entry_time is not None:
                held = (df.index >= bot.entry_time).sum() - 1
                should_exit = price < vw * (1 + bot.vwap_exit_threshold) or held >= bot.max_hold_candles

                if should_exit:
                    # Cancel stop-loss order
                    if bot.stop_loss_order_id:
                        try:
                            await submit_order(bot.client.futures_cancel_order,
                                symbol=bot.upper_symbol,
                                orderId=bot.stop_loss_order_id,
                                timestamp=server_time,
                                recvWindow=5000
                            )
                            logging.info(f"[{time_str} | {bot.interval}] Cancelled stop loss order {bot.stop_loss_order_id}")
                        except Exception as e:
                            logging.warning(f"[{time_str}] Error cancelling stop loss for {bot.upper_symbol}: {e}")

                    # Close short position (buy to cover)
                    await submit_order(bot.client.futures_create_order,
                        symbol=bot.upper_symbol,
                        side="BUY",
                        type="MARKET",
                        quantity=bot.quantity,
                        positionSide="SHORT",
                        timestamp=server_time,
                        recvWindow=5000
                    )

                    logging.info(f"[{time_str} | {bot.interval}] {bot.upper_symbol} BUY (COVER) @ {price:.7f} | Reason: EXIT | Held {held} candles")

                    # Reset bot state
                    bot.in_position = False
                    bot.entry_time = None
                    bot.entry_price = None
                    bot.last_trade_candle = current.name
                    bot.stop_loss_order_id = None
                    bot.last_exit_candle = current.name

                    async with OPEN_POSITIONS_LOCK:
                        OPEN_POSITIONS_COUNT -= 1

    except Exception as e:
        logging.info(f"[{time_str}] Order error for {bot.upper_symbol}: {e}")

# === Main async routine ===
async def main():
    bots = []
    # Initialize SymbolBot instances for all symbols
    for config in symbol_configs:
        symbol = config["symbol"]
        trade_qty = symbol_quantities.get(symbol, {}).get("trade_qty")
        price_precision = symbol_quantities.get(symbol, {}).get("price_precision", 2)

        # Merge backtest parameters with live config
        merged_config = {
            "interval": config["interval"],
            "quantity_to_trade": trade_qty,
            "stop_loss": config.get("params", {}).get("stop_loss"),
            "daily_capital_efficiency": config.get("daily_capital_efficiency", 0),
            **config["params"]
        }

        # Create bot instance
        bot = SymbolBot(symbol, merged_config, base_dir, client, data_folder_name="websocketDataShort")
        bot.load_candles()  # Load saved historical candles
        await bot.download_historical()  # Download missing candles

        # Assign bot-specific properties
        bot.stop_loss = merged_config["stop_loss"]
        bot.price_precision = price_precision
        bot.cooldown_candles = merged_config.get("cooldown_candles", 0)
        bot.last_exit_candle = None
        bot.strategy_fn = run_strategy

        bots.append(bot)

    # Start websocket manager to run all bots concurrently
    manager = CombinedWebSocketManagerBuffered(bots, strategy_fn=run_strategy)
    await manager.start()

# Entry point
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Stopped by user.")
