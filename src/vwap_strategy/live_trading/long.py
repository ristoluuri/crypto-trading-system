import sys, os, json, asyncio, logging, time
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from binance.client import Client

# === Base paths setup ===
base_dir = os.path.dirname(os.path.abspath(__file__))  # Directory of this script
PROJECT_ROOT = os.path.abspath(os.path.join(base_dir, "..", ".."))  # Project root
sys.path.append(PROJECT_ROOT)  # Add project root to path for imports

# Import trading helpers and shared variables
from helpers.live_trading.vwap_trading_helper import (
    SymbolBot,  # Class representing a trading bot for a single symbol
    CombinedWebSocketManagerBuffered,  # Manager to handle multiple websocket bots
    OPEN_POSITIONS_COUNT,  # Shared counter for open positions
    OPEN_POSITIONS_LOCK,  # Lock for thread-safe position updates
    MAX_OPEN_POSITIONS,  # Maximum allowed concurrent positions
)

from helpers.live_trading.futures_utils import load_symbol_quantities  # Load per-symbol quantity/precision info

# Use Windows-compatible event loop if on Windows
if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# === Load environment variables ===
env_path = os.path.join(base_dir, PROJECT_ROOT, ".env")
load_dotenv(env_path)

# === Logging setup ===
log_file = "live_trading_long.log"
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

# === Load trading configurations from backtest results ===
param_path = os.path.join(PROJECT_ROOT, "vwap_strategy", "backtest", "longParams.json")
with open(param_path) as f:
    symbol_configs = json.load(f)

# Load symbol quantities and price precisions
symbol_quantities = load_symbol_quantities(base_dir)

# Check that Binance account is in hedge mode (required for long/short positions)
mode = client.futures_get_position_mode()
if not mode["dualSidePosition"]:
    logging.info("You are NOT in hedge mode. Please enable it.")
    sys.exit(1)

# Thread pool executor to run synchronous Binance calls asynchronously
executor = ThreadPoolExecutor(max_workers=10)

# === Helper function: round price to symbol precision ===
def round_price_to_precision(price: float, precision: int) -> float:
    return float(f"{price:.{precision}f}")

# === Submit Binance orders asynchronously using executor ===
async def submit_order(order_func, **params):
    loop = asyncio.get_running_loop()
    # Run synchronous Binance function in thread executor
    return await loop.run_in_executor(executor, lambda: order_func(**params))

# === Core trading strategy logic per bot ===
async def run_strategy(bot, current=None):
    """
    Executes the VWAP long trading strategy for a single bot.
    Handles entry, exit, stop-loss, cooldowns, and position counting.
    """
    global OPEN_POSITIONS_COUNT

    # Skip if not enough candles for VWAP calculation
    if len(bot.candles_df) < bot.max_hold_candles + bot.vwap_window:
        return

    # Compute rolling VWAP
    df = bot.compute_rolling_vwap(bot.candles_df)
    if df.empty:
        return

    # Current candle information
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
        # === ENTRY LOGIC ===
        if not bot.in_position and price < vw * (1 - bot.vwap_distance_threshold):
            async with OPEN_POSITIONS_LOCK:
                # Skip if max open positions reached
                if OPEN_POSITIONS_COUNT >= MAX_OPEN_POSITIONS:
                    logging.info(f"[{time_str}] Max open positions reached, skipping {bot.symbol}")
                    return
                OPEN_POSITIONS_COUNT += 1

            # Place market buy order
            await submit_order(bot.client.futures_create_order,
                symbol=bot.upper_symbol,
                side="BUY",
                type="MARKET",
                quantity=bot.quantity,
                positionSide="LONG",
                timestamp=server_time,
                recvWindow=5000
            )

            logging.info(f"[{time_str} | {bot.interval}] {bot.upper_symbol} BUY @ {price:.7f} VWAP: {vw:.7f}")

            # Update bot state
            bot.in_position = True
            bot.entry_time = current.name
            bot.entry_price = price
            bot.last_trade_candle = current.name

            # Place stop-loss order
            stop_price = bot.entry_price * (1 - bot.stop_loss)
            rounded_stop_price = round_price_to_precision(stop_price, bot.price_precision)
            stop_loss_order = await submit_order(bot.client.futures_create_order,
                symbol=bot.upper_symbol,
                side="SELL",
                type="STOP_MARKET",
                quantity=bot.quantity,
                stopPrice=f"{rounded_stop_price:.{bot.price_precision}f}",
                positionSide="LONG",
                timestamp=server_time,
                recvWindow=5000
            )
            bot.stop_loss_order_id = stop_loss_order.get("orderId")

        # === EXIT LOGIC ===
        elif bot.in_position:
            stop_price = bot.entry_price * (1 - bot.stop_loss)

            # 1️⃣ Stop-loss exit
            if current["low"] <= stop_price:
                logging.info(f"[{time_str} | {bot.interval}] {bot.upper_symbol} STOP LOSS HIT @ {current['low']:.7f} <= {stop_price:.7f}")

                # Reset bot state
                bot.in_position = False
                bot.entry_time = None
                bot.entry_price = None
                bot.last_trade_candle = current.name
                bot.stop_loss_order_id = None
                bot.last_exit_candle = current.name

                async with OPEN_POSITIONS_LOCK:
                    OPEN_POSITIONS_COUNT -= 1  # Decrement open positions count
                return

            # 2️⃣ VWAP or max-hold exit
            if bot.entry_time is not None:
                held = (df.index >= bot.entry_time).sum() - 1
                should_exit = price > vw * (1 - bot.vwap_exit_threshold) or held >= bot.max_hold_candles

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

                    # Sell market
                    await submit_order(bot.client.futures_create_order,
                        symbol=bot.upper_symbol,
                        side="SELL",
                        type="MARKET",
                        quantity=bot.quantity,
                        positionSide="LONG",
                        timestamp=server_time,
                        recvWindow=5000
                    )

                    logging.info(f"[{time_str} | {bot.interval}] {bot.upper_symbol} SELL @ {price:.7f} | Reason: EXIT | Held {held} candles")

                    # Reset bot state
                    bot.in_position = False
                    bot.entry_time = None
                    bot.entry_price = None
                    bot.last_trade_candle = current.name
                    bot.stop_loss_order_id = None
                    bot.last_exit_candle = current.name

                    async with OPEN_POSITIONS_LOCK:
                        OPEN_POSITIONS_COUNT -= 1  # Decrement open positions count

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

        # Merge backtest parameters with trading config
        merged_config = {
            "interval": config["interval"],
            "quantity_to_trade": trade_qty,
            "stop_loss": config.get("params", {}).get("stop_loss"),
            "daily_capital_efficiency": config.get("daily_capital_efficiency", 0),
            **config["params"]
        }

        bot = SymbolBot(symbol, merged_config, base_dir, client, data_folder_name="websocketDataLong")
        bot.load_candles()  # Load previously saved candles
        await bot.download_historical()  # Download missing historical data

        # Assign bot-specific properties
        bot.stop_loss = merged_config["stop_loss"]
        bot.price_precision = price_precision
        bot.cooldown_candles = merged_config.get("cooldown_candles", 0)
        bot.last_exit_candle = None

        # Attach strategy function
        bot.strategy_fn = run_strategy

        bots.append(bot)

    # Start the websocket manager to run bots
    manager = CombinedWebSocketManagerBuffered(bots, strategy_fn=run_strategy)
    await manager.start()

# Entry point
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Stopped by user.")
