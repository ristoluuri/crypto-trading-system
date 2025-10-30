import os
import json
import time
import logging
import asyncio
import pandas as pd
import numpy as np
import websockets
from websockets.exceptions import ConnectionClosedError
from collections import defaultdict
from datetime import timedelta

GLOBAL_LEVERAGE = 5

class SymbolBot:
    def __init__(self, symbol, config, base_dir, client, data_folder_name="websocketDataLong"):
        self.symbol = symbol.lower()
        self.upper_symbol = symbol.upper()
        self.interval = config["interval"]
        self.quantity = config["quantity_to_trade"]
        self.vwap_distance_threshold = config["vwap_distance_threshold"]
        self.vwap_exit_threshold = config["vwap_exit_threshold"]
        self.max_hold_candles = config["max_hold_candles"]
        self.vwap_window = config.get("vwap_window")
        self.daily_capital_efficiency = config.get("daily_capital_efficiency")
        self.client = client

        self.data_dir = os.path.join(base_dir, data_folder_name)
        os.makedirs(self.data_dir, exist_ok=True)
        self.csv_file = os.path.join(self.data_dir, f"{self.symbol}_{self.interval}_candles.csv")
        self.rest_url = f"https://fapi.binance.com/fapi/v1/klines?symbol={self.upper_symbol}&interval={self.interval}&limit=500"

        self.candles_df = pd.DataFrame()
        self.entry_time = None
        self.last_trade_candle = None
        self.in_position = False
        self.stop_loss_order_id = None
        self.entry_price = None
        self.last_exit_candle = None
        self.price_precision = 2
        self.cooldown_candles = 0

        try:
            self.client.futures_change_leverage(symbol=self.upper_symbol, leverage=GLOBAL_LEVERAGE)
        except Exception as e:
            logging.info(f"Error setting leverage for {self.upper_symbol}: {e}")

        self.tick_size = self.get_tick_size()

    def get_tick_size(self):
        try:
            info = self.client.futures_exchange_info()
            symbol_info = next(s for s in info["symbols"] if s["symbol"] == self.upper_symbol)
            for f in symbol_info["filters"]:
                if f["filterType"] == "PRICE_FILTER":
                    return float(f["tickSize"])
        except Exception as e:
            logging.info(f"Error getting tick size for {self.upper_symbol}: {e}")
        return 0.01  # fallback

    def load_candles(self):
        if os.path.exists(self.csv_file):
            df = pd.read_csv(self.csv_file)
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df.set_index("timestamp", inplace=True)
            self.candles_df = df

    async def download_historical(self):
        import aiohttp  # local import to keep deps minimal here
        async with aiohttp.ClientSession() as session:
            async with session.get(self.rest_url) as resp:
                data = await resp.json()
                df = pd.DataFrame(data).iloc[:, :6]
                df.columns = ["timestamp", "open", "high", "low", "close", "volume"]
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
                df.set_index("timestamp", inplace=True)
                df = df.astype(float)
                self.candles_df = df
                self.candles_df.reset_index().to_csv(self.csv_file, index=False)

    def update_candles(self, candle):
        t = candle["timestamp"]
        row = pd.DataFrame([candle]).set_index("timestamp")
        if t in self.candles_df.index:
            self.candles_df.loc[t] = row.iloc[0]
        else:
            self.candles_df = pd.concat([self.candles_df, row])
            if len(self.candles_df) > 500:
                self.candles_df = self.candles_df.iloc[-500:]

        if not self.candles_df.index.is_monotonic_increasing:
            self.candles_df.sort_index(kind='stable', inplace=True)

    def save_candles(self):
        self.candles_df.reset_index().to_csv(self.csv_file, index=False)

    def compute_rolling_vwap(self, df):
        close = df["close"].values
        volume = df["volume"].values
        window = self.vwap_window

        cum_vol = np.cumsum(volume)
        cum_pv = np.cumsum(close * volume)

        vol_window = cum_vol[window-1:] - np.concatenate(([0], cum_vol[:-window]))
        pv_window = cum_pv[window-1:] - np.concatenate(([0], cum_pv[:-window]))

        with np.errstate(divide='ignore', invalid='ignore'):
            vwap = np.where(vol_window != 0, pv_window / vol_window, np.nan)

        full_vwap = np.empty(len(df))
        full_vwap[:] = np.nan
        full_vwap[window-1:] = vwap

        df = df.copy()
        df["vwap"] = full_vwap
        df.dropna(inplace=True)
        return df

    def interval_to_timedelta(self):
        unit = self.interval[-1]
        amount = int(self.interval[:-1])
        if unit == 'm': return timedelta(minutes=amount)
        elif unit == 'h': return timedelta(hours=amount)
        elif unit == 'd': return timedelta(days=amount)
        else: raise ValueError("Invalid interval")


# Buffer and synchronization globals for signal buffering
signal_buffer = defaultdict(list)  # {timestamp: list of (bot, current_df_row)}
buffer_lock = asyncio.Lock()
buffer_timeouts = {}  # {timestamp: asyncio.Task}
OPEN_POSITIONS_COUNT = 0
OPEN_POSITIONS_LOCK = asyncio.Lock()
MAX_OPEN_POSITIONS = 10

async def process_buffered_signals(timestamp, bots):
    async with buffer_lock:
        signals = signal_buffer.pop(timestamp, [])
        buffer_timeouts.pop(timestamp, None)

    if not signals:
        return

    # Count *all* bots currently in position, not just those in this batch
    open_positions_count = sum(1 for bot in bots if bot.in_position)
    open_slots = MAX_OPEN_POSITIONS - open_positions_count

    logging.info(f"Open positions: {open_positions_count}, Open slots: {open_slots}, Total signals: {len(signals)}")

    for bot, _ in signals:
        logging.info(f"Bot {bot.symbol} in_position={bot.in_position} daily_cap_eff={bot.daily_capital_efficiency}")

    # Sort signals by priority, e.g. daily capital efficiency descending
    def signal_score(item):
        bot, current = item
        return bot.daily_capital_efficiency

    signals.sort(key=signal_score, reverse=True)

    to_process = []

    for bot, current in signals:
        if bot.in_position:
            to_process.append((bot, current))
        else:
            if open_slots > 0:
                to_process.append((bot, current))
                open_slots -= 1

    await asyncio.gather(*(bot.strategy_fn(bot, current) for bot, current in to_process))


class CombinedWebSocketManager:
    def __init__(self, bots, strategy_fn):
        self.bots = bots
        self.strategy_fn = strategy_fn
        self.stream_to_bot = {f"{bot.symbol}@kline_{bot.interval}": bot for bot in bots}
        self.retry_delays = [0.1, 0.2, 0.5, 1, 2, 5]
        self.retry_count = 0


class CombinedWebSocketManagerBuffered(CombinedWebSocketManager):
    def __init__(self, bots, strategy_fn):
        super().__init__(bots, strategy_fn)
        self.symbols = set(bot.symbol for bot in bots)

    async def start(self):
        streams = "/".join(self.stream_to_bot.keys())
        url = f"wss://fstream.binance.com/stream?streams={streams}"

        while True:
            try:
                async with websockets.connect(url, ping_interval=20, ping_timeout=20) as ws:
                    logging.info(f"[Combined WS] Connected to {len(self.bots)} streams")
                    self.retry_count = 0

                    while True:
                        msg = await ws.recv()
                        data = json.loads(msg)
                        stream = data.get("stream")
                        kline = data.get("data", {}).get("k")

                        if not stream or not kline:
                            continue

                        bot = self.stream_to_bot.get(stream)
                        if not bot:
                            continue

                        candle = {
                            "timestamp": pd.to_datetime(kline["t"], unit="ms", utc=True),
                            "open": float(kline["o"]),
                            "high": float(kline["h"]),
                            "low": float(kline["l"]),
                            "close": float(kline["c"]),
                            "volume": float(kline["v"])
                        }

                        bot.update_candles(candle)

                        if kline["x"]:
                            bot.save_candles()

                            vwap_df = bot.compute_rolling_vwap(bot.candles_df)
                            ts = candle["timestamp"]

                            if ts in vwap_df.index:
                                vwap_value = vwap_df.loc[ts, "vwap"]
                            else:
                                vwap_value = None

                            candle_with_vwap = candle.copy()
                            candle_with_vwap["vwap"] = vwap_value

                            # Append to buffer
                            async with buffer_lock:
                                signal_buffer[ts].append((bot, candle_with_vwap))

                            # Immediately process all buffered signals for this timestamp
                            async with buffer_lock:
                                # Make a copy to avoid modifying while processing
                                buffered_signals = list(signal_buffer[ts])
                                signal_buffer.pop(ts, None)
                                buffer_timeouts.pop(ts, None)

                            if buffered_signals:
                                # Determine open slots
                                open_positions_count = sum(1 for b in self.bots if b.in_position)
                                open_slots = MAX_OPEN_POSITIONS - open_positions_count

                                # Sort by daily capital efficiency
                                buffered_signals.sort(key=lambda item: item[0].daily_capital_efficiency, reverse=True)

                                to_process = []
                                for b, c in buffered_signals:
                                    if b.in_position:
                                        to_process.append((b, c))
                                    else:
                                        if open_slots > 0:
                                            to_process.append((b, c))
                                            open_slots -= 1

                                # Run strategy for each selected bot
                                await asyncio.gather(*(b.strategy_fn(b, c) for b, c in to_process))

            except (ConnectionClosedError, Exception) as e:
                delay = self.retry_delays[min(self.retry_count, len(self.retry_delays) - 1)]
                logging.info(f"[Combined WS] Connection error: {e}. Reconnecting in {delay}s...")
                await asyncio.sleep(delay)
                self.retry_count += 1
