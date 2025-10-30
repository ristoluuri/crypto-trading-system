import os, json, time, logging
import numpy as np
import pandas as pd
from datetime import timedelta
import aiohttp
import websockets
from websockets.exceptions import ConnectionClosedError
import asyncio

GLOBAL_LEVERAGE = 5
MAX_OPEN_POSITIONS = 10  # 🔥 cap total concurrent positions

# Global state
OPEN_POSITIONS_COUNT = 0
OPEN_POSITIONS_LOCK = asyncio.Lock()


class SymbolBot:
    def __init__(self, symbol, config, base_dir, client, data_folder_name="websocketDataMomentum"):
        self.symbol = symbol.lower()
        self.upper_symbol = symbol.upper()
        self.interval = config["interval"]
        self.quantity = config["quantity_to_trade"]
        self.momentum_window = config["momentum_window"]
        self.volatility_window = config["volatility_window"]
        self.volume_factor = config["volume_factor"]
        self.max_hold_candles = config["max_hold_candles"]
        self.client = client

        self.data_dir = os.path.join(base_dir, data_folder_name)
        os.makedirs(self.data_dir, exist_ok=True)
        self.csv_file = os.path.join(self.data_dir, f"{self.symbol}_{self.interval}_candles.csv")
        self.rest_url = f"https://fapi.binance.com/fapi/v1/klines?symbol={self.upper_symbol}&interval={self.interval}&limit=500"

        self.candles_df = pd.DataFrame()
        self.entry_time = None
        self.last_trade_candle = None
        self.in_position = False

        try:
            self.client.futures_change_leverage(symbol=self.upper_symbol, leverage=GLOBAL_LEVERAGE)
        except Exception as e:
            logging.info(f"Error setting leverage for {self.upper_symbol}: {e}")

    async def download_historical(self):
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

    def interval_to_timedelta(self):
        unit = self.interval[-1]
        amount = int(self.interval[:-1])
        if unit == 'm': return timedelta(minutes=amount)
        elif unit == 'h': return timedelta(hours=amount)
        elif unit == 'd': return timedelta(days=amount)
        else: raise ValueError("Invalid interval")


class CombinedWebSocketManager:
    def __init__(self, bots, strategy_fn):
        self.bots = bots
        self.strategy_fn = strategy_fn
        self.stream_to_bot = {f"{bot.symbol}@kline_{bot.interval}": bot for bot in bots}
        self.retry_delays = [0.1, 0.2, 0.5, 1, 2, 5]
        self.retry_count = 0

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
                            asyncio.create_task(self.strategy_fn(bot))

            except (ConnectionClosedError, Exception) as e:
                delay = self.retry_delays[min(self.retry_count, len(self.retry_delays) - 1)]
                logging.info(f"[Combined WS] Connection error: {e}. Reconnecting in {delay}s...")
                await asyncio.sleep(delay)
                self.retry_count += 1
