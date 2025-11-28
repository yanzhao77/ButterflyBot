#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import json
import time
from collections import deque
from datetime import datetime
import pandas as pd
import joblib
import websocket
from binance.um_futures import UMFutures
from data.features import add_features
from config_stage1 import *
import threading


class LiveTraderBinanceFutures:
    def __init__(self, api_key=None, api_secret=None, test_mode=True, leverage=5):
        self.test_mode = test_mode
        self.symbol = SYMBOL.replace('/', '')
        self.interval = TIMEFRAME
        self.klines = deque(maxlen=200)
        self.current_kline = None
        self.leverage = leverage

        # 持仓状态
        self.position = None
        self.entry_price = 0
        self.entry_bars = 0
        self.highest_price = 0
        self.lowest_price = 0

        # 资金
        self.cash = INITIAL_CASH
        self.equity = INITIAL_CASH
        self.trades = []

        # Binance Futures 客户端
        if not self.test_mode:
            self.client = UMFutures(key=api_key, secret=api_secret)
            try:
                self.client.change_leverage(symbol=self.symbol, leverage=self.leverage)
            except Exception as e:
                print(f"⚠️ 设置杠杆失败: {e}")
        else:
            self.client = None

        # 加载模型
        self.model = self.load_model()

        # 获取交易规则
        self.min_qty, self.qty_precision, self.price_precision = self.get_symbol_info()

        # 启动账户监控线程
        self.monitor_thread = threading.Thread(target=self.account_monitor_loop, daemon=True)
        self.monitor_thread.start()

    def load_model(self):
        import glob
        model_files = glob.glob(f'{MODEL_DIR}/*{MODEL_TYPE}.pkl')
        if not model_files:
            raise FileNotFoundError(f"未找到模型文件: {MODEL_DIR}/*{MODEL_TYPE}.pkl")
        model_file = sorted(model_files)[-1]
        print(f"加载模型: {model_file}")
        return joblib.load(model_file)

    def get_symbol_info(self):
        if self.test_mode:
            return 0.01, 2, 2
        info = self.client.get_symbol_info(self.symbol)
        lot_size = next(f for f in info['filters'] if f['filterType'] == 'LOT_SIZE')
        price_filter = next(f for f in info['filters'] if f['filterType'] == 'PRICE_FILTER')
        min_qty = float(lot_size['minQty'])
        qty_precision = int(round(-np.log10(float(lot_size['stepSize']))))
        price_precision = int(round(-np.log10(float(price_filter['tickSize']))))
        return min_qty, qty_precision, price_precision

    def load_historical_data(self):
        df = pd.read_csv(f'{DATA_DIR}/binance_DOGE_USDT_15m.csv')
        df = df.tail(200)
        for _, row in df.iterrows():
            # 处理不同的时间戳格式
            timestamp = row['timestamp']
            # 如果是日期时间字符串，则转换为Unix时间戳
            if isinstance(timestamp, str):
                # 尝试解析带时区的日期时间字符串
                try:
                    dt = pd.to_datetime(timestamp)
                    timestamp_ms = int(dt.timestamp() * 1000)  # 转换为毫秒
                except:
                    # 如果解析失败，默认为0
                    timestamp_ms = 0
            else:
                # 如果已经是数值，则确保是整数
                timestamp_ms = int(timestamp)

            self.klines.append({
                'timestamp': timestamp_ms,  # 确保时间戳是整数（毫秒）
                'open': row['open'],
                'high': row['high'],
                'low': row['low'],
                'close': row['close'],
                'volume': row['volume']
            })
        print(f"✅ 加载历史K线: {len(self.klines)}根")

    def handle_kline(self, msg):
        if msg['e'] != 'kline':
            return
        k = msg['k']
        kline_data = {
            'timestamp': int(k['t']),  # 确保时间戳是整数
            'open': float(k['o']),
            'high': float(k['h']),
            'low': float(k['l']),
            'close': float(k['c']),
            'volume': float(k['v'])
        }
        if k['x']:
            self.klines.append(kline_data)
            self.current_kline = None
            self.on_bar_closed(kline_data)
        else:
            self.current_kline = kline_data
            if self.position:
                self.check_exit(kline_data['close'])

    def on_bar_closed(self, kline):
        price = kline['close']
        if self.position:
            self.entry_bars += 1
        # 确保价格是浮点数
        self.check_entry(float(price)) if not self.position else self.check_exit(float(price))

    def check_entry(self, current_price):
        df = pd.DataFrame(list(self.klines))
        features_df = add_features(df)
        if features_df.empty:
            return
        latest_features = features_df.iloc[-1]
        # 修复特征列选择，只排除timestamp，保留close作为特征
        feature_cols = [c for c in features_df.columns if c not in ['timestamp']]
        X = latest_features[self.model.feature_names_in_].values.reshape(1, -1)
        prob = self.model.predict_proba(X)[0][1]

        signal = None
        if prob > 0.5 + CONFIDENCE_THRESHOLD:
            signal = 'long'
        elif prob < 0.5 - CONFIDENCE_THRESHOLD:
            signal = 'short'

        if signal:
            self.open_position(signal, float(current_price), prob)  # 确保current_price是浮点数

    def calculate_amount(self, price):
        position_value = self.cash * MAX_POSITION_RATIO * self.leverage
        raw_amount = position_value / price
        amount = max(round(raw_amount, self.qty_precision), self.min_qty)
        return amount

    def open_position(self, direction, price, prob):
        amount = self.calculate_amount(price)
        print(f"\n🔔 开仓: {direction.upper()} 价格: {price:.5f} 数量: {amount} 概率: {prob:.4f}")

        if not self.test_mode:
            try:
                side = 'BUY' if direction == 'long' else 'SELL'
                order = self.client.new_order(symbol=self.symbol, side=side, type='MARKET', quantity=amount)
                print("✅ 开仓成功:", order)
            except Exception as e:
                print("❌ 开仓失败:", e)
                return

        self.position = direction
        self.entry_price = price
        self.entry_bars = 0
        self.highest_price = price
        self.lowest_price = price
        self.cash -= amount * price / self.leverage

    def check_exit(self, current_price):
        if not self.position:
            return
        self.highest_price = max(self.highest_price, float(current_price))  # 确保current_price是浮点数
        self.lowest_price = min(self.lowest_price, float(current_price))  # 确保current_price是浮点数
        pnl_pct = (float(current_price) - self.entry_price) / self.entry_price if self.position == 'long' else (
                                                                                                                           self.entry_price - float(
                                                                                                                       current_price)) / self.entry_price
        exit_reason = None

        if pnl_pct >= TAKE_PROFIT_PCT:
            exit_reason = '止盈'
        elif pnl_pct <= -STOP_LOSS_PCT:
            exit_reason = '止损'
        elif self.entry_bars >= TIME_STOP_BARS:
            exit_reason = '时间止损'
        else:
            df = pd.DataFrame(list(self.klines))
            features_df = add_features(df)
            if not features_df.empty:
                latest_features = features_df.iloc[-1]
                # 修复特征列选择，只排除timestamp，保留close作为特征
                feature_cols = [c for c in features_df.columns if c not in ['timestamp']]
                X = latest_features[feature_cols].values.reshape(1, -1)
                prob = self.model.predict_proba(X)[0][1]
                if self.position == 'long' and prob < 0.5 - CONFIDENCE_THRESHOLD:
                    exit_reason = '信号反转(做空)'
                elif self.position == 'short' and prob > 0.5 + CONFIDENCE_THRESHOLD:
                    exit_reason = '信号反转(做多)'

        if exit_reason:
            self.close_position(float(current_price), exit_reason)  # 确保current_price是浮点数

    def close_position(self, price, reason):
        amount = self.calculate_amount(self.entry_price)
        pnl = (price - self.entry_price) * amount if self.position == 'long' else (self.entry_price - price) * amount
        fee = amount * price * COMMISSION_PCT * 2
        pnl -= fee

        print(f"\n🔔 平仓: {self.position.upper()} 价格: {price:.5f} 原因: {reason} 盈亏: {pnl:.2f}")

        if not self.test_mode:
            try:
                side = 'SELL' if self.position == 'long' else 'BUY'
                self.client.new_order(symbol=self.symbol, side=side, type='MARKET', quantity=amount)
            except Exception as e:
                print("❌ 平仓失败:", e)

        self.cash += pnl + amount * price / self.leverage
        self.equity = self.cash
        self.trades.append(
            {'type': self.position, 'entry': self.entry_price, 'exit': price, 'pnl': pnl, 'reason': reason})

        self.position = None
        self.entry_price = 0
        self.entry_bars = 0
        self.highest_price = 0
        self.lowest_price = 0

    def account_monitor_loop(self):
        """账户实时监控线程，每15秒打印"""
        while True:
            if not self.test_mode and self.client:
                try:
                    account = self.client.account()
                    positions = account['positions']
                    pos = next((p for p in positions if p['symbol'] == self.symbol), None)
                    balance = float(account['totalWalletBalance'])
                    margin = float(account['totalMarginBalance'])
                    pnl = float(pos['unrealizedProfit']) if pos else 0
                    side = pos['positionSide'] if pos and float(pos['positionAmt']) != 0 else 'NONE'
                    amt = float(pos['positionAmt']) if pos else 0
                    print(f"\n[账户监控] 时间: {datetime.now().strftime('%H:%M:%S')}")
                    print(f"  总余额: {balance:.2f} USDT  可用保证金: {margin:.2f} USDT")
                    print(f"  仓位: {side} 数量: {amt} 浮盈: {pnl:.2f} USDT")
                except Exception as e:
                    print(f"⚠️ 获取账户信息失败: {e}")
            time.sleep(1)

    def start(self):
        """启动实时行情 WebSocket，支持断线重连、心跳保活"""
        self.load_historical_data()
        ws_url = f"wss://stream.binance.com:9443/ws/{self.symbol.lower()}@kline_{self.interval}"

        def on_open(ws):
            print("✅ WebSocket已连接")
            # 输出当前交易对信息
            print(f"📊 当前交易对: {self.symbol}")
            print(f"📊 时间框架: {self.interval}")
            print(f"📊 杠杆倍数: {self.leverage}x")

        def on_close(ws, code, msg):
            print(f"🛑 WebSocket已关闭 code={code} msg={msg}")
            print("⏳ 3秒后尝试重连...")
            time.sleep(3)

        def on_error(ws, error):
            print(f"⚠️ WebSocket错误: {error}")

        def on_message(ws, msg):
            self.handle_kline(json.loads(msg))

        self.ws = websocket.WebSocketApp(
            ws_url,
            on_open=on_open,
            on_message=on_message,
            on_close=on_close,
            on_error=on_error
        )

        # 关键：保持 WebSocket 心跳+自动重连
        def run_ws():
            while True:
                try:
                    self.ws.run_forever(
                        ping_interval=20,  # 每20秒心跳
                        ping_timeout=10,  # 10秒内没回应则断开
                        reconnect=5,
                        http_proxy_host="127.0.0.1",
                        http_proxy_port=7890,
                        proxy_type="http"  # socks5: "socks5"
                    )
                except Exception as e:
                    print(f"❌ WebSocket连接异常: {e}")
                print("🔄 断线重连中...")
                time.sleep(5)

        threading.Thread(target=run_ws, daemon=True).start()

        # 主线程保持运行（否则程序直接退出）
        while True:
            time.sleep(1)


if __name__ == "__main__":
    import os

    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_API_SECRET")
    trader = LiveTraderBinanceFutures(api_key, api_secret, test_mode=True, leverage=5)
    trader.start()
