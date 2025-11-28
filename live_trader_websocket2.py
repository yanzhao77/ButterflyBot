#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
重构后的实时交易脚本：
- 修复 predict vs predict_proba 问题并做 fallback
- 使用线程锁保护 shared state（position/cash/amount/trades）
- 开仓时保存 amount，平仓时复用，不重新计算
- stepSize 精度对齐（向下取整）
- 只在 K 线收盘时计算特征并缓存 latest_features（性能）
- 修复 pd.np -> np
- 改用 logging，account monitor sleep 调整为 15s
- WebSocket 重连由外层 run loop 控制（去掉不兼容参数）
"""

import os
import json
import time
import glob
import logging
import threading
from collections import deque
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
import websocket
from binance.um_futures import UMFutures

from data.features import add_features
from config_stage1 import *  # 假定包含 SYMBOL, TIMEFRAME, MODEL_DIR, MODEL_TYPE, INITIAL_CASH, MAX_POSITION_RATIO, COMMISSION_PCT, TAKE_PROFIT_PCT, STOP_LOSS_PCT, TIME_STOP_BARS, CONFIDENCE_THRESHOLD, DATA_DIR

# Logging 配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)


class LiveTraderBinanceFutures:
    def __init__(self, api_key=None, api_secret=None, test_mode=True, leverage=5, proxy=None):
        """
        proxy: dict like {'host': '127.0.0.1', 'port': 7890, 'type': 'http'}
        """
        self.test_mode = test_mode
        self.symbol = SYMBOL.replace('/', '')
        self.interval = TIMEFRAME
        self.klines = deque(maxlen=200)
        self.current_kline = None
        self.leverage = leverage

        # 持仓状态
        self.position = None
        self.entry_price = 0.0
        self.entry_bars = 0
        self.highest_price = 0.0
        self.lowest_price = 0.0

        # 资金与交易
        self.cash = INITIAL_CASH
        self.equity = INITIAL_CASH
        self.trades = []
        self.amount = None  # 开仓时记录的数量

        # 交易规则与精度
        self.min_qty = None
        self.qty_precision = None
        self.price_precision = None
        self.step_size = None  # 用于量化数量

        # 缓存特征（只在 bar close 时刷新）
        self.latest_features = None

        # 线程锁，保护 shared state
        self.lock = threading.Lock()

        # Binance 客户端
        self.client = None
        if not self.test_mode:
            try:
                self.client = UMFutures(key=api_key, secret=api_secret)
                self.client.change_leverage(symbol=self.symbol, leverage=self.leverage)
            except Exception as e:
                logging.warning(f"设置杠杆或初始化客户端失败: {e}")

        # 加载模型
        self.model = self.load_model()

        # 获取交易对 info（stepSize, tickSize 等）
        try:
            self.min_qty, self.qty_precision, self.price_precision = self.get_symbol_info()
        except Exception as e:
            logging.warning(f"获取交易对信息失败，使用默认精度: {e}")
            # 兜底默认值
            self.min_qty, self.qty_precision, self.price_precision = 0.01, 2, 2
            self.step_size = 0.01

        # 启动账户监控线程
        self.monitor_thread = threading.Thread(target=self.account_monitor_loop, daemon=True)
        self.monitor_thread.start()

        # web socket 相关
        self.ws = None
        self.proxy = proxy or {'host': '127.0.0.1', 'port': 7890, 'type': 'http'}

    def load_model(self):
        model_files = glob.glob(f'{MODEL_DIR}/*{MODEL_TYPE}.pkl')
        if not model_files:
            raise FileNotFoundError(f"未找到模型文件: {MODEL_DIR}/*{MODEL_TYPE}.pkl")
        model_file = sorted(model_files)[-1]
        logging.info(f"加载模型: {model_file}")
        model = joblib.load(model_file)
        return model

    def get_symbol_info(self):
        if self.test_mode:
            self.step_size = 0.01
            return 0.01, 2, 2

        info = self.client.get_symbol_info(self.symbol)
        lot_size = next(f for f in info['filters'] if f['filterType'] == 'LOT_SIZE')
        price_filter = next(f for f in info['filters'] if f['filterType'] == 'PRICE_FILTER')

        min_qty = float(lot_size['minQty'])
        step_size = float(lot_size['stepSize'])
        tick_size = float(price_filter['tickSize'])

        self.step_size = step_size
        qty_precision = int(round(-np.log10(step_size))) if step_size > 0 else 0
        price_precision = int(round(-np.log10(tick_size))) if tick_size > 0 else 0
        return min_qty, qty_precision, price_precision

    def load_historical_data(self):
        """
        加载历史K线，用于初始特征计算
        如果 timestamp 解析失败，会尝试跳过该行而不是设为0
        """
        csv_path = f'{DATA_DIR}/binance_DOGE_USDT_15m.csv'
        if not os.path.exists(csv_path):
            logging.warning(f"历史数据文件未找到: {csv_path}")
            return

        df = pd.read_csv(csv_path)
        df = df.tail(200)
        last_valid_ts = None
        for _, row in df.iterrows():
            timestamp = row.get('timestamp', None)

            # 逐步解析 timestamp，如果解析失败则跳过该行
            timestamp_ms = None
            if pd.isna(timestamp):
                timestamp_ms = None
            else:
                if isinstance(timestamp, str):
                    try:
                        dt = pd.to_datetime(timestamp, utc=True, errors='coerce')
                        if pd.isna(dt):
                            timestamp_ms = None
                        else:
                            timestamp_ms = int(dt.timestamp() * 1000)
                    except Exception:
                        timestamp_ms = None
                else:
                    try:
                        timestamp_ms = int(timestamp)
                        # 若认为是秒级时间戳（小于1e12），转为毫秒
                        if timestamp_ms < 1e12:
                            timestamp_ms = int(timestamp_ms * 1000)
                    except Exception:
                        timestamp_ms = None

            if timestamp_ms is None:
                # 尝试用上一个时间戳 + interval 推断（保守处理），否则跳过
                if last_valid_ts is not None:
                    # 时间框架解析为分钟数（例如 "15m" -> 15）
                    try:
                        minutes = int(''.join(filter(str.isdigit, self.interval)))
                        timestamp_ms = last_valid_ts + minutes * 60 * 1000
                    except Exception:
                        continue
                else:
                    # 没有可用时间戳，跳过
                    continue

            last_valid_ts = timestamp_ms
            self.klines.append({
                'timestamp': timestamp_ms,
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': float(row['volume'])
            })
        logging.info(f"✅ 加载历史K线: {len(self.klines)}根")

        # 计算初始特征缓存（如果可用）
        if len(self.klines) > 0:
            df_local = pd.DataFrame(list(self.klines))
            features_df = add_features(df_local)
            if not features_df.empty:
                self.latest_features = features_df.iloc[-1]

    def handle_kline(self, msg):
        """WebSocket 回调：处理 kline 消息（实时或已收盘）"""
        if msg.get('e') != 'kline':
            return
        k = msg['k']
        kline_data = {
            'timestamp': int(k['t']),
            'open': float(k['o']),
            'high': float(k['h']),
            'low': float(k['l']),
            'close': float(k['c']),
            'volume': float(k['v'])
        }

        # 如果是未闭合实时K线，更新 current_kline（不入队）
        if not k['x']:
            self.current_kline = kline_data
            # 实时价触发风险管理（只基于价格，不依赖新特征）
            if self.position:
                self.check_exit(kline_data['close'])
        else:
            # K线闭合：加入历史队列、更新特征缓存并触发 bar_closed 逻辑
            self.klines.append(kline_data)
            self.current_kline = None
            # 在 on_bar_closed 内只累加一次 entry_bars（避免重复）
            self.on_bar_closed(kline_data)

    def on_bar_closed(self, kline):
        """一根K线完全闭合时触发"""
        price = float(kline['close'])
        # 只有在已经持仓的情况下才增加 entry_bars（表示完整持仓 bar 数）
        if self.position:
            self.entry_bars += 1

        # 计算/更新特征缓存（仅在 bar 关闭时做一次，全量重算）
        df_local = pd.DataFrame(list(self.klines))
        features_df = add_features(df_local)
        if not features_df.empty:
            self.latest_features = features_df.iloc[-1]

        # 根据当前是否有仓位做不同的检查
        if not self.position:
            self.check_entry(price)
        else:
            self.check_exit(price)

    def get_model_prob(self, X):
        """统一获取模型给出的“做多”概率（1 表示多）"""
        if X is None:
            raise ValueError("输入 X 为空")
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X)
            # 确保返回结构正确
            try:
                return float(proba[0][1])
            except Exception:
                # idx fallback
                return float(proba[0][-1])
        elif hasattr(self.model, "predict"):
            # 退化处理：predict 返回 0/1 则直接返回该值（并记录警告）
            pred = self.model.predict(X)[0]
            logging.warning("模型不支持 predict_proba，使用 predict 的返回作为概率近似（0/1）")
            return float(pred)
        else:
            raise RuntimeError("模型既不支持 predict_proba 也不支持 predict")

    def check_entry(self, current_price):
        """检查是否满足开仓条件"""
        # 必须有最新特征（我们只在 bar close 计算并缓存）
        if self.latest_features is None:
            logging.debug("没有可用特征，跳过开仓判断")
            return

        # 取模型需要的特征顺序
        if hasattr(self.model, "feature_names_in_"):
            try:
                X = self.latest_features[self.model.feature_names_in_].values.reshape(1, -1)
            except Exception as e:
                logging.warning(f"从 latest_features 按 feature_names_in_ 取值失败: {e}")
                feature_cols = [c for c in self.latest_features.index if c != 'timestamp']
                X = self.latest_features[feature_cols].values.reshape(1, -1)
        else:
            feature_cols = [c for c in self.latest_features.index if c != 'timestamp']
            X = self.latest_features[feature_cols].values.reshape(1, -1)

        prob = self.get_model_prob(X)

        signal = None
        if prob > 0.5 + CONFIDENCE_THRESHOLD:
            signal = 'long'
        elif prob < 0.5 - CONFIDENCE_THRESHOLD:
            signal = 'short'

        if signal:
            self.open_position(signal, float(current_price), prob)

    def quantize_amount(self, raw_amount):
        """将原始数量向下取整为 step_size 的倍数，保证符合交易所 stepSize 规则"""
        if raw_amount <= 0:
            return self.min_qty
        if not self.step_size or self.step_size <= 0:
            # fallback: 以 qty_precision 四舍五入
            return max(round(raw_amount, self.qty_precision), self.min_qty)
        steps = int(np.floor(raw_amount / self.step_size))
        amt = steps * self.step_size
        # 四舍五入到显示精度以避免浮点误差
        return max(round(amt, self.qty_precision), self.min_qty)

    def calculate_amount(self, price):
        """根据当前可用资金、最大仓位比例、杠杆计算下单数量（不修改 state）"""
        with self.lock:
            position_value = self.cash * MAX_POSITION_RATIO * self.leverage
        raw_amount = position_value / price if price > 0 else 0
        amount = self.quantize_amount(raw_amount)
        return amount

    def open_position(self, direction, price, prob):
        """开仓（仅在持锁时修改 shared state）"""
        with self.lock:
            if self.position:
                logging.info("已有持仓，跳过开仓请求")
                return

            amount = self.calculate_amount(price)
            if amount < self.min_qty:
                logging.warning(f"计算出的下单数量过小：{amount}，跳过开仓")
                return

            logging.info(f"🔔 开仓请求: {direction.upper()} price={price:.8f} amount={amount} prob={prob:.4f}")

            if not self.test_mode and self.client:
                try:
                    side = 'BUY' if direction == 'long' else 'SELL'
                    order = self.client.new_order(symbol=self.symbol, side=side, type='MARKET', quantity=amount)
                    logging.info(f"✅ 开仓成功: {order}")
                except Exception as e:
                    logging.error(f"❌ 开仓失败: {e}")
                    return

            # 更新仓位状态（使用开仓时的 amount）
            self.position = direction
            self.entry_price = price
            self.entry_bars = 0
            self.highest_price = price
            self.lowest_price = price
            self.amount = amount
            # 资金变更（保证金计算是简化的：position_value / leverage 占用保证金）
            self.cash -= amount * price / self.leverage
            self.equity = self.cash  # 简化：不计算未实现盈亏 here

    def check_exit(self, current_price):
        """检查是否满足平仓条件（止盈/止损/时间止损/信号反转）"""
        if not self.position:
            return

        current_price = float(current_price)
        with self.lock:
            self.highest_price = max(self.highest_price, current_price)
            self.lowest_price = min(self.lowest_price, current_price)
            entry_price = self.entry_price
            position = self.position
            entry_bars = self.entry_bars

        # pnl_pct 以方向为准（long: (cur-entry)/entry, short: (entry-cur)/entry）
        pnl_pct = ((current_price - entry_price) / entry_price) if position == 'long' else ((entry_price - current_price) / entry_price)

        exit_reason = None
        if pnl_pct >= TAKE_PROFIT_PCT:
            exit_reason = '止盈'
        elif pnl_pct <= -STOP_LOSS_PCT:
            exit_reason = '止损'
        elif entry_bars >= TIME_STOP_BARS:
            exit_reason = '时间止损'
        else:
            # 信号反转检查：只有在我们有 latest_features 缓存时才计算（避免每次都重算特征）
            if self.latest_features is not None:
                # 准备 X
                if hasattr(self.model, "feature_names_in_"):
                    try:
                        X = self.latest_features[self.model.feature_names_in_].values.reshape(1, -1)
                    except Exception:
                        feature_cols = [c for c in self.latest_features.index if c != 'timestamp']
                        X = self.latest_features[feature_cols].values.reshape(1, -1)
                else:
                    feature_cols = [c for c in self.latest_features.index if c != 'timestamp']
                    X = self.latest_features[feature_cols].values.reshape(1, -1)

                prob = self.get_model_prob(X)
                if position == 'long' and prob < 0.5 - CONFIDENCE_THRESHOLD:
                    exit_reason = '信号反转(做空)'
                elif position == 'short' and prob > 0.5 + CONFIDENCE_THRESHOLD:
                    exit_reason = '信号反转(做多)'

        if exit_reason:
            self.close_position(current_price, exit_reason)

    def close_position(self, price, reason):
        """平仓（仅在持锁时修改 shared state）"""
        with self.lock:
            if not self.position:
                logging.info("没有持仓，跳过平仓")
                return

            amount = getattr(self, "amount", None)
            if amount is None:
                # fallback：若没有记录 amount，则使用 calculate_amount(entry_price) 作为兜底
                amount = self.calculate_amount(self.entry_price)

            pnl = (price - self.entry_price) * amount if self.position == 'long' else (self.entry_price - price) * amount
            fee = amount * price * COMMISSION_PCT * 2
            pnl -= fee

            logging.info(f"🔔 平仓: {self.position.upper()} price={price:.8f} amount={amount} reason={reason} pnl={pnl:.4f}")

            if not self.test_mode and self.client:
                try:
                    side = 'SELL' if self.position == 'long' else 'BUY'
                    order = self.client.new_order(symbol=self.symbol, side=side, type='MARKET', quantity=amount)
                    logging.info(f"✅ 平仓下单成功: {order}")
                except Exception as e:
                    logging.error(f"❌ 平仓下单失败: {e}")
                    # 不 return，仍更新本地仓位状态（根据你的业务逻辑可改）
            # 更新资金（简化计算）
            self.cash += pnl + amount * price / self.leverage
            self.equity = self.cash
            self.trades.append({
                'type': self.position,
                'entry': self.entry_price,
                'exit': price,
                'pnl': pnl,
                'reason': reason,
                'timestamp': datetime.utcnow().isoformat()
            })

            # 清理仓位
            self.position = None
            self.entry_price = 0.0
            self.entry_bars = 0
            self.highest_price = 0.0
            self.lowest_price = 0.0
            self.amount = None

    def account_monitor_loop(self):
        """账户实时监控线程，每15秒打印"""
        while True:
            if not self.test_mode and self.client:
                try:
                    account = self.client.account()
                    positions = account.get('positions', [])
                    pos = next((p for p in positions if p['symbol'] == self.symbol), None)
                    balance = float(account.get('totalWalletBalance', 0.0))
                    margin = float(account.get('totalMarginBalance', 0.0))
                    pnl = float(pos.get('unrealizedProfit', 0.0)) if pos else 0.0
                    side = pos.get('positionSide', 'NONE') if pos and float(pos.get('positionAmt', 0)) != 0 else 'NONE'
                    amt = float(pos.get('positionAmt', 0.0)) if pos else 0.0
                    logging.info(f"[账户监控] 时间: {datetime.now().strftime('%H:%M:%S')}")
                    logging.info(f"  总余额: {balance:.4f} USDT  可用保证金: {margin:.4f} USDT")
                    logging.info(f"  仓位: {side} 数量: {amt} 浮盈: {pnl:.4f} USDT")
                except Exception as e:
                    logging.warning(f"⚠️ 获取账户信息失败: {e}")
            time.sleep(15)  # 每15秒打印一次

    def start(self):
        """启动 WebSocket，外层循环负责重连"""
        self.load_historical_data()
        ws_url = f"wss://stream.binance.com:9443/ws/{self.symbol.lower()}@kline_{self.interval}"

        def on_open(ws):
            logging.info("✅ WebSocket已连接")
            logging.info(f"📊 当前交易对: {self.symbol}  时间框架: {self.interval}  杠杆: {self.leverage}x")

        def on_close(ws, code, msg):
            logging.warning(f"🛑 WebSocket已关闭 code={code} msg={msg}")
            # 不在这里直接重连，外层 run loop 会负责重连

        def on_error(ws, error):
            logging.error(f"⚠️ WebSocket错误: {error}")

        def on_message(ws, raw_msg):
            try:
                msg = json.loads(raw_msg)
                self.handle_kline(msg)
            except Exception as e:
                logging.exception(f"处理 websocket 消息失败: {e}")

        # create WebSocketApp
        self.ws = websocket.WebSocketApp(
            ws_url,
            on_open=on_open,
            on_message=on_message,
            on_close=on_close,
            on_error=on_error
        )

        # run loop with reconnect
        def run_ws_loop():
            while True:
                try:
                    # build kwargs for run_forever
                    run_kwargs = {
                        "ping_interval": 20,
                        "ping_timeout": 10,
                    }
                    # 如果需要代理则传入
                    if self.proxy:
                        run_kwargs.update({
                            "http_proxy_host": self.proxy.get('host'),
                            "http_proxy_port": int(self.proxy.get('port')),
                            "proxy_type": self.proxy.get('type', 'http')
                        })
                    logging.info("开始 run_forever()，若断开将自动重连")
                    self.ws.run_forever(**run_kwargs)
                except Exception as e:
                    logging.exception(f"❌ WebSocket 运行异常: {e}")
                logging.info("🔄 断线或异常后将于 5 秒后重连...")
                time.sleep(5)

        threading.Thread(target=run_ws_loop, daemon=True).start()

        # 主线程保持运行
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logging.info("收到 KeyboardInterrupt，准备退出...")
            try:
                if self.ws:
                    self.ws.close()
            except Exception:
                pass

if __name__ == "__main__":
    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_API_SECRET")
    proxy_cfg = {'host': '127.0.0.1', 'port': 7890, 'type': 'http'}  # 如需代理可修改
    trader = LiveTraderBinanceFutures(api_key=api_key, api_secret=api_secret, test_mode=True, leverage=5, proxy=proxy_cfg)
    trader.start()
