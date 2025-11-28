#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实时交易系统 - 基于 WebSocket
使用 python-binance WebSocket 获取实时行情，避免 CCXT 被封
使用 CCXT 执行交易订单
"""

import json
import os
import sys
import time
from collections import deque
from datetime import datetime

import ccxt
import joblib
import pandas as pd
import websocket

# 导入项目模块
from data.features import add_features
from config_stage1 import *


class LiveTraderWebSocket:
    """基于WebSocket的实时交易系统"""

    def __init__(self, api_key=None, api_secret=None, test_mode=True, proxy_host=None, proxy_port=None):
        """
        初始化交易系统
        
        Args:
            api_key: Binance API密钥
            api_secret: Binance API密钥
            test_mode: 测试模式（不执行真实交易）
        """
        self.test_mode = test_mode
        self.symbol = SYMBOL.replace('/', '')  # DOGEUSDT
        self.interval = TIMEFRAME  # 15m

        # WebSocket管理器
        self.twm = None

        # CCXT交易所（仅用于交易）
        if not test_mode:
            self.exchange = ccxt.binance({
                'apiKey': api_key,
                'secret': api_secret,
                'enableRateLimit': True,
            })
        else:
            self.exchange = None

        # K线数据缓存（保留最近200根）
        self.klines = deque(maxlen=200)
        self.current_kline = None

        # 加载模型
        self.model = self.load_model()

        # 交易状态
        self.position = None  # None, 'long', 'short'
        self.entry_price = 0
        self.entry_time = None
        self.entry_bars = 0
        self.highest_price = 0  # 用于跟踪止盈
        self.lowest_price = 0  # 用于跟踪止损

        # 冷却期
        self.cooldown_until = 0

        # 统计
        self.trades = []
        self.equity = INITIAL_CASH
        self.cash = INITIAL_CASH

        # 代理
        self.proxy_host = proxy_host
        self.proxy_port = proxy_port
        # 监控
        from stage1_monitor import Stage1Monitor
        self.monitor = Stage1Monitor()

        print("=" * 60)
        print("实时交易系统初始化")
        print("=" * 60)
        print(f"交易对: {SYMBOL}")
        print(f"时间框架: {TIMEFRAME}")
        print(f"测试模式: {test_mode}")
        print(f"初始资金: ${INITIAL_CASH}")
        print(f"模型: {MODEL_TYPE}")
        print("=" * 60)

    def load_model(self):
        """加载训练好的模型"""
        import glob
        model_files = glob.glob(f'{MODEL_DIR}/*{MODEL_TYPE}.pkl')
        if not model_files:
            raise FileNotFoundError(f"未找到模型文件: {MODEL_DIR}/*{MODEL_TYPE}.pkl")

        # 使用最新的模型
        model_file = sorted(model_files)[-1]
        print(f"加载模型: {model_file}")
        return joblib.load(model_file)

    def load_historical_data(self):
        """加载历史数据作为初始K线"""
        print("加载历史数据...")
        df = pd.read_csv(f'{DATA_DIR}/binance_DOGE_USDT_15m.csv')
        df = df.tail(200)  # 最近200根

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
                'volume': row['volume'],
            })

        print(f"✅ 加载了 {len(self.klines)} 根历史K线")

    def handle_kline(self, msg):
        """处理WebSocket K线消息"""
        data = json.loads(msg)
        if data.get('e') != 'kline':
            return

        kline = data['k']
        is_closed = kline['x']  # K线是否已完成

        # 构造K线数据，确保时间戳是数值类型
        kline_data = {
            'timestamp': int(kline['t']),  # 确保时间戳是整数
            'open': float(kline['o']),
            'high': float(kline['h']),
            'low': float(kline['l']),
            'close': float(kline['c']),
            'volume': float(kline['v']),
        }
        if self.position and not is_closed:  # 实时K线
            self.entry_bars += 1

        if is_closed:
            # K线完成，添加到历史数据
            self.klines.append(kline_data)
            self.current_kline = None

            # 处理交易逻辑
            self.on_bar_closed(kline_data)
        else:
            # K线未完成，更新当前K线
            self.current_kline = kline_data

            # 检查止盈止损（每次价格更新都检查）
            if self.position:
                self.check_exit(kline_data['close'])

    def on_bar_closed(self, kline):
        """K线完成时的处理"""
        timestamp = datetime.fromtimestamp(kline['timestamp'] / 1000)
        price = float(kline['close'])  # 确保价格是浮点数

        print(f"\n{'=' * 60}")
        print(
            f"[{timestamp}] K线完成: O:{kline['open']:.5f} H:{kline['high']:.5f} L:{kline['low']:.5f} C:{kline['close']:.5f}")

        # 更新持仓时间
        if self.position:
            self.entry_bars += 1

        # 检查是否在冷却期
        if time.time() < self.cooldown_until:
            remaining = int(self.cooldown_until - time.time())
            print(f"⏸️  冷却期中，剩余 {remaining} 秒")
            return

        # 检查是否有持仓
        if self.position:
            # 检查平仓条件
            self.check_exit(price)
        else:
            # 检查开仓条件
            self.check_entry(price)

        # 显示当前状态
        self.print_status()

    def check_entry(self, current_price):
        """检查开仓条件"""
        # 检查风控
        risk_check = self.monitor.check_risk_control()
        if risk_check['should_pause']:
            print("🛑 触发风控，暂停交易！")
            for danger in risk_check['dangers']:
                print(f"  ❌ {danger}")
            return

        # 计算特征
        df = pd.DataFrame(list(self.klines))
        features_df = add_features(df)

        if len(features_df) == 0:
            print("⚠️  特征计算失败")
            return

        # 获取最新特征
        latest_features = features_df.iloc[-1]

        # 准备模型输入 - 修复特征列选择，只排除timestamp，保留close作为特征
        feature_cols = [col for col in features_df.columns if col not in ['timestamp']]
        X = latest_features[feature_cols].values.reshape(1, -1)

        # 预测
        prob = self.model.predict(X)[0]

        print(f"📊 预测概率: {prob:.4f}")

        # 判断信号
        signal = None
        if prob > 0.5 + CONFIDENCE_THRESHOLD:
            signal = 'long'
            print(f"📈 做多信号 (prob={prob:.4f} > {0.5 + CONFIDENCE_THRESHOLD:.4f})")
        elif prob < 0.5 - CONFIDENCE_THRESHOLD:
            signal = 'short'
            print(f"📉 做空信号 (prob={prob:.4f} < {0.5 - CONFIDENCE_THRESHOLD:.4f})")
        else:
            print(
                f"⏸️  观望 (prob={prob:.4f} 在 [{0.5 - CONFIDENCE_THRESHOLD:.4f}, {0.5 + CONFIDENCE_THRESHOLD:.4f}] 内)")

        # 执行开仓
        if signal:
            self.open_position(signal, float(current_price), prob)  # 确保current_price是浮点数

    def open_position(self, direction, price, prob):
        """开仓"""
        # 计算仓位大小
        position_value = self.cash * MAX_POSITION_RATIO
        amount = position_value / price

        print(f"\n{'=' * 60}")
        print(f"🔔 开仓: {direction.upper()}")
        print(f"  价格: {price:.5f}")
        print(f"  数量: {amount:.2f}")
        print(f"  金额: ${position_value:.2f}")
        print(f"  概率: {prob:.4f}")
        print(f"{'=' * 60}")

        # 执行交易（如果不是测试模式）
        if not self.test_mode and self.exchange:
            try:
                if direction == 'long':
                    order = self.exchange.create_market_buy_order(SYMBOL, amount)
                else:
                    order = self.exchange.create_market_sell_order(SYMBOL, amount)
                print(f"✅ 订单已执行: {order['id']}")
            except Exception as e:
                print(f"❌ 订单执行失败: {e}")
                return

        # 更新状态
        self.position = direction
        self.entry_price = price
        self.entry_time = datetime.now()
        self.entry_bars = 0
        self.highest_price = price
        self.lowest_price = price

        # 扣除资金
        self.cash -= position_value

    def check_exit(self, current_price):
        """检查平仓条件"""
        if not self.position:
            return

        # 更新最高/最低价
        current_price = float(current_price)  # 确保current_price是浮点数
        if current_price > self.highest_price:
            self.highest_price = current_price
        if current_price < self.lowest_price:
            self.lowest_price = current_price

        # 计算收益率
        if self.position == 'long':
            pnl_pct = (current_price - self.entry_price) / self.entry_price
        else:  # short
            pnl_pct = (self.entry_price - current_price) / self.entry_price

        # 检查平仓条件
        exit_reason = None

        # 1. 止盈
        if pnl_pct >= TAKE_PROFIT_PCT:
            exit_reason = '止盈'

        # 2. 止损
        elif pnl_pct <= -STOP_LOSS_PCT:
            exit_reason = '止损'

        # 3. 时间止损
        elif self.entry_bars >= TIME_STOP_BARS:
            exit_reason = '时间止损'

        # 4. 信号反转（重新计算特征和预测）
        else:
            df = pd.DataFrame(list(self.klines))
            features_df = add_features(df)
            if len(features_df) > 0:
                latest_features = features_df.iloc[-1]
                # 修复特征列选择，只排除timestamp，保留close作为特征
                feature_cols = [col for col in features_df.columns if col not in ['timestamp']]
                X = latest_features[feature_cols].values.reshape(1, -1)
                prob = self.model.predict(X)[0]

                if self.position == 'long' and prob < 0.5 - CONFIDENCE_THRESHOLD:
                    exit_reason = '信号反转(做空)'
                elif self.position == 'short' and prob > 0.5 + CONFIDENCE_THRESHOLD:
                    exit_reason = '信号反转(做多)'

        # 执行平仓
        if exit_reason:
            self.close_position(current_price, exit_reason)

    def close_position(self, price, reason):
        """平仓"""
        # 计算盈亏
        position_value = self.cash / (1 - MAX_POSITION_RATIO) * MAX_POSITION_RATIO
        amount = position_value / self.entry_price

        if self.position == 'long':
            pnl = (price - self.entry_price) * amount
            pnl_pct = (price - self.entry_price) / self.entry_price
        else:  # short
            pnl = (self.entry_price - price) * amount
            pnl_pct = (self.entry_price - price) / self.entry_price

        # 扣除手续费
        fee = position_value * COMMISSION_PCT * 2  # 开仓+平仓
        pnl -= fee

        print(f"\n{'=' * 60}")
        print(f"🔔 平仓: {self.position.upper()}")
        print(f"  开仓价: {self.entry_price:.5f}")
        print(f"  平仓价: {price:.5f}")
        print(f"  收益率: {pnl_pct * 100:+.2f}%")
        print(f"  盈亏: ${pnl:+.2f}")
        print(f"  手续费: ${fee:.2f}")
        print(f"  持仓时间: {self.entry_bars}根K线")
        print(f"  平仓原因: {reason}")
        print(f"{'=' * 60}")

        # 执行交易（如果不是测试模式）
        if not self.test_mode and self.exchange:
            try:
                if self.position == 'long':
                    order = self.exchange.create_market_sell_order(SYMBOL, amount)
                else:
                    order = self.exchange.create_market_buy_order(SYMBOL, amount)
                print(f"✅ 订单已执行: {order['id']}")
            except Exception as e:
                print(f"❌ 订单执行失败: {e}")

        # 记录交易
        trade = {
            'type': self.position,
            'entry': self.entry_price,
            'exit': price,
            'pnl': pnl,
            'return': pnl_pct,
            'bars': self.entry_bars,
            'reason': reason,
        }
        self.trades.append(trade)
        self.monitor.add_trade(trade)

        # 更新资金
        self.cash += position_value + pnl
        self.equity = self.cash
        self.monitor.update_equity(self.equity, self.cash, None)

        # 重置状态
        self.position = None
        self.entry_price = 0
        self.entry_time = None
        self.entry_bars = 0

        # 设置冷却期
        self.cooldown_until = time.time() + COOLDOWN_BARS * 15 * 60  # 15分钟 * 冷却根数

        # 生成每日报告
        if len(self.trades) % 5 == 0:  # 每5笔交易生成一次报告
            self.monitor.generate_daily_report()

    def print_status(self):
        """打印当前状态"""
        print(f"\n📊 当前状态:")
        print(f"  权益: ${self.equity:.2f}")
        print(f"  现金: ${self.cash:.2f}")
        print(f"  持仓: {self.position or '无'}")

        if self.position:
            current_price = self.klines[-1]['close']
            if self.position == 'long':
                pnl_pct = (current_price - self.entry_price) / self.entry_price
            else:
                pnl_pct = (self.entry_price - current_price) / self.entry_price

            print(f"  开仓价: {self.entry_price:.5f}")
            print(f"  当前价: {current_price:.5f}")
            print(f"  浮动盈亏: {pnl_pct * 100:+.2f}%")
            print(f"  持仓时间: {self.entry_bars}根K线")

        print(f"  总交易: {len(self.trades)}笔")
        if self.trades:
            wins = sum(1 for t in self.trades if t['pnl'] > 0)
            print(f"  胜率: {wins / len(self.trades) * 100:.1f}%")

    def start(self):
        """启动交易系统"""
        print("\n🚀 启动实时交易系统...")

        # 加载历史数据
        self.load_historical_data()

        # 创建WebSocket管理器
        ws_url = f"wss://stream.binance.com:9443/ws/{self.symbol.lower()}@kline_{self.interval}"

        url = f"wss://stream.binance.com:9443/ws/{self.symbol.lower()}@kline_{self.interval}"

        def on_message(ws, message):
            self.handle_kline(message)

        def on_error(ws, error):
            print("\n❌ WebSocket 错误:", error)

        def on_close(ws, close_status_code, close_msg):
            print("\n🛑 WebSocket 已关闭")

        def on_open(ws):
            print("✅ WebSocket 已连接")

        self.twm = websocket.WebSocketApp(
            url,
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close
        )

        # 启动K线监听
        print(f"📡 启动 {self.symbol} {self.interval} K线监听...")

        def on_open(ws):
            print("✅ WebSocket已连接")
            # 输出当前交易对信息
            print(f"📊 当前交易对: {self.symbol}")
            print(f"📊 时间框架: {self.interval}")

        self.twm = websocket.WebSocketApp(
            url,
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close
        )

        print("✅ 系统已启动！")
        print("按 Ctrl+C 停止...")

        try:
            # 使用代理（可选）
            self.twm.run_forever(
                ping_interval=20,
                ping_timeout=10,
                reconnect=5,
                http_proxy_host="127.0.0.1",
                http_proxy_port=7890,
                proxy_type="http"  # socks5: "socks5"
            )
        except KeyboardInterrupt:
            print("\n\n🛑 停止交易系统...")
            self.stop()

    def stop(self):
        """停止交易系统"""
        # 停止WebSocket
        if self.twm:
            self.twm.stop()

        # 如果有持仓，平仓
        if self.position:
            current_price = self.klines[-1]['close']
            self.close_position(current_price, '手动停止')

        # 生成最终报告
        print("\n" + "=" * 60)
        print("最终报告")
        print("=" * 60)
        print(f"初始资金: ${INITIAL_CASH:.2f}")
        print(f"最终权益: ${self.equity:.2f}")
        print(f"总盈亏: ${self.equity - INITIAL_CASH:+.2f} ({(self.equity - INITIAL_CASH) / INITIAL_CASH * 100:+.2f}%)")
        print(f"总交易: {len(self.trades)}笔")

        if self.trades:
            wins = sum(1 for t in self.trades if t['pnl'] > 0)
            print(f"胜率: {wins / len(self.trades) * 100:.1f}%")

            total_win = sum(t['pnl'] for t in self.trades if t['pnl'] > 0)
            total_loss = abs(sum(t['pnl'] for t in self.trades if t['pnl'] <= 0))
            if total_loss > 0:
                print(f"盈亏比: {total_win / total_loss:.2f}:1")

        print("=" * 60)

        # 生成最终报告
        self.monitor.generate_daily_report()

        print("\n✅ 系统已停止")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='实时交易系统 - WebSocket版本')
    parser.add_argument('--api-key', type=str, help='Binance API Key')
    parser.add_argument('--api-secret', type=str, help='Binance API Secret')
    parser.add_argument('--live', action='store_true', help='实盘模式（默认为测试模式）')
    parser.add_argument('--proxy-host', type=str, default='127.0.0.1')
    parser.add_argument('--proxy-port', type=int, default=7890)
    args = parser.parse_args()

    # 从环境变量或参数获取API密钥
    api_key = args.api_key or os.getenv('BINANCE_API_KEY')
    api_secret = args.api_secret or os.getenv('BINANCE_API_SECRET')

    test_mode = not args.live

    if not test_mode and (not api_key or not api_secret):
        print("❌ 实盘模式需要提供 API 密钥！")
        print("方法1: 使用参数 --api-key 和 --api-secret")
        print("方法2: 设置环境变量 BINANCE_API_KEY 和 BINANCE_API_SECRET")
        sys.exit(1)

    # 创建交易系统
    trader = LiveTraderWebSocket(
        api_key=api_key,
        api_secret=api_secret,
        test_mode=test_mode,
        proxy_host=args.proxy_host,
        proxy_port=args.proxy_port
    )

    # 启动
    trader.start()


if __name__ == "__main__":
    main()
