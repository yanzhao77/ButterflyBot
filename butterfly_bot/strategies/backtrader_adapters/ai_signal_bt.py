# strategies/backtrader_adapters/ai_signal_bt.py
"""
Backtrader 适配器：包装 AISignalCore，用于回测
"""

import backtrader as bt
import pandas as pd
import numpy as np

from backtest.metrics import calculate_metrics, save_metrics
from config.settings import TIMEFRAME, MAX_POSITION_RATIO, STOP_LOSS_PCT, TAKE_PROFIT_PCT
from strategies.ai_signal_core import AISignalCore


class AISignalStrategy(bt.Strategy):
    """
    Backtrader 策略类，仅负责：
    - 调用 AISignalCore 生成信号
    - 执行 buy/sell
    - 记录交易与标签（用于 AUC 计算）
    """
    params = (
        ('timeframe', TIMEFRAME),
        ('save_trades', True),
        ('confidence_threshold', 0.6),
        ('cooldown_bars', 0),  # Backtrader 自带 bar 控制，通常设为 0
        ('trend_filter', False),  # 可选开启
    )

    def __init__(self):
        # 初始化核心信号引擎
        self.signal_engine = AISignalCore(
            timeframe=self.params.timeframe,
            confidence_threshold=self.params.confidence_threshold,
            cooldown_bars=self.params.cooldown_bars,
            trend_filter=self.params.trend_filter
        )

        # 状态变量
        self.order = None
        self.entry_price = None
        self.entry_bar = None
        self.atr = bt.indicators.ATR(self.data, period=14)
        self.trade_list = []
        self.y_true_list = []
        self.y_pred_list = []
        self.trade_list_bt = []  # Backtrader风格的完整交易记录

    def next(self):
        if self.order or len(self.data) < 100:
            return

        # 构建 DataFrame（与训练时一致）
        size = len(self.data)
        df = pd.DataFrame({
            'open': self.data.open.get(size=size),
            'high': self.data.high.get(size=size),
            'low': self.data.low.get(size=size),
            'close': self.data.close.get(size=size),
            'volume': self.data.volume.get(size=size),
        })
        df.index = pd.to_datetime(self.data.datetime.array, unit='s')[-size:]

        # 记录真实标签（用于 AUC）：用"当前 vs 前一根"避免未来数据泄露与越界
        try:
            if len(self.data.close) > 1:
                true_label = 1 if float(self.data.close[0]) > float(self.data.close[-1]) else 0
                self.y_true_list.append(true_label)
        except Exception:
            pass

        # 获取信号
        signal_info = self.signal_engine.generate_signal(df)
        prob = signal_info["confidence"]
        self.y_pred_list.append(prob)
        
        # ========== 简化趋势判断 (vs 严格过滤) ==========
        # 主要依赖 AI 信号，趋势仅作为辅助判断
        # 计算短期和长期均线（但不作为硬条件）
        close_prices = self.data.close.get(size=min(50, size))
        if len(close_prices) >= 20:
            ma5 = np.mean(close_prices[-5:])
            ma20 = np.mean(close_prices[-20:])
            current_price = float(self.data.close[0])
            
            # 辅助趋势：优化会值但不是必需
            trend_up = (ma5 > ma20)  # 短期 > 长期 = 上升趋势
            trend_down = (ma5 < ma20)  # 短期 < 长期 = 下降趋势
        else:
            trend_up = False
            trend_down = False

        # 执行交易
        if not self.position:
            # ========== 优化的买入逻辑 ==========
            # 主要依赖AI信号，趋势是加强因素(不是离场条件)
            if signal_info["signal"] == "buy":
                # 如果趋势向上，更有信心（但趋势向下也可以买）
                confidence_boost = 1.0 if trend_up else 0.8
                
                # 仓位 sizing：按资金占用上限
                cash = float(self.broker.getcash())
                price = float(self.data.close[0])
                comm = getattr(self.broker.getcommissioninfo(self.data).p, 'commission', 0.001)
                safety = 0.98
                budget = cash * float(MAX_POSITION_RATIO) * confidence_boost  # 趋势好时加大仓位
                unit_cost = price * (1.0 + float(comm))
                size = int((budget * safety) / unit_cost)
                if size > 0:
                    self.buy(size=size)
                    self.entry_price = price
                    self.entry_bar = len(self)
        else:
            # ========== 优化的平仓逻辑 ==========
            price = float(self.data.close[0])
            atr = float(self.atr[0]) if not np.isnan(float(self.atr[0])) else 0.0
            hit_sl = False
            hit_tp = False
            hit_time = False
            
            if self.entry_price:
                ret = (price - self.entry_price) / self.entry_price
                hit_sl = ret <= -float(STOP_LOSS_PCT)
                hit_tp = ret >= float(TAKE_PROFIT_PCT)
                # ATR 止损/止盈（与百分比二选一命中即可）
                if atr > 0:
                    hit_sl = hit_sl or (price <= self.entry_price - 1.5 * atr)
                    hit_tp = hit_tp or (price >= self.entry_price + 3.0 * atr)
            
            # 时间止损：持仓时间过长则平仓
            if self.entry_bar and len(self) - self.entry_bar >= 20:  # 20根K线（300分钟）
                hit_time = True
            
            # 平仓条件：只不依赖趋势，仅依AI信号和风控
            should_sell = (signal_info["signal"] == "sell") or hit_sl or hit_tp or hit_time
            if should_sell:
                self.sell(size=self.position.size)

    def notify_order(self, order):
        if order.status in [order.Completed]:
            if order.isbuy():
                self.entry_price = order.executed.price
            elif order.issell():
                exit_price = order.executed.price
                pnl_pct = (exit_price - self.entry_price) / self.entry_price
                self.trade_list.append({
                    "entry": self.entry_price,
                    "exit": exit_price,
                    "pnl_pct": pnl_pct,
                    "datetime": self.datetime.datetime(0)
                })

    def notify_trade(self, trade):
        if trade.isclosed:
            # 统一提供 backtest/metrics 所需字段
            self.trade_list_bt.append({
                "pnl": trade.pnlcomm,
                "pnl_pct": trade.pnlcomm / (trade.value - trade.pnlcomm) if (trade.value - trade.pnlcomm) != 0 else 0.0,
                "size": trade.size,
                "value": trade.value,
                "entry": trade.price,
                "exit": trade.price + (trade.pnl / trade.size if trade.size != 0 else 0),
                "duration": trade.barlen,
            })

    def stop(self):
        if self.params.save_trades:
            try:
                trades = self.trade_list_bt if hasattr(self, 'trade_list_bt') else []
                metrics = calculate_metrics(trades, y_true_for_auc=None)
                save_metrics(metrics)
                print(f"✅ 回测完成 | AUC: {metrics['auc']:.4f}, 胜率: {metrics['win_rate']:.1%}")
            except Exception as e:
                print(f"⚠️ 指标计算失败: {e}")
