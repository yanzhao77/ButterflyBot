# strategies/backtrader_adapters/ai_signal_bt.py
"""
Backtrader 适配器：包装 AISignalCore，用于回测
"""

import backtrader as bt
import pandas as pd

from strategies.ai_signal_core import AISignalCore
from backtest.metrics import calculate_metrics, save_metrics
from config.settings import TIMEFRAME


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
        ('cooldown_bars', 0),      # Backtrader 自带 bar 控制，通常设为 0
        ('trend_filter', False),   # 可选开启
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
        self.trade_list = []
        self.y_true_list = []
        self.y_pred_list = []

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

        # 记录真实标签（用于 AUC）
        if len(self.data.close) > 1:
            true_label = 1 if self.data.close[1] > self.data.close[0] else 0
            self.y_true_list.append(true_label)

        # 获取信号
        signal_info = self.signal_engine.generate_signal(df)
        prob = signal_info["confidence"]
        self.y_pred_list.append(prob)

        # 执行交易
        if not self.position:
            if signal_info["signal"] == "buy":
                self.buy()
                self.entry_price = self.data.close[0]
        else:
            if signal_info["signal"] == "sell":
                self.sell()

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

    def stop(self):
        if self.params.save_trades:
            try:
                metrics = calculate_metrics(
                    trade_list=self.trade_list,
                    y_true=self.y_true_list,
                    y_pred=self.y_pred_list
                )
                save_metrics(metrics)
                print(f"✅ 回测完成 | AUC: {metrics['auc']:.4f}, 胜率: {metrics['win_rate']:.1%}")
            except Exception as e:
                print(f"⚠️ 指标计算失败: {e}")