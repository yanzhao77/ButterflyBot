# backtest/run_backtest.py
"""
AI 量化策略回测主程序
"""

import backtrader as bt
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime

from data.fetcher import fetch_ohlcv
from data.features import add_features
from model.ensemble_model import EnsembleModel
from model.model_registry import load_latest_model_path
from config.settings import TIMEFRAME, INITIAL_CASH
from backtest.metrics import calculate_metrics


class AIButterflyStrategy(bt.Strategy):
    params = (
        ("model", None),  # 传入训练好的 EnsembleModel 实例
        ("printlog", False),
    )

    def __init__(self):
        self.data_close = self.datas[0].close
        self.order = None
        self.trade_list = []  # 记录每笔交易

    def next(self):
        if self.order:
            return  # 有未完成订单，跳过

        # 获取当前及历史数据（DataFrame 格式）
        df = pd.DataFrame({
            'timestamp': [bt.num2date(self.datas[0].datetime[i]) for i in range(len(self))],
            'open': [self.datas[0].open[i] for i in range(len(self))],
            'high': [self.datas[0].high[i] for i in range(len(self))],
            'low': [self.datas[0].low[i] for i in range(len(self))],
            'close': [self.datas[0].close[i] for i in range(len(self))],
            'volume': [self.datas[0].volume[i] for i in range(len(self))],
        })
        df.set_index('timestamp', inplace=True)
        df = add_features(df)

        # 预测上涨概率
        prob = self.params.model.predict(df)

        # 交易逻辑
        if not self.position:
            if prob > 0.6:  # 强买入信号
                size = self.broker.getcash() / self.data_close[0]
                self.order = self.buy(size=size)
                if self.p.printlog:
                    self.log(f"BUY CREATE, price={self.data_close[0]:.2f}, prob={prob:.3f}")
        else:
            if prob < 0.4:  # 强卖出信号
                self.order = self.sell(size=self.position.size)
                if self.p.printlog:
                    self.log(f"SELL CREATE, price={self.data_close[0]:.2f}, prob={prob:.3f}")

    def notify_order(self, order):
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f"BUY EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}")
            elif order.issell():
                self.log(f"SELL EXECUTED, Price: {order.executed.price:.2f}, Value: {order.executed.value:.2f}")
            self.bar_executed = len(self)
        self.order = None

    def notify_trade(self, trade):
        if trade.isclosed:
            self.trade_list.append({
                "pnl": trade.pnlcomm,
                "pnl_pct": trade.pnlcomm / (trade.value - trade.pnlcomm),
                "size": trade.size,
                "value": trade.value,
                "entry": trade.price,
                "exit": trade.price + (trade.pnl / trade.size if trade.size != 0 else 0),
                "duration": trade.barlen
            })
            self.log(f"OPERATION PROFIT, GROSS {trade.pnl:.2f}, NET {trade.pnlcomm:.2f}")

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print(f"{dt.isoformat()} {txt}")


def run_backtest():
    print("🔄 开始回测...")

    # 1. 获取数据
    df = fetch_ohlcv(limit=2000)  # 获取足够历史数据
    if len(df) < 300:
        raise ValueError("回测数据不足，请确保至少有 300 根K线")

    # 2. 加载最新模型
    model_path = load_latest_model_path()
    if not model_path:
        raise RuntimeError("未找到训练好的模型，请先运行 model/train.py")
    ensemble_model = EnsembleModel(model_path, TIMEFRAME)
    print(f"✅ 已加载模型: {os.path.basename(model_path)}")

    # 3. 初始化 Cerebro 引擎
    cerebro = bt.Cerebro()
    cerebro.addstrategy(AIButterflyStrategy, model=ensemble_model, printlog=False)

    # 转换为 Backtrader 数据格式
    data = bt.feeds.PandasData(
        dataname=df,
        datetime=None,
        open=0,
        high=1,
        low=2,
        close=3,
        volume=4,
        openinterest=-1
    )
    cerebro.adddata(data)
    cerebro.broker.setcash(INITIAL_CASH)
    cerebro.broker.setcommission(commission=0.001)  # 0.1% 手续费

    # 4. 运行回测
    start_value = cerebro.broker.getvalue()
    results = cerebro.run()
    end_value = cerebro.broker.getvalue()

    # 5. 计算指标
    strategy = results[0]
    trades = strategy.trade_list
    metrics = calculate_metrics(trades, df["target"].iloc[-len(trades):] if len(trades) > 0 else [])

    # 补充资金曲线指标
    metrics.update({
        "initial_cash": INITIAL_CASH,
        "final_value": round(end_value, 2),
        "total_return_pct": round((end_value - start_value) / start_value * 100, 2),
        "total_trades": len(trades)
    })

    # 6. 保存指标
    metrics_path = "backtest/strategy_metrics.json"
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)

    # 7. 打印摘要
    print("\n" + "="*50)
    print(f"💼 初始资金: {INITIAL_CASH:,.2f} USDT")
    print(f"💰 最终资金: {end_value:,.2f} USDT")
    print(f"📈 收益率: {metrics['total_return_pct']:.2f}%")
    print(f"📊 回测完成 | AUC: {metrics.get('auc', 'N/A')}, 胜率: {metrics.get('win_rate', 0)*100:.1f}%")
    print("="*50)

    return metrics


if __name__ == "__main__":
    try:
        metrics = run_backtest()
    except Exception as e:
        print(f"❌ 回测失败: {e}")
        raise