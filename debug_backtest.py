#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试版本 - 分析持仓状态和平仓逻辑
"""

import sys
import os
import pandas as pd
import numpy as np
import backtrader as bt
from datetime import datetime
import joblib
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import INITIAL_CASH, SYMBOL, TIMEFRAME, BASE_PATH, MAX_POSITION_RATIO
from data.features import add_features

STOP_LOSS_PCT = 0.02
TAKE_PROFIT_PCT = 0.03
TIME_STOP_BARS = 20
CONFIDENCE_THRESHOLD = 0.05
COOLDOWN_BARS = 3

class DebugStrategy(bt.Strategy):
    """调试策略 - 详细日志"""
    
    params = (('model', None),)
    
    def __init__(self):
        self.data_close = self.datas[0].close
        self.order = None
        self.entry_price = None
        self.entry_bar = None
        self.position_type = None
        self.cooldown_until = -1
        self.trades = []
        self.check_count = 0
        
    def log(self, txt):
        dt = self.datas[0].datetime.date(0)
        print(f'[{dt}] {txt}')
    
    def notify_order(self, order):
        """订单状态通知"""
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'✅ 买入订单执行: {order.executed.price:.6f}, size={order.executed.size}')
            elif order.issell():
                self.log(f'✅ 卖出订单执行: {order.executed.price:.6f}, size={order.executed.size}')
            self.order = None
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f'❌ 订单失败: {order.status}')
            self.order = None
    
    def next(self):
        current_bar = len(self)
        
        # 调试：检查持仓状态
        if current_bar % 100 == 0:
            print(f"\n[Bar {current_bar}] 持仓检查:")
            print(f"  self.position: {self.position}")
            print(f"  self.position.size: {self.position.size if self.position else 'N/A'}")
            print(f"  entry_price: {self.entry_price}")
            print(f"  position_type: {self.position_type}")
        
        if self.order:
            return
        
        if current_bar <= self.cooldown_until:
            return
        
        if current_bar < 100:
            return
        
        # 准备特征
        window = min(500, current_bar)
        start_idx = max(0, current_bar - window)
        agos = [i - (current_bar - 1) for i in range(start_idx, current_bar)]
        
        df = pd.DataFrame({
            'timestamp': [bt.num2date(self.datas[0].datetime[ago]) for ago in agos],
            'open': [self.datas[0].open[ago] for ago in agos],
            'high': [self.datas[0].high[ago] for ago in agos],
            'low': [self.datas[0].low[ago] for ago in agos],
            'close': [self.datas[0].close[ago] for ago in agos],
            'volume': [self.datas[0].volume[ago] for ago in agos],
        }).set_index('timestamp')
        
        try:
            df_feat = add_features(df)
            if df_feat.empty:
                return
            
            feature_cols = ['open', 'high', 'low', 'close', 'volume', 'return', 'log_return',
                           'ma20', 'ma50', 'ma_diff', 'rsi', 'macd', 'macd_signal', 
                           'macd_hist', 'volatility', 'volume_ratio']
            
            X = df_feat[feature_cols].iloc[[-1]]
            prob = self.params.model.predict(X)[0]
        except:
            return
        
        price_now = self.data_close[0]
        confidence = abs(prob - 0.5)
        
        # 检查是否有持仓
        has_position = self.position.size != 0
        
        # 调试：打印持仓判断
        if current_bar in [89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 105, 110]:
            print(f"\n[Bar {current_bar}] 持仓判断:")
            print(f"  self.position: {self.position}")
            print(f"  self.position.size: {self.position.size}")
            print(f"  has_position: {has_position}")
            print(f"  entry_price: {self.entry_price}")
            print(f"  position_type: {self.position_type}")
        
        if has_position:
            self.check_count += 1
            if self.check_count <= 5:  # 只打印前5次
                self.log(f"检查平仓 #{self.check_count}: price={price_now:.6f}, prob={prob:.4f}")
            self.check_exit(price_now, prob, current_bar, confidence)
        else:
            self.check_entry(price_now, prob, current_bar, confidence)
    
    def check_entry(self, price_now, prob, current_bar, confidence):
        if confidence < CONFIDENCE_THRESHOLD:
            return
        
        size = int((self.broker.getvalue() * MAX_POSITION_RATIO) / price_now)
        if size <= 0:
            return
        
        if prob > 0.5:
            self.order = self.buy(size=size)
            self.entry_price = price_now
            self.entry_bar = current_bar
            self.position_type = 'long'
            self.log(f'🟢 做多开仓 @ {price_now:.6f}, prob={prob:.4f}, size={size}')
        elif prob < 0.5:
            self.order = self.sell(size=size)
            self.entry_price = price_now
            self.entry_bar = current_bar
            self.position_type = 'short'
            self.check_count = 0
            self.log(f'🔴 做空开仓 @ {price_now:.6f}, prob={prob:.4f}, size={size}')
    
    def check_exit(self, price_now, prob, current_bar, confidence):
        if not self.entry_price or not self.position_type:
            self.log(f"⚠️  check_exit但没有entry信息")
            return
        
        # 计算收益率
        if self.position_type == 'long':
            ret = (price_now - self.entry_price) / self.entry_price
        else:
            ret = (self.entry_price - price_now) / self.entry_price
        
        bars_held = current_bar - self.entry_bar
        
        if self.check_count <= 5:
            self.log(f"  收益{ret*100:+.2f}%, 持仓{bars_held}根")
        
        should_exit = False
        exit_reason = ""
        
        # 止损
        if ret <= -STOP_LOSS_PCT:
            should_exit = True
            exit_reason = f"止损 {ret*100:.2f}%"
        # 止盈
        elif ret >= TAKE_PROFIT_PCT:
            should_exit = True
            exit_reason = f"止盈 {ret*100:.2f}%"
        # 时间止损
        elif bars_held >= TIME_STOP_BARS:
            should_exit = True
            exit_reason = f"时间止损 {bars_held}根"
        # 信号反转
        elif self.position_type == 'long' and prob < 0.5 and confidence >= CONFIDENCE_THRESHOLD:
            should_exit = True
            exit_reason = f"信号反转 prob={prob:.4f}"
        elif self.position_type == 'short' and prob > 0.5 and confidence >= CONFIDENCE_THRESHOLD:
            should_exit = True
            exit_reason = f"信号反转 prob={prob:.4f}"
        
        if should_exit:
            self.order = self.close()
            self.trades.append({
                'type': self.position_type,
                'entry': self.entry_price,
                'exit': price_now,
                'return': ret,
                'reason': exit_reason,
                'bars': bars_held
            })
            self.log(f'⭕ 平仓 @ {price_now:.6f}, {ret*100:+.2f}%, {exit_reason}')
            self.entry_price = None
            self.entry_bar = None
            self.position_type = None
            self.cooldown_until = current_bar + COOLDOWN_BARS
            self.check_count = 0

# 运行回测
cache_path = BASE_PATH / 'cached_data' / f"binance_{SYMBOL.replace('/', '_')}_{TIMEFRAME}.csv"
df = pd.read_csv(cache_path)
df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
df.set_index('timestamp', inplace=True)

model_dir = BASE_PATH / 'models' / 'registry'
with open(model_dir / 'latest_balanced.txt', 'r') as f:
    version = f.read().strip()
model = joblib.load(model_dir / f'{version}_balanced.pkl')

print("="*80)
print("调试回测 - 分析持仓和平仓逻辑")
print("="*80)
print(f"\n数据: {len(df)}条")
print(f"模型: {version}")
print(f"置信度: {CONFIDENCE_THRESHOLD}")
print(f"止损: {STOP_LOSS_PCT*100}%")
print(f"止盈: {TAKE_PROFIT_PCT*100}%")
print(f"时间止损: {TIME_STOP_BARS}根\n")

data = bt.feeds.PandasData(dataname=df, datetime=None, open='open', high='high', 
                           low='low', close='close', volume='volume', openinterest=-1)

cerebro = bt.Cerebro()
cerebro.adddata(data)
cerebro.addstrategy(DebugStrategy, model=model)
cerebro.broker.setcash(INITIAL_CASH)
cerebro.broker.setcommission(commission=0.001)

start_value = cerebro.broker.getvalue()
print(f"开始回测...\n")
results = cerebro.run()
strat = results[0]
end_value = cerebro.broker.getvalue()

pnl = end_value - start_value
pnl_pct = (pnl / start_value) * 100

print(f"\n{'='*80}")
print(f"最终结果:")
print(f"  资金: ${start_value:.2f} → ${end_value:.2f} ({pnl_pct:+.2f}%)")
print(f"  交易: {len(strat.trades)}次")

if strat.trades:
    print(f"\n交易明细:")
    for i, t in enumerate(strat.trades, 1):
        direction = "做多" if t['type'] == 'long' else "做空"
        print(f"  {i}. {direction} {t['entry']:.6f}→{t['exit']:.6f} {t['return']*100:+.2f}% {t['bars']}根 {t['reason']}")

print(f"{'='*80}")
