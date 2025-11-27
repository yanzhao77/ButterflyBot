#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
双向交易策略
- 概率>0.5+阈值：做多
- 概率<0.5-阈值：做空
- 其他：观望
"""

import sys
import os
import pandas as pd
import numpy as np
import backtrader as bt
from datetime import datetime
import joblib
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import (
    INITIAL_CASH, SYMBOL, TIMEFRAME, BASE_PATH,
    MAX_POSITION_RATIO, TIME_STOP_BARS
)
from data.features import add_features
from model.model_registry import load_latest_model_path

# 策略参数
STOP_LOSS_PCT = 0.02  # 2%
TAKE_PROFIT_PCT = 0.03  # 3%
CONFIDENCE_THRESHOLD = 0.15  # 置信度阈值：距离0.5的距离
COOLDOWN_BARS = 3

print("=" * 80)
print("双向交易策略回测")
print("=" * 80)

class BidirectionalStrategy(bt.Strategy):
    """双向交易策略"""
    
    params = (
        ('model', None),
        ('printlog', False),
        ('window', 200),
    )
    
    def __init__(self):
        self.data_close = self.datas[0].close
        self.order = None
        self.entry_price = None
        self.entry_bar = None
        self.position_type = None  # 'long' or 'short'
        self.cooldown_until = -1
        self.trades = []
        
    def log(self, txt, dt=None):
        if self.params.printlog:
            dt = dt or self.datas[0].datetime.date(0)
            print(f'[{dt.isoformat()}] {txt}')
    
    def next(self):
        if self.order:
            return
        
        current_bar = len(self)
        if current_bar <= self.cooldown_until:
            return
        
        total_bars = len(self)
        if total_bars < 100:
            return
        
        # 准备特征数据
        window = min(500, total_bars)
        start_idx = max(0, total_bars - window)
        idx_range = range(start_idx, total_bars)
        current_idx = total_bars - 1
        agos = [i - current_idx for i in idx_range]
        
        df = pd.DataFrame({
            'timestamp': [bt.num2date(self.datas[0].datetime[ago]) for ago in agos],
            'open': [self.datas[0].open[ago] for ago in agos],
            'high': [self.datas[0].high[ago] for ago in agos],
            'low': [self.datas[0].low[ago] for ago in agos],
            'close': [self.datas[0].close[ago] for ago in agos],
            'volume': [self.datas[0].volume[ago] for ago in agos],
        })
        
        df.set_index('timestamp', inplace=True)
        
        try:
            df_feat = add_features(df)
            if df_feat.empty:
                return
        except:
            return
        
        feature_cols = ['open', 'high', 'low', 'close', 'volume', 'return', 'log_return',
                       'ma20', 'ma50', 'ma_diff', 'rsi', 'macd', 'macd_signal', 
                       'macd_hist', 'volatility', 'volume_ratio']
        
        missing = [col for col in feature_cols if col not in df_feat.columns]
        if missing:
            return
        
        try:
            X = df_feat[feature_cols].iloc[[-1]]
            prob = self.params.model.predict(X)[0]
        except:
            return
        
        price_now = self.data_close[0]
        
        # 计算置信度（距离0.5的距离）
        confidence = abs(prob - 0.5)
        
        if self.position:
            self.check_exit(price_now, prob, current_bar, confidence)
        else:
            self.check_entry(price_now, prob, current_bar, confidence)
    
    def check_entry(self, price_now, prob, current_bar, confidence):
        """检查开仓信号"""
        
        # 只在高置信度时交易
        if confidence < CONFIDENCE_THRESHOLD:
            return
        
        size = int((self.broker.getvalue() * MAX_POSITION_RATIO) / price_now)
        if size <= 0:
            return
        
        # 做多信号
        if prob > 0.5:
            self.order = self.buy(size=size)
            self.entry_price = price_now
            self.entry_bar = current_bar
            self.position_type = 'long'
            self.log(f'做多 @ {price_now:.6f}, prob={prob:.4f}, conf={confidence:.4f}')
        
        # 做空信号
        elif prob < 0.5:
            self.order = self.sell(size=size)
            self.entry_price = price_now
            self.entry_bar = current_bar
            self.position_type = 'short'
            self.log(f'做空 @ {price_now:.6f}, prob={prob:.4f}, conf={confidence:.4f}')
    
    def check_exit(self, price_now, prob, current_bar, confidence):
        """检查平仓信号"""
        
        if not self.entry_price or not self.position_type:
            return
        
        should_exit = False
        exit_reason = ""
        
        # 计算收益率
        if self.position_type == 'long':
            ret = (price_now - self.entry_price) / self.entry_price
        else:  # short
            ret = (self.entry_price - price_now) / self.entry_price
        
        # 止损
        if ret <= -STOP_LOSS_PCT:
            should_exit = True
            exit_reason = f"止损{ret*100:.2f}%"
        
        # 止盈
        elif ret >= TAKE_PROFIT_PCT:
            should_exit = True
            exit_reason = f"止盈{ret*100:.2f}%"
        
        # 时间止损
        elif (current_bar - self.entry_bar) >= TIME_STOP_BARS:
            should_exit = True
            exit_reason = f"时间{current_bar - self.entry_bar}根"
        
        # 信号反转
        elif self.position_type == 'long' and prob < 0.5 - CONFIDENCE_THRESHOLD:
            should_exit = True
            exit_reason = f"信号反转{prob:.4f}"
        elif self.position_type == 'short' and prob > 0.5 + CONFIDENCE_THRESHOLD:
            should_exit = True
            exit_reason = f"信号反转{prob:.4f}"
        
        if should_exit:
            self.order = self.close()
            self.trades.append({
                'type': self.position_type,
                'entry': self.entry_price,
                'exit': price_now,
                'return': ret,
                'reason': exit_reason
            })
            self.log(f'平仓 @ {price_now:.6f}, {ret*100:+.2f}%, {exit_reason}')
            self.entry_price = None
            self.entry_bar = None
            self.position_type = None
            self.cooldown_until = current_bar + COOLDOWN_BARS

def run_backtest():
    cache_path = BASE_PATH / 'cached_data' / f"binance_{SYMBOL.replace('/', '_')}_{TIMEFRAME}.csv"
    
    if not cache_path.exists():
        print(f"❌ 数据文件不存在")
        return None
    
    print(f"\n📂 加载数据...")
    df = pd.read_csv(cache_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df.set_index('timestamp', inplace=True)
    print(f"✅ {len(df)}条, {df.index[0].date()} 至 {df.index[-1].date()}")
    
    print(f"\n🤖 加载模型...")
    model_path = load_latest_model_path()
    if not model_path:
        print("❌ 未找到模型")
        return None
    model = joblib.load(model_path)
    print(f"✅ 模型加载成功: {model_path}")
    
    print(f"\n📊 策略参数:")
    print(f"  置信度阈值: {CONFIDENCE_THRESHOLD:.2f} (距离0.5)")
    print(f"  做多条件: prob > {0.5 + CONFIDENCE_THRESHOLD:.2f}")
    print(f"  做空条件: prob < {0.5 - CONFIDENCE_THRESHOLD:.2f}")
    print(f"  止损: {STOP_LOSS_PCT*100:.0f}%")
    print(f"  止盈: {TAKE_PROFIT_PCT*100:.0f}%")
    print(f"  盈亏比: {TAKE_PROFIT_PCT/STOP_LOSS_PCT:.1f}:1")
    
    data = bt.feeds.PandasData(dataname=df, datetime=None, open='open', high='high', 
                               low='low', close='close', volume='volume', openinterest=-1)
    
    cerebro = bt.Cerebro()
    cerebro.adddata(data)
    cerebro.addstrategy(BidirectionalStrategy, model=model, printlog=False)
    cerebro.broker.setcash(INITIAL_CASH)
    cerebro.broker.setcommission(commission=0.001)
    
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    
    start_value = cerebro.broker.getvalue()
    print(f"\n开始回测...")
    results = cerebro.run()
    strat = results[0]
    end_value = cerebro.broker.getvalue()
    
    pnl = end_value - start_value
    pnl_pct = (pnl / start_value) * 100
    
    trades = strat.analyzers.trades.get_analysis()
    drawdown = strat.analyzers.drawdown.get_analysis()
    
    print(f"\n{'='*80}")
    print("回测结果")
    print(f"{'='*80}")
    print(f"\n💰 资金: ${start_value:.2f} → ${end_value:.2f} ({pnl:+.2f}, {pnl_pct:+.2f}%)")
    
    if trades.total.total > 0:
        print(f"\n📈 交易: {trades.total.total}次")
        try:
            won = trades.won.total
            lost = trades.lost.total
            print(f"  盈利: {won}, 亏损: {lost}, 胜率: {won/(won+lost)*100:.1f}%")
            print(f"  平均盈利: ${trades.won.pnl.average:.2f}")
            print(f"  平均亏损: ${abs(trades.lost.pnl.average):.2f}")
            print(f"  盈亏比: {trades.won.pnl.average/abs(trades.lost.pnl.average):.2f}:1")
        except:
            pass
        
        days = (df.index[-1] - df.index[0]).days
        print(f"\n⏱️  {days}天, 月均{trades.total.total/days*30:.1f}次交易")
        
        # 统计做多和做空
        long_trades = [t for t in strat.trades if t['type'] == 'long']
        short_trades = [t for t in strat.trades if t['type'] == 'short']
        
        print(f"\n📊 交易类型:")
        print(f"  做多: {len(long_trades)}次")
        print(f"  做空: {len(short_trades)}次")
        
        if long_trades:
            long_returns = [t['return'] for t in long_trades]
            print(f"  做多平均收益: {np.mean(long_returns)*100:+.2f}%")
        
        if short_trades:
            short_returns = [t['return'] for t in short_trades]
            print(f"  做空平均收益: {np.mean(short_returns)*100:+.2f}%")
    
    print(f"\n📉 最大回撤: {drawdown.max.drawdown:.2f}%")
    
    if len(strat.trades) > 0:
        print(f"\n📋 交易明细(前20笔):")
        for i, t in enumerate(strat.trades[:20], 1):
            direction = "多" if t['type'] == 'long' else "空"
            print(f"  {i}. {direction} {t['entry']:.6f}→{t['exit']:.6f} {t['return']*100:+.2f}% {t['reason']}")
    
    print(f"\n{'='*80}")
    return {'pnl_pct': pnl_pct, 'trades': trades.total.total if trades.total.total > 0 else 0}

if __name__ == "__main__":
    result = run_backtest()
    sys.exit(0 if result and result['pnl_pct'] > 0 else 1)
