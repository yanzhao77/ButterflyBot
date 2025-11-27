#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化后的纯做空策略
- 做空阈值：bottom 40%
- 止盈：3%
- 止损：2%
- 盈亏比：1.5:1
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
    MAX_POSITION_RATIO, TIME_STOP_BARS, COOLDOWN_BARS,
    USE_TRAILING_STOP, TRAILING_STOP_ACTIVATION, TRAILING_STOP_DISTANCE
)
from data.features import add_features
from model.model_registry import load_latest_model_path

# 优化后的参数
STOP_LOSS_PCT = 0.02  # 2%
TAKE_PROFIT_PCT = 0.03  # 3%
SHORT_QUANTILE = 0.40  # bottom 40%
EXIT_QUANTILE = 0.65  # top 35%

print("=" * 80)
print("ButterflyBot 优化策略回测")
print("=" * 80)

class OptimizedStrategy(bt.Strategy):
    """优化后的纯做空策略"""
    
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
        self.cooldown_until = -1
        self.lowest_price = None
        self.trailing_active = False
        self.prob_history = deque(maxlen=self.params.window)
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
            self.prob_history.append(prob)
        except:
            return
        
        if len(self.prob_history) < 50:
            return
        
        probs = np.array(self.prob_history)
        short_thresh = np.percentile(probs, SHORT_QUANTILE * 100)
        exit_thresh = np.percentile(probs, EXIT_QUANTILE * 100)
        
        price_now = self.data_close[0]
        
        if self.position:
            self.check_exit(price_now, prob, current_bar, exit_thresh)
        else:
            self.check_short_entry(price_now, prob, current_bar, short_thresh)
    
    def check_short_entry(self, price_now, prob, current_bar, short_thresh):
        if prob <= short_thresh:
            size = int((self.broker.getvalue() * MAX_POSITION_RATIO) / price_now)
            if size > 0:
                self.order = self.sell(size=size)
                self.entry_price = price_now
                self.entry_bar = current_bar
                self.lowest_price = price_now
                self.trailing_active = False
                self.log(f'做空 @ {price_now:.6f}, prob={prob:.4f} (≤{short_thresh:.4f})')
    
    def check_exit(self, price_now, prob, current_bar, exit_thresh):
        if not self.entry_price:
            return
        
        should_exit = False
        exit_reason = ""
        ret = (self.entry_price - price_now) / self.entry_price
        
        if ret <= -STOP_LOSS_PCT:
            should_exit = True
            exit_reason = f"止损{ret*100:.2f}%"
        elif ret >= TAKE_PROFIT_PCT:
            should_exit = True
            exit_reason = f"止盈{ret*100:.2f}%"
        elif USE_TRAILING_STOP and ret >= TRAILING_STOP_ACTIVATION:
            self.trailing_active = True
            if self.lowest_price is None or price_now < self.lowest_price:
                self.lowest_price = price_now
            if self.lowest_price is not None:
                drawup = (price_now - self.lowest_price) / self.lowest_price
                if drawup >= TRAILING_STOP_DISTANCE:
                    should_exit = True
                    exit_reason = f"跟踪止盈{drawup*100:.2f}%"
        elif (current_bar - self.entry_bar) >= TIME_STOP_BARS:
            should_exit = True
            exit_reason = f"时间止损{current_bar - self.entry_bar}根"
        elif prob >= exit_thresh:
            should_exit = True
            exit_reason = f"概率反转{prob:.4f}"
        
        if should_exit:
            self.order = self.close()
            self.trades.append({'entry': self.entry_price, 'exit': price_now, 'return': ret, 'reason': exit_reason})
            self.log(f'平仓 @ {price_now:.6f}, {ret*100:+.2f}%, {exit_reason}')
            self.entry_price = None
            self.entry_bar = None
            self.lowest_price = None
            self.trailing_active = False
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
    print(f"✅ 模型加载成功")
    
    print(f"\n📊 策略参数:")
    print(f"  做空阈值: bottom {SHORT_QUANTILE*100:.0f}%")
    print(f"  平仓阈值: top {(1-EXIT_QUANTILE)*100:.0f}%")
    print(f"  止损: {STOP_LOSS_PCT*100:.0f}%")
    print(f"  止盈: {TAKE_PROFIT_PCT*100:.0f}%")
    print(f"  盈亏比: {TAKE_PROFIT_PCT/STOP_LOSS_PCT:.1f}:1")
    
    data = bt.feeds.PandasData(dataname=df, datetime=None, open='open', high='high', 
                               low='low', close='close', volume='volume', openinterest=-1)
    
    cerebro = bt.Cerebro()
    cerebro.adddata(data)
    cerebro.addstrategy(OptimizedStrategy, model=model, printlog=False)
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
    
    print(f"\n📉 最大回撤: {drawdown.max.drawdown:.2f}%")
    
    if len(strat.trades) > 0:
        print(f"\n📋 交易明细(前20笔):")
        for i, t in enumerate(strat.trades[:20], 1):
            print(f"  {i}. {t['entry']:.6f}→{t['exit']:.6f} {t['return']*100:+.2f}% {t['reason']}")
    
    print(f"\n{'='*80}")
    return {'pnl_pct': pnl_pct, 'trades': trades.total.total if trades.total.total > 0 else 0}

if __name__ == "__main__":
    result = run_backtest()
    sys.exit(0 if result and result['pnl_pct'] > 0 else 1)
