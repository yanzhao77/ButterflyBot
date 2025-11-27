#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速回测版本 - 预先计算所有特征和预测
"""

import sys
import os
import pandas as pd
import numpy as np
import backtrader as bt
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

class FastStrategy(bt.Strategy):
    """快速策略 - 使用预计算的预测"""
    
    params = (('predictions', None),)
    
    def __init__(self):
        self.data_close = self.datas[0].close
        self.order = None
        self.entry_price = None
        self.entry_bar = None
        self.position_type = None
        self.cooldown_until = -1
        self.trades = []
        
    def notify_order(self, order):
        """订单状态通知"""
        if order.status in [order.Completed]:
            self.order = None
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.order = None
    
    def next(self):
        if self.order:
            return
        
        current_bar = len(self) - 1
        
        if current_bar <= self.cooldown_until:
            return
        
        if current_bar >= len(self.params.predictions):
            return
        
        prob = self.params.predictions[current_bar]
        price_now = self.data_close[0]
        confidence = abs(prob - 0.5)
        
        has_position = self.position.size != 0
        
        if has_position:
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
        elif prob < 0.5:
            self.order = self.sell(size=size)
            self.entry_price = price_now
            self.entry_bar = current_bar
            self.position_type = 'short'
    
    def check_exit(self, price_now, prob, current_bar, confidence):
        if not self.entry_price or not self.position_type:
            return
        
        if self.position_type == 'long':
            ret = (price_now - self.entry_price) / self.entry_price
        else:
            ret = (self.entry_price - price_now) / self.entry_price
        
        bars_held = current_bar - self.entry_bar
        
        should_exit = False
        exit_reason = ""
        
        if ret <= -STOP_LOSS_PCT:
            should_exit = True
            exit_reason = f"止损"
        elif ret >= TAKE_PROFIT_PCT:
            should_exit = True
            exit_reason = f"止盈"
        elif bars_held >= TIME_STOP_BARS:
            should_exit = True
            exit_reason = f"时间止损"
        elif self.position_type == 'long' and prob < 0.5 and confidence >= CONFIDENCE_THRESHOLD:
            should_exit = True
            exit_reason = f"信号反转"
        elif self.position_type == 'short' and prob > 0.5 and confidence >= CONFIDENCE_THRESHOLD:
            should_exit = True
            exit_reason = f"信号反转"
        
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
            self.entry_price = None
            self.entry_bar = None
            self.position_type = None
            self.cooldown_until = current_bar + COOLDOWN_BARS

def run_backtest():
    cache_path = BASE_PATH / 'cached_data' / f"binance_{SYMBOL.replace('/', '_')}_{TIMEFRAME}.csv"
    
    print("="*80)
    print("快速回测 - 平衡模型双向交易")
    print("="*80)
    
    print(f"\n📂 加载数据...")
    df = pd.read_csv(cache_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df.set_index('timestamp', inplace=True)
    print(f"✅ {len(df)}条")
    
    print(f"\n🔧 计算特征...")
    df_feat = add_features(df)
    
    feature_cols = ['open', 'high', 'low', 'close', 'volume', 'return', 'log_return',
                   'ma20', 'ma50', 'ma_diff', 'rsi', 'macd', 'macd_signal', 
                   'macd_hist', 'volatility', 'volume_ratio']
    
    df_feat = df_feat.dropna(subset=feature_cols)
    print(f"✅ {len(df_feat)}条有效数据")
    
    print(f"\n🤖 加载模型并预测...")
    model_dir = BASE_PATH / 'models' / 'registry'
    with open(model_dir / 'latest_balanced.txt', 'r') as f:
        version = f.read().strip()
    model = joblib.load(model_dir / f'{version}_balanced.pkl')
    
    X = df_feat[feature_cols].values
    predictions = model.predict(X)
    print(f"✅ 预测完成: {len(predictions)}条")
    
    print(f"\n📊 策略参数:")
    print(f"  置信度: {CONFIDENCE_THRESHOLD}")
    print(f"  止损: {STOP_LOSS_PCT*100}%")
    print(f"  止盈: {TAKE_PROFIT_PCT*100}%")
    print(f"  时间止损: {TIME_STOP_BARS}根")
    
    print(f"\n⏳ 开始回测...")
    
    data = bt.feeds.PandasData(dataname=df_feat, datetime=None, open='open', high='high', 
                               low='low', close='close', volume='volume', openinterest=-1)
    
    cerebro = bt.Cerebro()
    cerebro.adddata(data)
    cerebro.addstrategy(FastStrategy, predictions=predictions)
    cerebro.broker.setcash(INITIAL_CASH)
    cerebro.broker.setcommission(commission=0.001)
    
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    
    start_value = cerebro.broker.getvalue()
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
    print(f"\n💰 资金变化:")
    print(f"  初始: ${start_value:.2f}")
    print(f"  最终: ${end_value:.2f}")
    print(f"  盈亏: ${pnl:+.2f} ({pnl_pct:+.2f}%)")
    
    if trades.total.total > 0:
        print(f"\n📈 交易统计:")
        print(f"  总交易: {trades.total.total}次")
        
        try:
            won = trades.won.total
            lost = trades.lost.total
            win_rate = won/(won+lost)*100
            print(f"  盈利: {won}次")
            print(f"  亏损: {lost}次")
            print(f"  胜率: {win_rate:.1f}%")
            
            avg_win = trades.won.pnl.average
            avg_loss = abs(trades.lost.pnl.average)
            profit_factor = avg_win / avg_loss
            
            print(f"\n  平均盈利: ${avg_win:.2f}")
            print(f"  平均亏损: ${avg_loss:.2f}")
            print(f"  盈亏比: {profit_factor:.2f}:1")
        except:
            pass
        
        days = (df_feat.index[-1] - df_feat.index[0]).days
        monthly_trades = trades.total.total / days * 30
        print(f"\n⏱️  回测周期: {days}天")
        print(f"  月均交易: {monthly_trades:.1f}次")
        
        long_trades = [t for t in strat.trades if t['type'] == 'long']
        short_trades = [t for t in strat.trades if t['type'] == 'short']
        
        print(f"\n📊 交易类型:")
        print(f"  做多: {len(long_trades)}次")
        print(f"  做空: {len(short_trades)}次")
        
        if long_trades:
            long_returns = [t['return'] for t in long_trades]
            long_wins = sum(1 for r in long_returns if r > 0)
            print(f"\n  做多详情:")
            print(f"    胜率: {long_wins/len(long_trades)*100:.1f}%")
            print(f"    平均收益: {np.mean(long_returns)*100:+.2f}%")
        
        if short_trades:
            short_returns = [t['return'] for t in short_trades]
            short_wins = sum(1 for r in short_returns if r > 0)
            print(f"\n  做空详情:")
            print(f"    胜率: {short_wins/len(short_trades)*100:.1f}%")
            print(f"    平均收益: {np.mean(short_returns)*100:+.2f}%")
    
    print(f"\n📉 风险指标:")
    print(f"  最大回撤: {drawdown.max.drawdown:.2f}%")
    
    if len(strat.trades) > 0:
        print(f"\n📋 交易明细 (前20笔):")
        for i, t in enumerate(strat.trades[:20], 1):
            direction = "做多" if t['type'] == 'long' else "做空"
            print(f"  {i:2d}. {direction} {t['return']*100:+.2f}% {t['bars']:3d}根 {t['reason']}")
    
    print(f"\n{'='*80}")
    
    return {
        'pnl_pct': pnl_pct,
        'trades': trades.total.total if trades.total.total > 0 else 0,
        'long_trades': len(long_trades) if 'long_trades' in locals() else 0,
        'short_trades': len(short_trades) if 'short_trades' in locals() else 0,
    }

if __name__ == "__main__":
    result = run_backtest()
