#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于平衡模型的双向交易策略
- 使用重新训练的平衡模型
- 根据预测概率自动选择做多或做空
- 置信度阈值控制交易频率
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

from config.settings import (
    INITIAL_CASH, SYMBOL, TIMEFRAME, BASE_PATH,
    MAX_POSITION_RATIO
)

TIME_STOP_BARS = 20  # 降低时间止损至20根K线
from data.features import add_features

# 策略参数
STOP_LOSS_PCT = 0.02  # 2%
TAKE_PROFIT_PCT = 0.03  # 3%
CONFIDENCE_THRESHOLD = 0.05  # 置信度阈值
COOLDOWN_BARS = 3

print("=" * 80)
print("平衡模型双向交易策略")
print("=" * 80)

class BalancedBidirectionalStrategy(bt.Strategy):
    """基于平衡模型的双向交易策略"""
    
    params = (
        ('model', None),
        ('printlog', True),
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
            self.log(f'做多开仓 @ {price_now:.6f}, prob={prob:.4f}, conf={confidence:.4f}')
        
        # 做空信号
        elif prob < 0.5:
            self.order = self.sell(size=size)
            self.entry_price = price_now
            self.entry_bar = current_bar
            self.position_type = 'short'
            self.log(f'做空开仓 @ {price_now:.6f}, prob={prob:.4f}, conf={confidence:.4f}')
    
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
            exit_reason = f"止损"
        
        # 止盈
        elif ret >= TAKE_PROFIT_PCT:
            should_exit = True
            exit_reason = f"止盈"
        
        # 时间止损
        elif (current_bar - self.entry_bar) >= TIME_STOP_BARS:
            should_exit = True
            exit_reason = f"时间止损"
        
        # 信号反转（高置信度）
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
                'bars': current_bar - self.entry_bar
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
    
    print(f"\n🤖 加载平衡模型...")
    model_dir = BASE_PATH / 'models' / 'registry'
    latest_path = model_dir / 'latest_balanced.txt'
    
    if not latest_path.exists():
        print("❌ 未找到平衡模型")
        return None
    
    with open(latest_path, 'r') as f:
        version = f.read().strip()
    
    model_path = model_dir / f'{version}_balanced.pkl'
    model = joblib.load(model_path)
    print(f"✅ 模型加载成功: {version}")
    
    print(f"\n📊 策略参数:")
    print(f"  置信度阈值: {CONFIDENCE_THRESHOLD:.2f}")
    print(f"  做多条件: prob > {0.5 + CONFIDENCE_THRESHOLD:.2f}")
    print(f"  做空条件: prob < {0.5 - CONFIDENCE_THRESHOLD:.2f}")
    print(f"  止损: {STOP_LOSS_PCT*100:.0f}%")
    print(f"  止盈: {TAKE_PROFIT_PCT*100:.0f}%")
    print(f"  盈亏比: {TAKE_PROFIT_PCT/STOP_LOSS_PCT:.1f}:1")
    print(f"  时间止损: {TIME_STOP_BARS}根K线")
    print(f"  冷却期: {COOLDOWN_BARS}根K线")
    
    data = bt.feeds.PandasData(dataname=df, datetime=None, open='open', high='high', 
                               low='low', close='close', volume='volume', openinterest=-1)
    
    cerebro = bt.Cerebro()
    cerebro.adddata(data)
    cerebro.addstrategy(BalancedBidirectionalStrategy, model=model, printlog=True)
    cerebro.broker.setcash(INITIAL_CASH)
    cerebro.broker.setcommission(commission=0.001)
    
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    
    start_value = cerebro.broker.getvalue()
    print(f"\n⏳ 开始回测...")
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
        
        days = (df.index[-1] - df.index[0]).days
        monthly_trades = trades.total.total / days * 30
        print(f"\n⏱️  回测周期: {days}天")
        print(f"  月均交易: {monthly_trades:.1f}次")
        
        # 统计做多和做空
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
            print(f"    最大收益: {max(long_returns)*100:+.2f}%")
            print(f"    最大亏损: {min(long_returns)*100:+.2f}%")
        
        if short_trades:
            short_returns = [t['return'] for t in short_trades]
            short_wins = sum(1 for r in short_returns if r > 0)
            print(f"\n  做空详情:")
            print(f"    胜率: {short_wins/len(short_trades)*100:.1f}%")
            print(f"    平均收益: {np.mean(short_returns)*100:+.2f}%")
            print(f"    最大收益: {max(short_returns)*100:+.2f}%")
            print(f"    最大亏损: {min(short_returns)*100:+.2f}%")
    
    print(f"\n📉 风险指标:")
    print(f"  最大回撤: {drawdown.max.drawdown:.2f}%")
    
    if len(strat.trades) > 0:
        print(f"\n📋 交易明细 (前20笔):")
        for i, t in enumerate(strat.trades[:20], 1):
            direction = "做多" if t['type'] == 'long' else "做空"
            print(f"  {i:2d}. {direction} {t['entry']:.6f}→{t['exit']:.6f} "
                  f"{t['return']*100:+.2f}% {t['bars']:3d}根 {t['reason']}")
    
    print(f"\n{'='*80}")
    
    return {
        'pnl_pct': pnl_pct,
        'trades': trades.total.total if trades.total.total > 0 else 0,
        'long_trades': len(long_trades) if 'long_trades' in locals() else 0,
        'short_trades': len(short_trades) if 'short_trades' in locals() else 0,
    }

if __name__ == "__main__":
    result = run_backtest()
    sys.exit(0 if result and result['trades'] > 0 else 1)
