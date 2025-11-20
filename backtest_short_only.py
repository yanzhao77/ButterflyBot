#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
纯做空策略 - 顺应模型预测特性
由于模型识别下跌趋势很准确，专注于做空
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
    STOP_LOSS_PCT, TAKE_PROFIT_PCT,
    MAX_POSITION_RATIO, TIME_STOP_BARS, COOLDOWN_BARS,
    USE_TRAILING_STOP, TRAILING_STOP_ACTIVATION, TRAILING_STOP_DISTANCE
)
from data.features import add_features
from model.model_registry import load_latest_model_path

print("=" * 80)
print("ButterflyBot 纯做空策略")
print("=" * 80)

class ShortOnlyStrategy(bt.Strategy):
    """纯做空策略 - 在低概率时做空"""
    
    params = (
        ('model', None),
        ('printlog', True),
        ('window', 200),  # 滚动窗口
        ('short_quantile', 0.35),  # 做空分位数（bottom 35%）
        ('exit_quantile', 0.65),  # 平仓分位数（top 35%）
    )
    
    def __init__(self):
        self.data_close = self.datas[0].close
        self.order = None
        self.entry_price = None
        self.entry_bar = None
        self.cooldown_until = -1
        self.lowest_price = None
        self.trailing_active = False
        
        # 概率历史
        self.prob_history = deque(maxlen=self.params.window)
        
        # 统计
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
        
        # 获取历史数据
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
        
        # 添加特征
        try:
            df_feat = add_features(df)
            if df_feat.empty:
                return
        except:
            return
        
        # 预测
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
        
        # 需要足够的历史数据
        if len(self.prob_history) < 50:
            return
        
        # 计算动态阈值
        probs = np.array(self.prob_history)
        short_thresh = np.percentile(probs, self.params.short_quantile * 100)
        exit_thresh = np.percentile(probs, self.params.exit_quantile * 100)
        
        price_now = self.data_close[0]
        
        # 如果有持仓，检查平仓
        if self.position:
            self.check_exit(price_now, prob, current_bar, exit_thresh)
        else:
            # 无持仓，检查做空信号
            self.check_short_entry(price_now, prob, current_bar, short_thresh)
    
    def check_short_entry(self, price_now, prob, current_bar, short_thresh):
        """检查做空开仓信号"""
        
        # 做空信号：概率在bottom 20%
        if prob <= short_thresh:
            size = int((self.broker.getvalue() * MAX_POSITION_RATIO) / price_now)
            if size > 0:
                self.order = self.sell(size=size)
                self.entry_price = price_now
                self.entry_bar = current_bar
                self.lowest_price = price_now
                self.trailing_active = False
                self.log(f'做空开仓 @ {price_now:.6f}, prob={prob:.4f} (阈值≤{short_thresh:.4f}), 数量={size}')
    
    def check_exit(self, price_now, prob, current_bar, exit_thresh):
        """检查平仓信号"""
        
        if not self.entry_price:
            return
        
        should_exit = False
        exit_reason = ""
        
        # 做空收益计算
        ret = (self.entry_price - price_now) / self.entry_price
        
        # 止损
        if ret <= -STOP_LOSS_PCT:
            should_exit = True
            exit_reason = f"止损 (亏损{ret*100:.2f}%)"
        
        # 止盈
        elif ret >= TAKE_PROFIT_PCT:
            should_exit = True
            exit_reason = f"止盈 (盈利{ret*100:.2f}%)"
        
        # 跟踪止盈
        elif USE_TRAILING_STOP and ret >= TRAILING_STOP_ACTIVATION:
            self.trailing_active = True
            if self.lowest_price is None or price_now < self.lowest_price:
                self.lowest_price = price_now
            
            if self.lowest_price is not None:
                drawup = (price_now - self.lowest_price) / self.lowest_price
                if drawup >= TRAILING_STOP_DISTANCE:
                    should_exit = True
                    exit_reason = f"跟踪止盈 (从最低点反弹{drawup*100:.2f}%)"
        
        # 时间止损
        elif (current_bar - self.entry_bar) >= TIME_STOP_BARS:
            should_exit = True
            exit_reason = f"时间止损 ({current_bar - self.entry_bar}根K线)"
        
        # 概率反转（回到top 40%）
        elif prob >= exit_thresh:
            should_exit = True
            exit_reason = f"概率反转 (prob={prob:.4f} >= {exit_thresh:.4f})"
        
        if should_exit:
            self.order = self.close()
            
            self.trades.append({
                'entry': self.entry_price,
                'exit': price_now,
                'return': ret,
                'reason': exit_reason,
            })
            
            self.log(f'平仓 @ {price_now:.6f}, 收益={ret*100:+.2f}%, {exit_reason}')
            
            self.entry_price = None
            self.entry_bar = None
            self.lowest_price = None
            self.trailing_active = False
            self.cooldown_until = current_bar + COOLDOWN_BARS

def load_real_data():
    """加载真实历史数据"""
    cache_dir = BASE_PATH / 'cached_data'
    filename = f"binance_{SYMBOL.replace('/', '_')}_{TIMEFRAME}.csv"
    cache_path = cache_dir / filename
    
    if not cache_path.exists():
        print(f"❌ 数据文件不存在: {cache_path}")
        return None
    
    print(f"\n📂 加载数据: {cache_path}")
    df = pd.read_csv(cache_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df.set_index('timestamp', inplace=True)
    print(f"✅ 加载成功: {len(df)} 条数据")
    return df

def run_backtest(df, model):
    """运行回测"""
    
    print("\n" + "=" * 80)
    print("回测配置")
    print("=" * 80)
    print(f"初始资金: ${INITIAL_CASH:.2f}")
    print(f"策略: 纯做空（顺应模型特性）")
    print(f"  做空: bottom 35% 概率")
    print(f"  平仓: top 35% 概率")
    print(f"  止损: {STOP_LOSS_PCT*100:.1f}%")
    print(f"  止盈: {TAKE_PROFIT_PCT*100:.1f}%")
    print(f"  盈亏比: {TAKE_PROFIT_PCT/STOP_LOSS_PCT:.2f}:1")
    
    data = bt.feeds.PandasData(dataname=df, datetime=None, open='open', high='high', 
                               low='low', close='close', volume='volume', openinterest=-1)
    
    cerebro = bt.Cerebro()
    cerebro.adddata(data)
    cerebro.addstrategy(ShortOnlyStrategy, model=model, printlog=False)
    cerebro.broker.setcash(INITIAL_CASH)
    cerebro.broker.setcommission(commission=0.001)
    
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    
    start_value = cerebro.broker.getvalue()
    print("\n开始回测...")
    results = cerebro.run()
    strat = results[0]
    end_value = cerebro.broker.getvalue()
    
    pnl = end_value - start_value
    pnl_pct = (pnl / start_value) * 100
    
    trades = strat.analyzers.trades.get_analysis()
    drawdown = strat.analyzers.drawdown.get_analysis()
    sharpe = strat.analyzers.sharpe.get_analysis()
    
    print("\n" + "=" * 80)
    print("回测结果")
    print("=" * 80)
    print(f"\n📊 资金变化:")
    print(f"   初始: ${start_value:.2f}")
    print(f"   最终: ${end_value:.2f}")
    print(f"   盈亏: ${pnl:+.2f} ({pnl_pct:+.2f}%)")
    
    if pnl > 0:
        print(f"\n✅ 策略盈利 {pnl_pct:.2f}%")
    else:
        print(f"\n❌ 策略亏损 {abs(pnl_pct):.2f}%")
    
    if trades.total.total > 0:
        print(f"\n📈 交易统计:")
        print(f"   总交易: {trades.total.total}")
        
        won = 0
        lost = 0
        try:
            won = trades.won.total
            lost = trades.lost.total
            print(f"   盈利: {won}")
            print(f"   亏损: {lost}")
            
            if won + lost > 0:
                win_rate = won / (won + lost) * 100
                print(f"   胜率: {win_rate:.1f}%")
            
            avg_win = trades.won.pnl.average
            avg_loss = abs(trades.lost.pnl.average)
            print(f"   平均盈利: ${avg_win:.2f}")
            print(f"   平均亏损: ${avg_loss:.2f}")
            
            if avg_loss > 0:
                print(f"   盈亏比: {avg_win/avg_loss:.2f}:1")
        except:
            pass
        
        # 计算交易频率
        total_days = (df.index[-1] - df.index[0]).days
        trades_per_month = (trades.total.total / total_days) * 30
        print(f"\n⏱️  交易频率:")
        print(f"   回测天数: {total_days}")
        print(f"   月均交易: {trades_per_month:.1f} 次")
    
    print(f"\n📉 风险指标:")
    if hasattr(drawdown, 'max'):
        print(f"   最大回撤: {drawdown.max.drawdown:.2f}%")
    
    if hasattr(sharpe, 'sharperatio') and sharpe.sharperatio is not None:
        print(f"   夏普比率: {sharpe.sharperatio:.2f}")
    
    # 打印交易明细
    if len(strat.trades) > 0:
        print(f"\n📋 交易明细:")
        for i, trade in enumerate(strat.trades[:10], 1):  # 只显示前10笔
            print(f"   {i}. 开仓{trade['entry']:.6f} → 平仓{trade['exit']:.6f}, "
                  f"收益{trade['return']*100:+.2f}%, {trade['reason']}")
        if len(strat.trades) > 10:
            print(f"   ... 还有 {len(strat.trades)-10} 笔交易")
    
    return {
        'success': pnl > 0,
        'pnl': pnl,
        'pnl_pct': pnl_pct,
        'total_trades': trades.total.total if trades.total.total > 0 else 0,
        'won': won,
        'lost': lost,
    }

if __name__ == "__main__":
    try:
        print("\n🤖 加载模型...")
        model_path = load_latest_model_path()
        if not model_path:
            print("❌ 未找到模型")
            sys.exit(1)
        model = joblib.load(model_path)
        print(f"✅ 模型加载成功")
        
        df = load_real_data()
        if df is None:
            sys.exit(1)
        
        result = run_backtest(df, model)
        
        print("\n" + "=" * 80)
        print("最终总结")
        print("=" * 80)
        
        if result['success']:
            print(f"\n🎉 策略成功！盈利 {result['pnl_pct']:.2f}% (${result['pnl']:.2f})")
            print(f"\n关键指标:")
            print(f"  ✅ 总交易: {result['total_trades']}")
            print(f"  ✅ 盈利交易: {result['won']}")
            print(f"  ✅ 亏损交易: {result['lost']}")
            
            if result['won'] + result['lost'] > 0:
                win_rate = result['won'] / (result['won'] + result['lost']) * 100
                print(f"  ✅ 胜率: {win_rate:.1f}%")
        else:
            print(f"\n⚠️  策略亏损 {abs(result['pnl_pct']):.2f}%")
            print(f"交易次数: {result['total_trades']}")
        
        print("\n" + "=" * 80)
        
        sys.exit(0 if result['success'] else 1)
        
    except Exception as e:
        print(f"\n💥 失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
