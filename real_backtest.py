#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用真实历史数据进行回测
"""

import sys
import os
import pandas as pd
import numpy as np
import backtrader as bt
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import (
    INITIAL_CASH, SYMBOL, TIMEFRAME, BASE_PATH,
    CONFIDENCE_THRESHOLD, SELL_THRESHOLD,
    STOP_LOSS_PCT, TAKE_PROFIT_PCT,
    MAX_POSITION_RATIO, TIME_STOP_BARS, COOLDOWN_BARS,
    USE_TRAILING_STOP, TRAILING_STOP_ACTIVATION, TRAILING_STOP_DISTANCE
)
from data.features import add_features
from backtest.run_backtest import AIButterflyStrategy

print("=" * 80)
print("ButterflyBot 真实数据回测")
print("=" * 80)

# 简化的模拟模型（用于测试策略逻辑）
class SimpleModel:
    """基于技术指标的简单模型"""
    
    def predict(self, df):
        """基于技术指标生成预测概率"""
        if df is None or len(df) == 0:
            return 0.5
        
        try:
            last_row = df.iloc[-1]
            prob = 0.5
            
            # RSI 信号
            if 'rsi' in last_row and not pd.isna(last_row['rsi']):
                rsi = last_row['rsi']
                if rsi < 30:
                    prob += 0.25
                elif rsi < 40:
                    prob += 0.15
                elif rsi > 70:
                    prob -= 0.25
                elif rsi > 60:
                    prob -= 0.15
            
            # MACD 信号
            if 'macd_hist' in last_row and not pd.isna(last_row['macd_hist']):
                if last_row['macd_hist'] > 0:
                    prob += 0.15
                else:
                    prob -= 0.10
            
            # 均线信号
            if 'ma_diff' in last_row and not pd.isna(last_row['ma_diff']):
                if last_row['ma_diff'] > 0:
                    prob += 0.10
                else:
                    prob -= 0.10
            
            # 成交量信号
            if 'volume_ratio' in last_row and not pd.isna(last_row['volume_ratio']):
                if last_row['volume_ratio'] > 1.5:
                    prob += 0.10
                elif last_row['volume_ratio'] < 0.7:
                    prob -= 0.05
            
            # 限制在 [0, 1] 范围
            prob = max(0.0, min(1.0, prob))
            
            return prob
            
        except Exception as e:
            print(f"[WARNING] 预测出错: {e}")
            return 0.5

def load_real_data():
    """加载真实历史数据"""
    cache_dir = BASE_PATH / 'cached_data'
    filename = f"binance_{SYMBOL.replace('/', '_')}_{TIMEFRAME}.csv"
    cache_path = cache_dir / filename
    
    if not cache_path.exists():
        print(f"❌ 数据文件不存在: {cache_path}")
        print("请先运行: python3 fetch_real_data.py")
        return None
    
    print(f"\n📂 加载数据: {cache_path}")
    
    df = pd.read_csv(cache_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df.set_index('timestamp', inplace=True)
    
    print(f"✅ 加载成功: {len(df)} 条数据")
    print(f"   时间范围: {df.index[0]} 至 {df.index[-1]}")
    print(f"   价格范围: {df['close'].min():.6f} - {df['close'].max():.6f}")
    
    return df

def run_backtest(df, model):
    """运行回测"""
    
    print("\n" + "=" * 80)
    print("回测配置")
    print("=" * 80)
    print(f"初始资金: ${INITIAL_CASH:.2f}")
    print(f"交易对: {SYMBOL}")
    print(f"周期: {TIMEFRAME}")
    print(f"数据量: {len(df)} 条")
    print(f"\n策略参数:")
    print(f"  买入阈值: {CONFIDENCE_THRESHOLD}")
    print(f"  卖出阈值: {SELL_THRESHOLD}")
    print(f"  止损: {STOP_LOSS_PCT*100:.1f}%")
    print(f"  止盈: {TAKE_PROFIT_PCT*100:.1f}%")
    print(f"  盈亏比: {TAKE_PROFIT_PCT/STOP_LOSS_PCT:.2f}:1")
    print(f"  仓位比例: {MAX_POSITION_RATIO*100:.0f}%")
    print(f"  时间止损: {TIME_STOP_BARS} 根K线")
    print(f"  冷却期: {COOLDOWN_BARS} 根K线")
    if USE_TRAILING_STOP:
        print(f"  跟踪止盈: 启用 (激活{TRAILING_STOP_ACTIVATION*100:.0f}%, 回撤{TRAILING_STOP_DISTANCE*100:.1f}%)")
    
    # 创建 Backtrader 数据源
    data = bt.feeds.PandasData(
        dataname=df,
        datetime=None,
        open='open',
        high='high',
        low='low',
        close='close',
        volume='volume',
        openinterest=-1
    )
    
    # 创建 Cerebro 引擎
    cerebro = bt.Cerebro()
    cerebro.adddata(data)
    
    # 添加策略
    cerebro.addstrategy(AIButterflyStrategy, model=model, printlog=False)
    
    # 设置初始资金和手续费
    cerebro.broker.setcash(INITIAL_CASH)
    cerebro.broker.setcommission(commission=0.001)  # 0.1%
    
    # 添加分析器
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    
    # 记录初始资金
    start_value = cerebro.broker.getvalue()
    start_time = datetime.now()
    
    print("\n" + "=" * 80)
    print("开始回测...")
    print("=" * 80)
    
    # 运行回测
    results = cerebro.run()
    strat = results[0]
    
    # 获取最终资金
    end_value = cerebro.broker.getvalue()
    end_time = datetime.now()
    
    # 计算收益
    pnl = end_value - start_value
    pnl_pct = (pnl / start_value) * 100
    
    # 获取分析结果
    sharpe = strat.analyzers.sharpe.get_analysis()
    drawdown = strat.analyzers.drawdown.get_analysis()
    returns = strat.analyzers.returns.get_analysis()
    trades = strat.analyzers.trades.get_analysis()
    
    # 打印结果
    print("\n" + "=" * 80)
    print("回测结果")
    print("=" * 80)
    
    print(f"\n📊 资金变化:")
    print(f"   初始资金: ${start_value:.2f}")
    print(f"   最终资金: ${end_value:.2f}")
    print(f"   盈亏金额: ${pnl:+.2f}")
    print(f"   盈亏比例: {pnl_pct:+.2f}%")
    
    if pnl > 0:
        print(f"\n✅ 策略盈利 {pnl_pct:.2f}%")
    else:
        print(f"\n❌ 策略亏损 {abs(pnl_pct):.2f}%")
    
    # 交易统计
    if trades.total.total > 0:
        print(f"\n📈 交易统计:")
        print(f"   总交易次数: {trades.total.total}")
        print(f"   盈利交易: {trades.won.total if hasattr(trades, 'won') else 0}")
        print(f"   亏损交易: {trades.lost.total if hasattr(trades, 'lost') else 0}")
        
        if hasattr(trades, 'won') and trades.total.total > 0:
            win_rate = (trades.won.total / trades.total.total) * 100
            print(f"   胜率: {win_rate:.1f}%")
            
            if hasattr(trades.won, 'pnl') and hasattr(trades.lost, 'pnl'):
                avg_win = trades.won.pnl.average if trades.won.total > 0 else 0
                avg_loss = abs(trades.lost.pnl.average) if trades.lost.total > 0 else 0
                print(f"   平均盈利: ${avg_win:.2f}")
                print(f"   平均亏损: ${avg_loss:.2f}")
                
                if avg_loss > 0:
                    actual_ratio = avg_win / avg_loss
                    print(f"   实际盈亏比: {actual_ratio:.2f}:1")
    else:
        print(f"\n⚠️  未产生任何交易")
    
    # 风险指标
    print(f"\n📉 风险指标:")
    if hasattr(drawdown, 'max'):
        print(f"   最大回撤: {drawdown.max.drawdown:.2f}%")
        print(f"   最大回撤金额: ${drawdown.max.moneydown:.2f}")
    
    if hasattr(sharpe, 'sharperatio') and sharpe.sharperatio is not None:
        print(f"   夏普比率: {sharpe.sharperatio:.2f}")
    
    if hasattr(returns, 'rnorm100'):
        print(f"   年化收益率: {returns.rnorm100:.2f}%")
    
    # 计算交易频率
    total_days = (df.index[-1] - df.index[0]).days
    if trades.total.total > 0:
        trades_per_day = trades.total.total / total_days
        trades_per_month = trades_per_day * 30
        print(f"\n⏱️  交易频率:")
        print(f"   回测天数: {total_days} 天")
        print(f"   平均每月交易: {trades_per_month:.1f} 次")
    
    # 计算手续费
    if trades.total.total > 0:
        total_commission = trades.total.total * 2 * 0.001 * (start_value * MAX_POSITION_RATIO)
        commission_pct = (total_commission / start_value) * 100
        print(f"\n💰 手续费分析:")
        print(f"   预估总手续费: ${total_commission:.2f}")
        print(f"   手续费占比: {commission_pct:.2f}%")
    
    print(f"\n⏱️  回测耗时: {(end_time - start_time).total_seconds():.1f} 秒")
    
    print("\n" + "=" * 80)
    
    # 返回结果摘要
    return {
        'success': pnl > 0,
        'pnl': pnl,
        'pnl_pct': pnl_pct,
        'total_trades': trades.total.total if trades.total.total > 0 else 0,
        'win_rate': (trades.won.total / trades.total.total * 100) if (hasattr(trades, 'won') and trades.total.total > 0) else 0,
        'max_drawdown': drawdown.max.drawdown if hasattr(drawdown, 'max') else 0,
        'sharpe': sharpe.sharperatio if (hasattr(sharpe, 'sharperatio') and sharpe.sharperatio is not None) else None,
    }

if __name__ == "__main__":
    try:
        # 加载数据
        df = load_real_data()
        if df is None:
            sys.exit(1)
        
        # 创建模型
        print("\n🤖 初始化模型...")
        model = SimpleModel()
        print("✅ 使用基于技术指标的简单模型")
        
        # 运行回测
        result = run_backtest(df, model)
        
        # 总结
        print("\n" + "=" * 80)
        print("回测总结")
        print("=" * 80)
        
        if result['success']:
            print(f"\n🎉 回测成功！策略实现盈利 {result['pnl_pct']:.2f}%")
            print(f"\n关键指标:")
            print(f"  ✅ 盈利: ${result['pnl']:.2f} ({result['pnl_pct']:+.2f}%)")
            print(f"  ✅ 交易次数: {result['total_trades']}")
            print(f"  ✅ 胜率: {result['win_rate']:.1f}%")
            print(f"  ✅ 最大回撤: {result['max_drawdown']:.2f}%")
            if result['sharpe']:
                print(f"  ✅ 夏普比率: {result['sharpe']:.2f}")
        else:
            print(f"\n⚠️  回测显示策略亏损 {abs(result['pnl_pct']):.2f}%")
            print(f"\n可能原因:")
            print(f"  • 使用的是简化模型，未使用训练好的 AI 模型")
            print(f"  • 当前市场环境可能不适合该策略")
            print(f"  • 参数可能需要进一步优化")
            print(f"\n建议:")
            print(f"  1. 训练真实的 AI 模型: python3 -m model.train")
            print(f"  2. 使用训练好的模型进行回测")
            print(f"  3. 根据市场环境调整参数")
        
        print("\n" + "=" * 80)
        
        sys.exit(0 if result['success'] else 1)
        
    except Exception as e:
        print(f"\n💥 回测失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
