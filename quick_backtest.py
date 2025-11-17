#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速回测脚本 - 使用模拟数据测试优化后的策略
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 80)
print("ButterflyBot 优化策略快速回测")
print("=" * 80)

# 生成模拟数据用于测试
def generate_mock_data(days=180, timeframe='15m'):
    """生成模拟的OHLCV数据"""
    print(f"\n📊 生成 {days} 天的模拟数据 (周期: {timeframe})...")
    
    # 计算数据点数量
    if timeframe == '15m':
        points_per_day = 96  # 24 * 4
    elif timeframe == '1h':
        points_per_day = 24
    elif timeframe == '1d':
        points_per_day = 1
    else:
        points_per_day = 96
    
    total_points = days * points_per_day
    
    # 生成时间序列
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days)
    
    if timeframe == '15m':
        freq = '15T'
    elif timeframe == '1h':
        freq = '1H'
    elif timeframe == '1d':
        freq = '1D'
    else:
        freq = '15T'
    
    timestamps = pd.date_range(start=start_time, end=end_time, freq=freq)
    
    # 生成价格数据（模拟趋势+震荡）
    np.random.seed(42)
    base_price = 0.08  # DOGE 基础价格
    
    # 生成带趋势的随机游走
    returns = np.random.normal(0.0001, 0.02, len(timestamps))  # 均值略为正，模拟上涨趋势
    prices = base_price * np.exp(np.cumsum(returns))
    
    # 生成OHLCV
    data = []
    for i, ts in enumerate(timestamps):
        close = prices[i]
        # 添加日内波动
        high = close * (1 + abs(np.random.normal(0, 0.01)))
        low = close * (1 - abs(np.random.normal(0, 0.01)))
        open_price = close * (1 + np.random.normal(0, 0.005))
        volume = abs(np.random.normal(1000000, 500000))
        
        data.append({
            'timestamp': ts,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
    
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    
    print(f"✅ 生成了 {len(df)} 条数据")
    print(f"   价格范围: {df['close'].min():.6f} - {df['close'].max():.6f}")
    print(f"   时间范围: {df.index[0]} 至 {df.index[-1]}")
    
    return df

# 保存模拟数据到缓存
def save_mock_data_to_cache():
    """生成并保存模拟数据到缓存目录"""
    from config.settings import SYMBOL, TIMEFRAME, EXCHANGE_NAME, BASE_PATH
    
    # 创建缓存目录
    cache_dir = BASE_PATH / 'cached_data'
    os.makedirs(cache_dir, exist_ok=True)
    
    # 生成数据
    df = generate_mock_data(days=180, timeframe=TIMEFRAME)
    
    # 保存到缓存
    filename = f"{EXCHANGE_NAME}_{SYMBOL.replace('/', '_')}_{TIMEFRAME}.csv"
    cache_path = cache_dir / filename
    
    df.to_csv(cache_path)
    print(f"\n💾 数据已保存至: {cache_path}")
    
    return df

# 简单的模拟模型
class MockModel:
    """模拟模型，用于测试策略逻辑"""
    def __init__(self):
        print("\n🤖 初始化模拟模型...")
        
    def predict(self, df):
        """基于简单规则生成预测概率"""
        if df is None or len(df) == 0:
            return 0.5
        
        # 使用最后一行的技术指标
        last_row = df.iloc[-1]
        
        # 简单的规则：RSI低+MACD金叉 -> 高概率
        prob = 0.5  # 基础概率
        
        # RSI 因子
        if 'rsi' in last_row and not pd.isna(last_row['rsi']):
            if last_row['rsi'] < 30:
                prob += 0.2
            elif last_row['rsi'] > 70:
                prob -= 0.2
        
        # MACD 因子
        if 'macd_hist' in last_row and not pd.isna(last_row['macd_hist']):
            if last_row['macd_hist'] > 0:
                prob += 0.1
            else:
                prob -= 0.1
        
        # 均线因子
        if 'ma_diff' in last_row and not pd.isna(last_row['ma_diff']):
            if last_row['ma_diff'] > 0:
                prob += 0.1
            else:
                prob -= 0.1
        
        # 限制在 [0, 1] 范围
        prob = max(0.0, min(1.0, prob))
        
        return prob

def run_quick_backtest():
    """运行快速回测"""
    import backtrader as bt
    from config.settings import INITIAL_CASH, SYMBOL, TIMEFRAME
    from backtest.run_backtest import AIButterflyStrategy
    
    print("\n" + "=" * 80)
    print("开始回测")
    print("=" * 80)
    
    # 生成或加载数据
    df = save_mock_data_to_cache()
    
    # 创建 Backtrader 数据源
    data = bt.feeds.PandasData(
        dataname=df,
        datetime=None,  # 使用索引作为时间
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
    
    # 创建模拟模型
    mock_model = MockModel()
    
    # 添加策略
    cerebro.addstrategy(AIButterflyStrategy, model=mock_model, printlog=False)
    
    # 设置初始资金
    cerebro.broker.setcash(INITIAL_CASH)
    
    # 设置手续费
    cerebro.broker.setcommission(commission=0.001)  # 0.1%
    
    # 记录初始资金
    start_value = cerebro.broker.getvalue()
    print(f"\n💰 初始资金: ${start_value:.2f}")
    print(f"📈 交易对: {SYMBOL}")
    print(f"⏰ 周期: {TIMEFRAME}")
    print(f"📊 数据量: {len(df)} 条")
    
    # 运行回测
    print("\n🚀 运行回测中...")
    try:
        results = cerebro.run()
        
        # 获取最终资金
        end_value = cerebro.broker.getvalue()
        pnl = end_value - start_value
        pnl_pct = (pnl / start_value) * 100
        
        print("\n" + "=" * 80)
        print("回测结果")
        print("=" * 80)
        print(f"初始资金: ${start_value:.2f}")
        print(f"最终资金: ${end_value:.2f}")
        print(f"盈亏金额: ${pnl:.2f}")
        print(f"盈亏比例: {pnl_pct:.2f}%")
        
        if pnl > 0:
            print("\n✅ 策略盈利！")
        else:
            print("\n❌ 策略亏损")
        
        print("=" * 80)
        
        # 尝试获取交易统计
        strat = results[0]
        if hasattr(strat, 'trade_list') and len(strat.trade_list) > 0:
            print(f"\n📊 交易统计:")
            print(f"   总交易次数: {len(strat.trade_list)}")
            
            wins = [t for t in strat.trade_list if t.get('pnl', 0) > 0]
            losses = [t for t in strat.trade_list if t.get('pnl', 0) < 0]
            
            if len(strat.trade_list) > 0:
                win_rate = len(wins) / len(strat.trade_list) * 100
                print(f"   胜率: {win_rate:.1f}%")
                print(f"   盈利交易: {len(wins)}")
                print(f"   亏损交易: {len(losses)}")
                
                if len(wins) > 0:
                    avg_win = sum([t['pnl'] for t in wins]) / len(wins)
                    print(f"   平均盈利: ${avg_win:.2f}")
                
                if len(losses) > 0:
                    avg_loss = sum([t['pnl'] for t in losses]) / len(losses)
                    print(f"   平均亏损: ${avg_loss:.2f}")
        
        return pnl > 0
        
    except Exception as e:
        print(f"\n❌ 回测失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    try:
        success = run_quick_backtest()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n💥 程序异常: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
