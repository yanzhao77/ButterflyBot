#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模拟实盘实时测试系统
模拟真实交易环境，包括滑点、延迟、手续费等
"""

import sys
import os
import time
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import INITIAL_CASH, SYMBOL, TIMEFRAME, BASE_PATH, MAX_POSITION_RATIO
from data.features import add_features

# 策略参数
STOP_LOSS_PCT = 0.02
TAKE_PROFIT_PCT = 0.03
TIME_STOP_BARS = 20
CONFIDENCE_THRESHOLD = 0.05
COOLDOWN_BARS = 3

# 模拟参数
SLIPPAGE_PCT = 0.0005  # 滑点0.05%
COMMISSION_PCT = 0.001  # 手续费0.1%
ORDER_DELAY_BARS = 1    # 订单延迟1根K线

class LiveSimulator:
    """模拟实盘交易器"""
    
    def __init__(self, model, initial_cash=1000):
        self.model = model
        self.cash = initial_cash
        self.initial_cash = initial_cash
        self.position = None  # {'type': 'long/short', 'size': 100, 'entry_price': 0.4, 'entry_bar': 0}
        self.pending_order = None  # {'type': 'buy/sell', 'size': 100, 'execute_bar': 100}
        self.cooldown_until = -1
        self.trades = []
        self.equity_curve = []
        
    def get_equity(self, current_price):
        """计算当前权益"""
        if self.position:
            if self.position['type'] == 'long':
                position_value = self.position['size'] * current_price
            else:  # short
                position_value = self.position['size'] * (2 * self.position['entry_price'] - current_price)
            return self.cash + position_value
        return self.cash
    
    def execute_order(self, order, price):
        """执行订单（考虑滑点和手续费）"""
        # 应用滑点
        if order['type'] == 'buy':
            execution_price = price * (1 + SLIPPAGE_PCT)
        else:  # sell
            execution_price = price * (1 - SLIPPAGE_PCT)
        
        cost = order['size'] * execution_price
        commission = cost * COMMISSION_PCT
        
        if order['type'] == 'buy':
            # 开多仓
            self.cash -= (cost + commission)
            self.position = {
                'type': 'long',
                'size': order['size'],
                'entry_price': execution_price,
                'entry_bar': order['execute_bar']
            }
            return f"✅ 开多仓: size={order['size']}, price={execution_price:.6f}, cost=${cost:.2f}, fee=${commission:.2f}"
        
        elif order['type'] == 'sell':
            if self.position and self.position['type'] == 'long':
                # 平多仓
                revenue = order['size'] * execution_price
                self.cash += (revenue - commission)
                ret = (execution_price - self.position['entry_price']) / self.position['entry_price']
                pnl = (execution_price - self.position['entry_price']) * order['size'] - commission
                
                self.trades.append({
                    'type': 'long',
                    'entry': self.position['entry_price'],
                    'exit': execution_price,
                    'return': ret,
                    'pnl': pnl,
                    'bars': order['execute_bar'] - self.position['entry_bar']
                })
                
                self.position = None
                return f"✅ 平多仓: price={execution_price:.6f}, return={ret*100:+.2f}%, pnl=${pnl:+.2f}"
            else:
                # 开空仓
                self.cash -= commission
                self.position = {
                    'type': 'short',
                    'size': order['size'],
                    'entry_price': execution_price,
                    'entry_bar': order['execute_bar']
                }
                return f"✅ 开空仓: size={order['size']}, price={execution_price:.6f}, fee=${commission:.2f}"
        
        elif order['type'] == 'buy_to_cover':
            # 平空仓
            cost = order['size'] * execution_price
            pnl = (self.position['entry_price'] - execution_price) * order['size'] - commission
            self.cash += pnl
            ret = (self.position['entry_price'] - execution_price) / self.position['entry_price']
            
            self.trades.append({
                'type': 'short',
                'entry': self.position['entry_price'],
                'exit': execution_price,
                'return': ret,
                'pnl': pnl,
                'bars': order['execute_bar'] - self.position['entry_bar']
            })
            
            self.position = None
            return f"✅ 平空仓: price={execution_price:.6f}, return={ret*100:+.2f}%, pnl=${pnl:+.2f}"
    
    def process_bar(self, bar_idx, row, prob):
        """处理每根K线"""
        price = row['close']
        timestamp = row.name
        
        # 记录权益曲线
        equity = self.get_equity(price)
        self.equity_curve.append({
            'timestamp': timestamp,
            'equity': equity,
            'cash': self.cash,
            'position': 'long' if self.position and self.position['type'] == 'long' else 'short' if self.position else 'none'
        })
        
        # 执行挂单
        if self.pending_order and bar_idx >= self.pending_order['execute_bar']:
            result = self.execute_order(self.pending_order, price)
            print(f"[{timestamp}] {result}")
            self.pending_order = None
        
        # 如果有挂单，等待执行
        if self.pending_order:
            return
        
        # 冷却期
        if bar_idx <= self.cooldown_until:
            return
        
        confidence = abs(prob - 0.5)
        
        # 检查平仓
        if self.position:
            self.check_exit(bar_idx, price, prob, confidence, timestamp)
        else:
            self.check_entry(bar_idx, price, prob, confidence, timestamp)
    
    def check_entry(self, bar_idx, price, prob, confidence, timestamp):
        """检查开仓"""
        if confidence < CONFIDENCE_THRESHOLD:
            return
        
        # 计算仓位大小
        equity = self.get_equity(price)
        position_value = equity * MAX_POSITION_RATIO
        size = int(position_value / price)
        
        if size <= 0:
            return
        
        # 提交订单
        if prob > 0.5:
            print(f"[{timestamp}] 🔵 做多信号: prob={prob:.4f}, confidence={confidence:.4f}")
            self.pending_order = {
                'type': 'buy',
                'size': size,
                'execute_bar': bar_idx + ORDER_DELAY_BARS
            }
        elif prob < 0.5:
            print(f"[{timestamp}] 🔴 做空信号: prob={prob:.4f}, confidence={confidence:.4f}")
            self.pending_order = {
                'type': 'sell',
                'size': size,
                'execute_bar': bar_idx + ORDER_DELAY_BARS
            }
    
    def check_exit(self, bar_idx, price, prob, confidence, timestamp):
        """检查平仓"""
        if not self.position:
            return
        
        # 计算收益
        if self.position['type'] == 'long':
            ret = (price - self.position['entry_price']) / self.position['entry_price']
        else:
            ret = (self.position['entry_price'] - price) / self.position['entry_price']
        
        bars_held = bar_idx - self.position['entry_bar']
        
        should_exit = False
        exit_reason = ""
        
        # 止损
        if ret <= -STOP_LOSS_PCT:
            should_exit = True
            exit_reason = "止损"
        # 止盈
        elif ret >= TAKE_PROFIT_PCT:
            should_exit = True
            exit_reason = "止盈"
        # 时间止损
        elif bars_held >= TIME_STOP_BARS:
            should_exit = True
            exit_reason = "时间止损"
        # 信号反转
        elif self.position['type'] == 'long' and prob < 0.5 and confidence >= CONFIDENCE_THRESHOLD:
            should_exit = True
            exit_reason = "信号反转"
        elif self.position['type'] == 'short' and prob > 0.5 and confidence >= CONFIDENCE_THRESHOLD:
            should_exit = True
            exit_reason = "信号反转"
        
        if should_exit:
            print(f"[{timestamp}] ⭕ 平仓信号: {exit_reason}, return={ret*100:+.2f}%, bars={bars_held}")
            
            # 提交平仓订单
            if self.position['type'] == 'long':
                self.pending_order = {
                    'type': 'sell',
                    'size': self.position['size'],
                    'execute_bar': bar_idx + ORDER_DELAY_BARS
                }
            else:
                self.pending_order = {
                    'type': 'buy_to_cover',
                    'size': self.position['size'],
                    'execute_bar': bar_idx + ORDER_DELAY_BARS
                }
            
            self.cooldown_until = bar_idx + COOLDOWN_BARS

def run_live_simulation():
    """运行模拟实盘测试"""
    
    print("="*80)
    print("模拟实盘实时测试")
    print("="*80)
    
    # 加载数据
    cache_path = BASE_PATH / 'cached_data' / f"binance_{SYMBOL.replace('/', '_')}_{TIMEFRAME}.csv"
    print(f"\n📂 加载历史数据...")
    df = pd.read_csv(cache_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df.set_index('timestamp', inplace=True)
    print(f"✅ {len(df)}条数据")
    
    # 加载模型
    print(f"\n🤖 加载平衡模型...")
    model_dir = BASE_PATH / 'models' / 'registry'
    with open(model_dir / 'latest_balanced.txt', 'r') as f:
        version = f.read().strip()
    model = joblib.load(model_dir / f'{version}_balanced.pkl')
    print(f"✅ 模型: {version}")
    
    # 策略参数
    print(f"\n📊 策略参数:")
    print(f"  置信度阈值: {CONFIDENCE_THRESHOLD}")
    print(f"  止损: {STOP_LOSS_PCT*100}%")
    print(f"  止盈: {TAKE_PROFIT_PCT*100}%")
    print(f"  时间止损: {TIME_STOP_BARS}根")
    print(f"  滑点: {SLIPPAGE_PCT*100}%")
    print(f"  手续费: {COMMISSION_PCT*100}%")
    print(f"  订单延迟: {ORDER_DELAY_BARS}根K线")
    
    # 使用最近的数据进行测试（模拟实时）
    test_days = 30  # 测试最近30天
    test_bars = test_days * 24 * 4  # 15分钟K线
    df_test = df.iloc[-test_bars:].copy()
    
    print(f"\n📅 测试周期:")
    print(f"  开始: {df_test.index[0]}")
    print(f"  结束: {df_test.index[-1]}")
    print(f"  K线数: {len(df_test)}")
    
    # 计算特征
    print(f"\n🔧 计算特征...")
    feature_cols = ['open', 'high', 'low', 'close', 'volume', 'return', 'log_return',
                   'ma20', 'ma50', 'ma_diff', 'rsi', 'macd', 'macd_signal', 
                   'macd_hist', 'volatility', 'volume_ratio']
    
    # 需要使用全部数据计算特征（因为需要历史数据）
    df_full = add_features(df)
    df_test = df_full.iloc[-test_bars:].copy()
    df_test = df_test.dropna(subset=feature_cols)
    
    print(f"✅ {len(df_test)}条有效数据")
    
    # 预测
    print(f"\n🔮 生成预测...")
    X = df_test[feature_cols].values
    predictions = model.predict(X)
    print(f"✅ 预测完成")
    
    # 创建模拟器
    simulator = LiveSimulator(model, initial_cash=INITIAL_CASH)
    
    print(f"\n{'='*80}")
    print("开始模拟实盘交易")
    print(f"{'='*80}\n")
    
    # 逐根K线处理
    for idx, (timestamp, row) in enumerate(df_test.iterrows()):
        prob = predictions[idx]
        simulator.process_bar(idx, row, prob)
        
        # 每100根K线显示一次状态
        if (idx + 1) % 100 == 0:
            equity = simulator.get_equity(row['close'])
            pnl_pct = (equity - simulator.initial_cash) / simulator.initial_cash * 100
            print(f"\n[{timestamp}] 📊 状态更新:")
            print(f"  K线: {idx+1}/{len(df_test)}")
            print(f"  权益: ${equity:.2f} ({pnl_pct:+.2f}%)")
            print(f"  现金: ${simulator.cash:.2f}")
            print(f"  持仓: {simulator.position['type'] if simulator.position else '无'}")
            print(f"  交易数: {len(simulator.trades)}")
    
    # 最终结果
    final_price = df_test.iloc[-1]['close']
    final_equity = simulator.get_equity(final_price)
    final_pnl = final_equity - simulator.initial_cash
    final_pnl_pct = (final_pnl / simulator.initial_cash) * 100
    
    print(f"\n{'='*80}")
    print("模拟实盘测试结果")
    print(f"{'='*80}")
    
    print(f"\n💰 资金变化:")
    print(f"  初始: ${simulator.initial_cash:.2f}")
    print(f"  最终: ${final_equity:.2f}")
    print(f"  盈亏: ${final_pnl:+.2f} ({final_pnl_pct:+.2f}%)")
    
    if len(simulator.trades) > 0:
        print(f"\n📈 交易统计:")
        print(f"  总交易: {len(simulator.trades)}次")
        
        wins = [t for t in simulator.trades if t['pnl'] > 0]
        losses = [t for t in simulator.trades if t['pnl'] <= 0]
        
        print(f"  盈利: {len(wins)}次")
        print(f"  亏损: {len(losses)}次")
        print(f"  胜率: {len(wins)/len(simulator.trades)*100:.1f}%")
        
        if wins:
            avg_win = np.mean([t['pnl'] for t in wins])
            print(f"  平均盈利: ${avg_win:.2f}")
        
        if losses:
            avg_loss = np.mean([t['pnl'] for t in losses])
            print(f"  平均亏损: ${avg_loss:.2f}")
        
        # 分类统计
        long_trades = [t for t in simulator.trades if t['type'] == 'long']
        short_trades = [t for t in simulator.trades if t['type'] == 'short']
        
        print(f"\n📊 交易类型:")
        print(f"  做多: {len(long_trades)}次")
        print(f"  做空: {len(short_trades)}次")
        
        if long_trades:
            long_wins = sum(1 for t in long_trades if t['pnl'] > 0)
            long_pnl = sum(t['pnl'] for t in long_trades)
            print(f"\n  做多详情:")
            print(f"    胜率: {long_wins/len(long_trades)*100:.1f}%")
            print(f"    总盈亏: ${long_pnl:+.2f}")
            print(f"    平均收益: {np.mean([t['return'] for t in long_trades])*100:+.2f}%")
        
        if short_trades:
            short_wins = sum(1 for t in short_trades if t['pnl'] > 0)
            short_pnl = sum(t['pnl'] for t in short_trades)
            print(f"\n  做空详情:")
            print(f"    胜率: {short_wins/len(short_trades)*100:.1f}%")
            print(f"    总盈亏: ${short_pnl:+.2f}")
            print(f"    平均收益: {np.mean([t['return'] for t in short_trades])*100:+.2f}%")
        
        # 权益曲线分析
        equity_df = pd.DataFrame(simulator.equity_curve)
        max_equity = equity_df['equity'].max()
        drawdowns = (equity_df['equity'] - max_equity) / max_equity * 100
        max_drawdown = drawdowns.min()
        
        print(f"\n📉 风险指标:")
        print(f"  最大回撤: {abs(max_drawdown):.2f}%")
        print(f"  最高权益: ${max_equity:.2f}")
        
        # 交易明细
        print(f"\n📋 交易明细:")
        for i, t in enumerate(simulator.trades, 1):
            direction = "做多" if t['type'] == 'long' else "做空"
            print(f"  {i:2d}. {direction} {t['return']*100:+6.2f}% ${t['pnl']:+7.2f} {t['bars']:3d}根")
    
    else:
        print(f"\n⚠️  没有产生任何交易")
    
    print(f"\n{'='*80}")
    
    return {
        'final_pnl_pct': final_pnl_pct,
        'trades': len(simulator.trades),
        'win_rate': len(wins)/len(simulator.trades)*100 if simulator.trades else 0,
    }

if __name__ == "__main__":
    result = run_live_simulation()
