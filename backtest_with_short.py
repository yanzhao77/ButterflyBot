#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
支持做空的回测脚本
利用模型的低概率预测进行做空操作
"""

import sys
import os
import pandas as pd
import numpy as np
import backtrader as bt
from datetime import datetime
import joblib

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
print("ButterflyBot 双向交易回测（支持做空）")
print("=" * 80)

class BidirectionalStrategy(bt.Strategy):
    """支持做多和做空的双向交易策略"""
    
    params = (
        ('model', None),
        ('printlog', False),
        ('long_threshold', 0.15),  # 做多阈值（降低以增加交易）
        ('short_threshold', 0.10),  # 做空阈值（降低以增加交易）
        ('exit_threshold', 0.12),  # 平仓阈值
    )
    
    def __init__(self):
        self.data_close = self.datas[0].close
        self.order = None
        self.position_type = None  # 'long' or 'short'
        self.entry_price = None
        self.entry_bar = None
        self.cooldown_until = -1
        self.highest_price = None
        self.lowest_price = None
        self.trailing_active = False
        self.trade_count = 0
        self.win_count = 0
        self.loss_count = 0
        
    def log(self, txt, dt=None):
        if self.params.printlog:
            dt = dt or self.datas[0].datetime.date(0)
            print(f'[{dt.isoformat()}] {txt}')
    
    def next(self):
        if self.order:
            return
        
        # 冷却期检查
        current_bar = len(self)
        if current_bar <= self.cooldown_until:
            return
        
        # 获取历史数据
        total_bars = len(self)
        if total_bars < 100:  # 需要足够的历史数据计算特征
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
        except Exception as e:
            self.log(f"特征计算失败: {e}")
            return
        
        # 获取特征列
        feature_cols = ['open', 'high', 'low', 'close', 'volume', 'return', 'log_return',
                       'ma20', 'ma50', 'ma_diff', 'rsi', 'macd', 'macd_signal', 
                       'macd_hist', 'volatility', 'volume_ratio']
        
        # 检查特征是否存在
        missing = [col for col in feature_cols if col not in df_feat.columns]
        if missing:
            self.log(f"缺少特征: {missing}")
            return
        
        # 预测
        try:
            X = df_feat[feature_cols].iloc[[-1]]
            prob = self.params.model.predict(X)[0]
        except Exception as e:
            self.log(f"预测失败: {e}")
            return
        
        price_now = self.data_close[0]
        
        # 如果有持仓，检查止盈止损和平仓信号
        if self.position:
            self.check_exit(price_now, prob, current_bar)
        else:
            # 无持仓，检查开仓信号
            self.check_entry(price_now, prob, current_bar)
    
    def check_entry(self, price_now, prob, current_bar):
        """检查开仓信号"""
        
        # 做多信号
        if prob >= self.params.long_threshold:
            size = int((self.broker.getvalue() * MAX_POSITION_RATIO) / price_now)
            if size > 0:
                self.order = self.buy(size=size)
                self.position_type = 'long'
                self.entry_price = price_now
                self.entry_bar = current_bar
                self.highest_price = price_now
                self.trailing_active = False
                self.log(f'做多开仓 @ {price_now:.6f}, 概率={prob:.4f}, 数量={size}')
        
        # 做空信号
        elif prob <= self.params.short_threshold:
            size = int((self.broker.getvalue() * MAX_POSITION_RATIO) / price_now)
            if size > 0:
                self.order = self.sell(size=size)
                self.position_type = 'short'
                self.entry_price = price_now
                self.entry_bar = current_bar
                self.lowest_price = price_now
                self.trailing_active = False
                self.log(f'做空开仓 @ {price_now:.6f}, 概率={prob:.4f}, 数量={size}')
    
    def check_exit(self, price_now, prob, current_bar):
        """检查平仓信号"""
        
        if not self.entry_price:
            return
        
        should_exit = False
        exit_reason = ""
        
        if self.position_type == 'long':
            # 做多持仓的退出逻辑
            ret = (price_now - self.entry_price) / self.entry_price
            
            # 止损
            if ret <= -STOP_LOSS_PCT:
                should_exit = True
                exit_reason = f"止损 (亏损{ret*100:.2f}%)"
            
            # 止盈
            elif ret >= TAKE_PROFIT_PCT:
                should_exit = True
                exit_reason = f"止盈 (盈利{ret*100:.2f}%)"
            
            # 跟踪止盈
            elif USE_TRAILING_STOP:
                if ret >= TRAILING_STOP_ACTIVATION:
                    self.trailing_active = True
                
                if self.trailing_active:
                    if self.highest_price is None or price_now > self.highest_price:
                        self.highest_price = price_now
                    
                    if self.highest_price is not None:
                        drawdown = (self.highest_price - price_now) / self.highest_price
                        if drawdown >= TRAILING_STOP_DISTANCE:
                            should_exit = True
                            exit_reason = f"跟踪止盈 (从最高点回撤{drawdown*100:.2f}%)"
            
            # 时间止损
            elif (current_bar - self.entry_bar) >= TIME_STOP_BARS:
                should_exit = True
                exit_reason = f"时间止损 ({current_bar - self.entry_bar}根K线)"
            
            # 概率反转
            elif prob <= self.params.exit_threshold:
                should_exit = True
                exit_reason = f"概率反转 (prob={prob:.4f})"
        
        elif self.position_type == 'short':
            # 做空持仓的退出逻辑
            ret = (self.entry_price - price_now) / self.entry_price
            
            # 止损
            if ret <= -STOP_LOSS_PCT:
                should_exit = True
                exit_reason = f"止损 (亏损{ret*100:.2f}%)"
            
            # 止盈
            elif ret >= TAKE_PROFIT_PCT:
                should_exit = True
                exit_reason = f"止盈 (盈利{ret*100:.2f}%)"
            
            # 跟踪止盈（做空时跟踪最低价）
            elif USE_TRAILING_STOP:
                if ret >= TRAILING_STOP_ACTIVATION:
                    self.trailing_active = True
                
                if self.trailing_active:
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
            
            # 概率反转
            elif prob >= self.params.exit_threshold:
                should_exit = True
                exit_reason = f"概率反转 (prob={prob:.4f})"
        
        if should_exit:
            self.order = self.close()
            
            # 记录交易结果
            if self.position_type == 'long':
                ret = (price_now - self.entry_price) / self.entry_price
            else:
                ret = (self.entry_price - price_now) / self.entry_price
            
            self.trade_count += 1
            if ret > 0:
                self.win_count += 1
            else:
                self.loss_count += 1
            
            self.log(f'{self.position_type.upper()}平仓 @ {price_now:.6f}, '
                    f'收益={ret*100:+.2f}%, {exit_reason}')
            
            # 重置状态
            self.entry_price = None
            self.entry_bar = None
            self.position_type = None
            self.highest_price = None
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
    print(f"  做多阈值: 0.15")
    print(f"  做空阈值: 0.10")
    print(f"  平仓阈值: 0.12")
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
    cerebro.addstrategy(BidirectionalStrategy, model=model, printlog=False)
    
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
        
        won_total = 0
        lost_total = 0
        try:
            won_total = trades.won.total
        except:
            pass
        try:
            lost_total = trades.lost.total
        except:
            pass
        
        print(f"   盈利交易: {won_total}")
        print(f"   亏损交易: {lost_total}")
        
        if won_total + lost_total > 0:
            win_rate = (won_total / (won_total + lost_total)) * 100
            print(f"   胜率: {win_rate:.1f}%")
            
            try:
                avg_win = trades.won.pnl.average if won_total > 0 else 0
                avg_loss = abs(trades.lost.pnl.average) if lost_total > 0 else 0
                print(f"   平均盈利: ${avg_win:.2f}")
                print(f"   平均亏损: ${avg_loss:.2f}")
                
                if avg_loss > 0:
                    actual_ratio = avg_win / avg_loss
                    print(f"   实际盈亏比: {actual_ratio:.2f}:1")
            except:
                pass
    else:
        print(f"\n⚠️  未产生任何交易")
    
    # 风险指标
    print(f"\n📉 风险指标:")
    if hasattr(drawdown, 'max'):
        print(f"   最大回撤: {drawdown.max.drawdown:.2f}%")
    
    if hasattr(sharpe, 'sharperatio') and sharpe.sharperatio is not None:
        print(f"   夏普比率: {sharpe.sharperatio:.2f}")
    
    # 计算交易频率
    total_days = (df.index[-1] - df.index[0]).days
    if trades.total.total > 0:
        trades_per_day = trades.total.total / total_days
        trades_per_month = trades_per_day * 30
        print(f"\n⏱️  交易频率:")
        print(f"   回测天数: {total_days} 天")
        print(f"   平均每月交易: {trades_per_month:.1f} 次")
    
    print(f"\n⏱️  回测耗时: {(end_time - start_time).total_seconds():.1f} 秒")
    
    print("\n" + "=" * 80)
    
    # 计算胜率
    won_total = 0
    lost_total = 0
    try:
        won_total = trades.won.total
    except:
        pass
    try:
        lost_total = trades.lost.total
    except:
        pass
    
    win_rate = 0
    if won_total + lost_total > 0:
        win_rate = (won_total / (won_total + lost_total)) * 100
    
    return {
        'success': pnl > 0,
        'pnl': pnl,
        'pnl_pct': pnl_pct,
        'total_trades': trades.total.total if trades.total.total > 0 else 0,
        'win_rate': win_rate,
        'max_drawdown': drawdown.max.drawdown if hasattr(drawdown, 'max') else 0,
        'won_total': won_total,
        'lost_total': lost_total,
    }

if __name__ == "__main__":
    try:
        # 加载模型
        print("\n🤖 加载最新模型...")
        model_path = load_latest_model_path()
        if not model_path:
            print("❌ 未找到训练好的模型")
            sys.exit(1)
        model = joblib.load(model_path)
        print(f"✅ 模型加载成功: {model_path}")
        
        # 加载数据
        df = load_real_data()
        if df is None:
            sys.exit(1)
        
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
        else:
            print(f"\n⚠️  回测显示策略亏损 {abs(result['pnl_pct']):.2f}%")
        
        print("\n" + "=" * 80)
        
        sys.exit(0 if result['success'] else 1)
        
    except Exception as e:
        print(f"\n💥 回测失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
