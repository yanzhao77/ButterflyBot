#!/usr/bin/env python3
"""
批量回测脚本 - 支持多维度回测分析
"""

import sys
import os
import argparse
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import logging

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from butterfly_bot.config.settings import *
from butterfly_bot.data.fetcher import fetch_historical_data
from butterfly_bot.strategies.ai_signal_core import AISignalCore
from butterfly_bot.core.broker.backtest import BacktestBroker
from butterfly_bot.core.broker.base import ContractType
from butterfly_bot.core.risk.risk_manager import RiskManager
from butterfly_bot.core.engine.trading_engine import TradingEngine
from butterfly_bot.core.reporter.report_generator import ReportGenerator
from butterfly_bot.analysis.metrics import PerformanceMetrics

# 配置日志
logging.basicConfig(
    level=logging.WARNING,  # 减少日志输出
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # 只显示batch_backtest的INFO日志


class BatchBacktest:
    """批量回测管理器"""
    
    def __init__(self, output_dir: str = "reports/batch_backtest"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = []
    
    def run_single_backtest(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        initial_balance: float = 1000.0,
        contract_type: ContractType = ContractType.SPOT,
        leverage: int = 1,
    ) -> dict:
        """运行单次回测
        
        Args:
            symbol: 交易对
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            initial_balance: 初始资金
            contract_type: 合约类型
            leverage: 杠杆倍数
            
        Returns:
            回测结果字典
        """
        logger.info(f"开始回测: {symbol} {start_date} ~ {end_date}, {contract_type.name}, {leverage}x")
        
        try:
            # 获取数据
            data = fetch_historical_data(symbol, start_date, end_date)
            
            if data.empty or len(data) < 100:
                logger.warning(f"数据量不足: {len(data)}条")
                return None
            
            logger.info(f"获取数据: {len(data)}条")
            
            # 初始化组件
            broker = BacktestBroker(initial_balance, leverage, contract_type, data)
            risk_manager = RiskManager(initial_balance, **RISK_MANAGEMENT_CONFIG)
            strategy = AISignalCore(**AI_SIGNAL_CONFIG)
            engine = TradingEngine(broker, risk_manager, symbol, strategy.get_signal)
            
            # 运行回测
            engine.start()
            equity_history = [initial_balance]
            timestamps = [data.index[0]]
            
            for index, row in data.iterrows():
                current_price = row["close"]
                
                # 获取信号
                signal_data = strategy.get_signal(data.loc[:index])
                signal = signal_data["signal"]
                confidence = signal_data["confidence"]
                
                # 执行信号
                engine.execute_signal(
                    signal,
                    confidence,
                    current_price,
                    stop_loss_pct=RISK_MANAGEMENT_CONFIG.get("stop_loss_pct", 0.03),
                    take_profit_pct=RISK_MANAGEMENT_CONFIG.get("take_profit_pct", 0.08)
                )
                
                # 记录权益
                account_info = broker.get_account_info()
                position = broker.get_position(symbol)
                
                # 计算当前权益
                current_equity = account_info["totalWalletBalance"]
                if position["size"] > 0:
                    unrealized_pnl = (current_price - position["entry_price"]) * position["size"]
                    current_equity += unrealized_pnl
                
                equity_history.append(current_equity)
                timestamps.append(index)
            
            engine.stop()
            
            # 获取交易记录
            trades = broker.trades
            
            # 构建权益曲线
            equity_curve = pd.Series(equity_history, index=timestamps)
            
            # 计算回测天数
            period_days = (pd.Timestamp(end_date) - pd.Timestamp(start_date)).days
            
            # 计算指标
            metrics_calculator = PerformanceMetrics(initial_balance=initial_balance)
            metrics = metrics_calculator.calculate_all_metrics(
                equity_curve=equity_curve,
                trades=trades,
                period_days=period_days
            )
            
            # 构建结果
            result = {
                'config': {
                    'symbol': symbol,
                    'start_date': start_date,
                    'end_date': end_date,
                    'period_days': period_days,
                    'initial_balance': initial_balance,
                    'contract_type': contract_type.name,
                    'leverage': leverage,
                },
                'metrics': metrics,
                'account_info': broker.get_account_info(),
                'data_points': len(data),
                'trades_count': len(trades),
            }
            
            logger.info(f"✅ 回测完成: 总收益={metrics['total_return_pct']:.2f}%, 最大回撤={metrics['max_drawdown_pct']:.2f}%, 夏普比率={metrics['sharpe_ratio']:.3f}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ 回测失败: {e}", exc_info=True)
            return None
    
    def run_time_dimension(
        self,
        symbol: str = "DOGE/USDT",
        periods: list = None
    ):
        """时间维度回测
        
        Args:
            symbol: 交易对
            periods: 时间周期列表，如 ['1M', '3M', '6M']
        """
        if periods is None:
            periods = ['1M', '3M', '6M']
        
        logger.info(f"\n{'='*60}")
        logger.info(f"时间维度回测: {symbol}")
        logger.info(f"{'='*60}\n")
        
        end_date = datetime.now()
        
        for period in periods:
            # 解析周期
            if period.endswith('M'):
                months = int(period[:-1])
                start_date = end_date - timedelta(days=months * 30)
            elif period.endswith('Y'):
                years = int(period[:-1])
                start_date = end_date - timedelta(days=years * 365)
            else:
                logger.warning(f"未知周期格式: {period}")
                continue
            
            result = self.run_single_backtest(
                symbol=symbol,
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d'),
            )
            
            if result:
                result['dimension'] = 'time'
                result['period'] = period
                self.results.append(result)
                self.save_result(result, f"time_{period}_{symbol.replace('/', '_')}")
    
    def run_symbol_dimension(
        self,
        symbols: list = None,
        period: str = "3M"
    ):
        """币种维度回测
        
        Args:
            symbols: 交易对列表
            period: 回测周期
        """
        if symbols is None:
            symbols = ['DOGE/USDT', 'BTC/USDT', 'ETH/USDT']
        
        logger.info(f"\n{'='*60}")
        logger.info(f"币种维度回测: {symbols}")
        logger.info(f"{'='*60}\n")
        
        # 计算日期范围
        end_date = datetime.now()
        if period.endswith('M'):
            months = int(period[:-1])
            start_date = end_date - timedelta(days=months * 30)
        else:
            start_date = end_date - timedelta(days=90)
        
        for symbol in symbols:
            result = self.run_single_backtest(
                symbol=symbol,
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d'),
            )
            
            if result:
                result['dimension'] = 'symbol'
                result['period'] = period
                self.results.append(result)
                self.save_result(result, f"symbol_{symbol.replace('/', '_')}")
    
    def run_leverage_dimension(
        self,
        symbol: str = "DOGE/USDT",
        leverages: list = None,
        period: str = "3M"
    ):
        """杠杆维度回测
        
        Args:
            symbol: 交易对
            leverages: 杠杆倍数列表
            period: 回测周期
        """
        if leverages is None:
            leverages = [1, 3, 5]
        
        logger.info(f"\n{'='*60}")
        logger.info(f"杠杆维度回测: {symbol} {leverages}x")
        logger.info(f"{'='*60}\n")
        
        # 计算日期范围
        end_date = datetime.now()
        if period.endswith('M'):
            months = int(period[:-1])
            start_date = end_date - timedelta(days=months * 30)
        else:
            start_date = end_date - timedelta(days=90)
        
        for leverage in leverages:
            contract_type = ContractType.SPOT if leverage == 1 else ContractType.USDT_M
            
            result = self.run_single_backtest(
                symbol=symbol,
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d'),
                contract_type=contract_type,
                leverage=leverage,
            )
            
            if result:
                result['dimension'] = 'leverage'
                result['period'] = period
                self.results.append(result)
                self.save_result(result, f"leverage_{leverage}x_{symbol.replace('/', '_')}")
    
    def save_result(self, result: dict, filename: str):
        """保存单个回测结果"""
        filepath = self.output_dir / f"{filename}.json"
        with open(filepath, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        logger.info(f"💾 结果已保存: {filepath}")
    
    def save_summary(self):
        """保存汇总报告"""
        if not self.results:
            logger.warning("没有回测结果")
            return
        
        summary_file = self.output_dir / "summary.json"
        with open(summary_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        logger.info(f"\n💾 汇总报告已保存: {summary_file}")
        
        # 生成对比表格
        self.generate_comparison_table()
    
    def generate_comparison_table(self):
        """生成对比表格"""
        if not self.results:
            return
        
        rows = []
        for result in self.results:
            config = result['config']
            metrics = result['metrics']
            
            row = {
                '维度': result.get('dimension', 'unknown'),
                '交易对': config['symbol'],
                '周期': result.get('period', f"{config.get('period_days', 'N/A')}天"),
                '杠杆': f"{config['leverage']}x",
                '总收益(%)': f"{metrics['total_return_pct']:.2f}",
                '年化收益(%)': f"{metrics['annualized_return_pct']:.2f}",
                '最大回撤(%)': f"{metrics['max_drawdown_pct']:.2f}",
                '夏普比率': f"{metrics['sharpe_ratio']:.3f}",
                '胜率(%)': f"{metrics.get('win_rate_pct', 0):.2f}",
                '交易次数': metrics.get('total_trades', 0),
                '盈亏比': f"{metrics.get('profit_loss_ratio', 0):.2f}",
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # 保存为CSV
        csv_file = self.output_dir / "comparison.csv"
        df.to_csv(csv_file, index=False, encoding='utf-8-sig')
        logger.info(f"💾 对比表格已保存: {csv_file}")
        
        # 打印表格
        print("\n" + "=" * 140)
        print("📊 回测结果对比")
        print("=" * 140)
        print(df.to_string(index=False))
        print("=" * 140)
        print()


def main():
    parser = argparse.ArgumentParser(description='批量回测工具')
    parser.add_argument('--dimension', type=str, default='time',
                        choices=['time', 'symbol', 'leverage', 'all'],
                        help='回测维度')
    parser.add_argument('--symbol', type=str, default='DOGE/USDT',
                        help='交易对')
    parser.add_argument('--period', type=str, default='3M',
                        help='回测周期')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("🚀 ButterflyBot 批量回测系统")
    print("=" * 60)
    print(f"维度: {args.dimension}")
    print(f"交易对: {args.symbol}")
    print(f"周期: {args.period}")
    print("=" * 60 + "\n")
    
    # 创建批量回测管理器
    batch = BatchBacktest()
    
    # 根据维度执行回测
    if args.dimension == 'time' or args.dimension == 'all':
        batch.run_time_dimension(
            symbol=args.symbol,
            periods=['1M', '3M', '6M']
        )
    
    if args.dimension == 'symbol' or args.dimension == 'all':
        batch.run_symbol_dimension(
            symbols=['DOGE/USDT', 'BTC/USDT', 'ETH/USDT'],
            period=args.period
        )
    
    if args.dimension == 'leverage' or args.dimension == 'all':
        batch.run_leverage_dimension(
            symbol=args.symbol,
            leverages=[1, 3, 5],
            period=args.period
        )
    
    # 保存汇总报告
    batch.save_summary()
    
    print("\n" + "=" * 60)
    print("✅ 批量回测完成！")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
