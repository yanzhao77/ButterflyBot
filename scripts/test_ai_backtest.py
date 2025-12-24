#!/usr/bin/env python3
"""
使用AI策略的完整回测脚本
基于test_simple_trade.py的成功经验
"""
import sys
import logging
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from butterfly_bot.data.fetcher import fetch_ohlcv
from butterfly_bot.data.features import add_features
from butterfly_bot.core.broker.backtest import BacktestBroker, ContractType
from butterfly_bot.core.engine.trading_engine import TradingEngine
from butterfly_bot.strategies.ai_signal_core import AISignalCore
from butterfly_bot.core.risk.risk_manager import RiskManager
from butterfly_bot.core.reporter.report_generator import ReportGenerator
from butterfly_bot.config.settings import (
    SYMBOL, EXCHANGE_NAME, TIMEFRAME, INITIAL_CASH,
    MAX_DRAWDOWN
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """主函数"""
    logger.info("="*60)
    logger.info("开始AI策略回测")
    logger.info("="*60)
    
    # 1. 获取数据
    logger.info(f"\n📊 获取数据: {SYMBOL}, {TIMEFRAME}")
    data = fetch_ohlcv(
        symbol=SYMBOL,
        exchange_name=EXCHANGE_NAME,
        timeframe=TIMEFRAME,
        limit=1000
    )
    logger.info(f"获取到 {len(data)} 根K线数据")
    
    # 2. 添加特征
    logger.info("\n🔧 添加技术指标特征...")
    data = add_features(data)
    logger.info(f"特征添加完成，共 {len(data.columns)} 列")
    
    # 3. 初始化组件
    logger.info("\n🏗️ 初始化回测组件...")
    
    # 初始化broker（使用空数据，稍后在循环中更新）
    broker = BacktestBroker(
        initial_balance=INITIAL_CASH,
        leverage=1,
        contract_type=ContractType.SPOT,
        data=data.iloc[:100]  # 初始化时给一些数据
    )
    
    # 初始化AI策略
    strategy = AISignalCore(
        symbol=SYMBOL,
        timeframe=TIMEFRAME
    )
    
    # 初始化风险管理器
    risk_manager = RiskManager(
        initial_balance=INITIAL_CASH,
        max_drawdown_pct=MAX_DRAWDOWN
    )
    
    # 初始化交易引擎
    engine = TradingEngine(
        broker=broker,
        strategy=strategy,
        risk_manager=risk_manager,
        symbol=SYMBOL
    )
    
    logger.info("✅ 所有组件初始化完成")
    
    # 4. 回测循环
    logger.info(f"\n🔄 开始回测循环 (共{len(data)}根K线)...")
    logger.info("="*60)
    
    for index in range(100, len(data)):  # 从第100根开始，确保有足够的历史数据
        # 更新broker的数据到当前时间点
        current_data = data.iloc[:index+1]
        broker.data = current_data
        
        # 获取当前K线数据
        current_bar = current_data.iloc[-1]
        current_price = current_bar['close']
        
        # 生成信号
        signal_info = strategy.get_signal(current_data)
        signal = signal_info.get('signal', 'hold')
        confidence = signal_info.get('confidence', 0.0)
        
        # 执行信号
        if signal != 'hold':
            logger.info(f"\nK线 #{index}/{len(data)}: 价格={current_price:.5f}, 信号={signal}, 置信度={confidence:.3f}")
            engine.execute_signal(
                signal=signal,
                confidence=confidence,
                current_price=current_price
            )
        
        # 每100根K线输出一次进度
        if index % 100 == 0:
            total_value = broker.get_total_value()
            logger.info(f"进度: {index}/{len(data)}, 总资产: {total_value:.2f}")
    
    logger.info("="*60)
    logger.info("🏁 回测循环完成")
    
    # 5. 生成报告
    logger.info("\n📊 生成回测报告...")
    reporter = ReportGenerator(broker, SYMBOL, ContractType.SPOT, 1)
    report = reporter.generate_report()
    
    # 保存报告
    report_path = reporter.save_report(report)
    logger.info(f"✅ 回测完成！报告已保存: {report_path}")
    
    # 打印关键指标
    logger.info("\n" + "="*60)
    logger.info("📊 最终报告")
    logger.info("="*60)
    logger.info(f"\n初始余额: {report['initial_balance']:.2f}")
    logger.info(f"最终余额: {report['final_balance']:.2f}")
    logger.info(f"总交易数: {len(report['trades'])}")
    
    if len(report['trades']) > 0:
        logger.info(f"\n详细交易记录:")
        for i, trade in enumerate(report['trades'], 1):
            logger.info(f"  交易#{i}:")
            logger.info(f"    买入价: {trade['entry_price']:.5f}")
            logger.info(f"    卖出价: {trade['exit_price']:.5f}")
            logger.info(f"    数量: {trade['size']:.2f}")
            logger.info(f"    盈亏: {trade['pnl']:.2f} ({trade['pnl_pct']:.2f}%)")
    else:
        logger.info("\n⚠️ 没有产生任何交易")
    
    logger.info("="*60)
    
    # 验证结果
    logger.info("\n" + "="*60)
    logger.info("🔍 验证结果")
    logger.info("="*60)
    
    if len(report['trades']) > 0:
        logger.info(f"✅ 测试通过：产生了{len(report['trades'])}笔完整交易")
        profit = report['final_balance'] - report['initial_balance']
        profit_pct = (profit / report['initial_balance']) * 100
        logger.info(f"初始余额: {report['initial_balance']:.2f}")
        logger.info(f"最终余额: {report['final_balance']:.2f}")
        logger.info(f"盈亏: {profit:.2f} ({profit_pct:.2f}%)")
    else:
        logger.info("❌ 测试失败：没有产生任何交易")
        logger.info("可能的原因:")
        logger.info("1. AI策略从不生成卖出信号")
        logger.info("2. 买入和卖出阈值设置不合理")
        logger.info("3. 其他过滤条件过于严格")

if __name__ == "__main__":
    main()
