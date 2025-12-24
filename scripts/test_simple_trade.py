"""
最简化的1买1卖回测测试
用于验证整个交易流程是否正常工作
"""
import logging
import sys
from datetime import datetime

sys.path.insert(0, ".")

from butterfly_bot.core.engine.trading_engine import TradingEngine
from butterfly_bot.core.broker.backtest import BacktestBroker
from butterfly_bot.core.risk.risk_manager import RiskManager
from butterfly_bot.core.reporter.report_generator import ReportGenerator
from butterfly_bot.data.fetcher import fetch_historical_data
from butterfly_bot.core.broker.base import ContractType
from butterfly_bot.config.settings import RISK_MANAGEMENT_CONFIG, BACKTEST_CONFIG

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def simple_strategy(data):
    """
    最简单的策略：
    - 第10根K线：买入
    - 第20根K线：卖出
    """
    current_bar = len(data)
    
    if current_bar == 10:
        return {
            "signal": "buy",
            "confidence": 0.8,
            "reason": "测试买入",
            "timestamp": data.index[-1]
        }
    elif current_bar == 20:
        return {
            "signal": "sell",
            "confidence": 0.8,
            "reason": "测试卖出",
            "timestamp": data.index[-1]
        }
    else:
        return {
            "signal": "hold",
            "confidence": 0.5,
            "reason": "等待",
            "timestamp": data.index[-1]
        }

def run_simple_backtest():
    logger.info('====== 开始简单回测: 1买1卖测试 ======')
    
    # 1. 加载数据
    symbol = "DOGE/USDT"
    data = fetch_historical_data(symbol, BACKTEST_CONFIG["start_date"], BACKTEST_CONFIG["end_date"])
    if data.empty:
        logger.error("数据加载失败")
        return
    
    logger.info(f"加载数据: {len(data)}根K线")
    
    # 2. 初始化核心组件
    initial_balance = 1000.0
    leverage = 1
    contract_type = ContractType.SPOT
    
    broker = BacktestBroker(initial_balance, leverage, contract_type, data)
    risk_manager = RiskManager(initial_balance, **RISK_MANAGEMENT_CONFIG)
    engine = TradingEngine(broker, risk_manager, symbol, simple_strategy)
    
    # 3. 运行回测
    engine.start()
    
    for i, (index, row) in enumerate(data.iterrows()):
        current_price = row["close"]
        
        # 更新broker的数据（截止到当前时间点）
        broker.data = data.loc[:index]
        
        # 获取信号
        signal_data = simple_strategy(data.loc[:index])
        signal = signal_data["signal"]
        confidence = signal_data["confidence"]
        
        logger.info(f"\n{'='*60}")
        logger.info(f"K线 #{i+1}/{len(data)}: 价格={current_price:.5f}, 信号={signal}")
        logger.info(f"{'='*60}")
        
        # 执行信号
        success = engine.execute_signal(
            signal,
            confidence,
            current_price,
            stop_loss_pct=0.03,
            take_profit_pct=0.06
        )
        
        if success and signal in ["buy", "sell"]:
            logger.info(f"✅ 信号执行成功: {signal}")
            logger.info(f"   当前余额: {broker.balance:.2f}")
            logger.info(f"   当前持仓: {broker.position}")
            logger.info(f"   总资产: {broker.get_total_value(symbol):.2f}")
            logger.info(f"   已完成交易数: {len(broker.trades)}")
    
    engine.stop()
    
    # 4. 生成报告
    reporter = ReportGenerator(broker, risk_manager, engine)
    report = reporter.generate_report()
    
    logger.info(f"\n{'='*60}")
    logger.info("📊 最终报告")
    logger.info(f"{'='*60}")
    reporter.print_report(report)
    
    logger.info(f"\n详细交易记录:")
    for i, trade in enumerate(report["trades"], 1):
        logger.info(f"  交易#{i}:")
        logger.info(f"    买入价: {trade['entry_price']:.5f}")
        logger.info(f"    卖出价: {trade['exit_price']:.5f}")
        logger.info(f"    数量: {trade['size']:.2f}")
        logger.info(f"    盈亏: {trade['pnl']:.2f} ({trade['pnl_pct']:.2f}%)")
    
    # 保存报告
    report_path = f'reports/backtest/simple_test_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    reporter.save_report(report, report_path)
    logger.info(f'\n✅ 回测完成！报告已保存: {report_path}')
    
    return report

if __name__ == "__main__":
    report = run_simple_backtest()
    
    # 验证结果
    print(f"\n{'='*60}")
    print("🔍 验证结果")
    print(f"{'='*60}")
    
    if len(report["trades"]) == 1:
        print("✅ 测试通过：产生了1笔完整交易（1买1卖）")
    elif len(report["trades"]) == 0:
        print("❌ 测试失败：没有产生任何交易")
    else:
        print(f"⚠️ 测试异常：产生了{len(report['trades'])}笔交易")
    
    print(f"\n初始余额: {report['initial_balance']:.2f}")
    print(f"最终余额: {report['final_balance']:.2f}")
    print(f"盈亏: {report['final_balance'] - report['initial_balance']:.2f}")
