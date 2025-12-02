import logging
import sys
from datetime import datetime

sys.path.insert(0, ".")

from butterfly_bot.core.engine.trading_engine import TradingEngine
from butterfly_bot.core.broker.backtest import BacktestBroker
from butterfly_bot.core.risk.risk_manager import RiskManager
from butterfly_bot.core.reporter.report_generator import ReportGenerator
from butterfly_bot.data.fetcher import fetch_historical_data
from butterfly_bot.strategies.ai_signal_core import AISignalCore
from butterfly_bot.config.settings import (
    AI_SIGNAL_CONFIG,
    RISK_MANAGEMENT_CONFIG,
    BACKTEST_CONFIG
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def run_backtest(symbol, initial_balance, leverage, contract_type):
    logging.info(f'====== 开始回测: {symbol}, {contract_type.name}, {leverage}x ======')

    # 1. 加载数据
    data = fetch_historical_data(symbol, BACKTEST_CONFIG["start_date"], BACKTEST_CONFIG["end_date"])
    if data.empty:
        logging.error("数据加载失败")
        return

    # 2. 初始化核心组件
    broker = BacktestBroker(initial_balance, leverage, contract_type, data)
    risk_manager = RiskManager(initial_balance, **RISK_MANAGEMENT_CONFIG)
    strategy = AISignalCore(**AI_SIGNAL_CONFIG)
    engine = TradingEngine(broker, risk_manager, symbol, strategy.get_signal)

    # 3. 运行回测
    engine.start()
    for index, row in data.iterrows():
        current_price = row["close"]
        timestamp = index
        
        # 获取信号
        signal_data = strategy.get_signal(data.loc[:index])
        signal = signal_data["signal"]
        confidence = signal_data["confidence"]
        
        # 执行信号
        engine.execute_signal(
            signal,
            confidence,
            current_price,
            stop_loss_pct=RISK_MANAGEMENT_CONFIG["stop_loss_pct"],
            take_profit_pct=RISK_MANAGEMENT_CONFIG["take_profit_pct"]
        )
    engine.stop()

    # 4. 生成报告
    reporter = ReportGenerator(broker, risk_manager, engine)
    report = reporter.generate_report()
    reporter.print_report(report)
    
    report_path = f'reports/backtest/backtest_{contract_type.name}_leverage{leverage}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    reporter.save_report(report, report_path)
    logging.info(f'\n✅ 回测完成！报告已保存: {report_path}')

if __name__ == "__main__":
    from butterfly_bot.core.broker.base import ContractType
    # 运行现货回测
    run_backtest(
        symbol="DOGE/USDT",
        initial_balance=1000.0,
        leverage=1,
        contract_type=ContractType.SPOT
    )

    # 运行USDT-M永续合约回测 (3x杠杆)
    run_backtest(
        symbol="DOGE/USDT",
        initial_balance=1000.0,
        leverage=3,
        contract_type=ContractType.USDT_M
    )

    # 运行USDT-M永续合约回测 (5x杠杆)
    run_backtest(
        symbol="DOGE/USDT",
        initial_balance=1000.0,
        leverage=5,
        contract_type=ContractType.USDT_M
    )
