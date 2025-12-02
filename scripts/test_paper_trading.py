"""
实时模拟交易测试脚本
使用PaperBroker进行纸上交易模拟
"""

import logging
import sys
import time
from datetime import datetime

sys.path.insert(0, ".")

from butterfly_bot.core.broker.paper_broker import PaperBroker
from butterfly_bot.core.broker.base import ContractType, OrderSide, OrderType
from butterfly_bot.core.risk.risk_manager import RiskManager
from butterfly_bot.core.engine.trading_engine import TradingEngine
from butterfly_bot.data.realtime_fetcher import RealtimeFetcher
from butterfly_bot.strategies.ai_signal_core import AISignalCore
from butterfly_bot.config.settings import (
    AI_SIGNAL_CONFIG,
    RISK_MANAGEMENT_CONFIG,
    SYMBOL,
    TIMEFRAME
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_paper_trading(
    symbol: str = SYMBOL,
    initial_balance: float = 1000.0,
    leverage: int = 1,
    contract_type: ContractType = ContractType.SPOT,
    duration_minutes: int = 5,
    check_interval_seconds: int = 60
):
    """运行纸上交易模拟
    
    Args:
        symbol: 交易对
        initial_balance: 初始余额
        leverage: 杠杆倍数
        contract_type: 合约类型
        duration_minutes: 运行时长（分钟）
        check_interval_seconds: 检查间隔（秒）
    """
    logger.info(f"====== 开始实时模拟交易测试 ======")
    logger.info(f"交易对: {symbol}")
    logger.info(f"初始余额: {initial_balance}")
    logger.info(f"杠杆: {leverage}x")
    logger.info(f"合约类型: {contract_type.name}")
    logger.info(f"运行时长: {duration_minutes}分钟")
    logger.info(f"检查间隔: {check_interval_seconds}秒")
    
    # 1. 初始化组件
    logger.info("初始化交易组件...")
    
    # 初始化实时数据获取器
    fetcher = RealtimeFetcher()
    
    # 初始化PaperBroker
    broker = PaperBroker(
        initial_balance=initial_balance,
        leverage=leverage,
        contract_type=contract_type
    )
    
    # 初始化风险管理器
    risk_manager = RiskManager(initial_balance, **RISK_MANAGEMENT_CONFIG)
    
    # 初始化AI策略
    try:
        strategy = AISignalCore(symbol=symbol, timeframe=TIMEFRAME, **AI_SIGNAL_CONFIG)
        logger.info("AI策略初始化成功")
    except Exception as e:
        logger.error(f"AI策略初始化失败: {e}")
        logger.info("使用简单策略代替...")
        strategy = None
    
    # 初始化交易引擎
    if strategy:
        engine = TradingEngine(broker, risk_manager, symbol, strategy.get_signal)
    else:
        # 如果AI策略失败，使用简单的测试策略
        def simple_test_signal(df):
            return {"signal": "hold", "confidence": 0.0, "reason": "测试模式"}
        engine = TradingEngine(broker, risk_manager, symbol, simple_test_signal)
    
    # 2. 启动交易引擎
    engine.start()
    logger.info("交易引擎已启动")
    
    # 3. 运行交易循环
    start_time = time.time()
    end_time = start_time + duration_minutes * 60
    iteration = 0
    
    try:
        while time.time() < end_time:
            iteration += 1
            logger.info(f"\n{'='*60}")
            logger.info(f"迭代 #{iteration} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            # 获取当前价格
            current_price = fetcher.get_current_price(symbol)
            if current_price is None:
                logger.warning("无法获取当前价格，跳过本次迭代")
                time.sleep(check_interval_seconds)
                continue
            
            logger.info(f"当前价格: {current_price}")
            
            # 获取最近的K线数据
            df = fetcher.get_recent_klines(symbol, TIMEFRAME, limit=200)
            if df.empty:
                logger.warning("无法获取K线数据，跳过本次迭代")
                time.sleep(check_interval_seconds)
                continue
            
            logger.info(f"获取到 {len(df)} 根K线数据")
            
            # 更新未实现盈亏
            broker.update_unrealized_pnl(symbol, current_price)
            
            # 获取交易信号
            if strategy:
                try:
                    signal_data = strategy.get_signal(df)
                    signal = signal_data["signal"]
                    confidence = signal_data["confidence"]
                    reason = signal_data.get("reason", "")
                    
                    logger.info(f"交易信号: {signal} (置信度: {confidence:.3f})")
                    logger.info(f"信号原因: {reason}")
                    
                    # 执行信号
                    engine.execute_signal(
                        signal,
                        confidence,
                        current_price,
                        stop_loss_pct=RISK_MANAGEMENT_CONFIG["stop_loss_pct"],
                        take_profit_pct=RISK_MANAGEMENT_CONFIG["take_profit_pct"]
                    )
                except Exception as e:
                    logger.error(f"信号生成或执行失败: {e}")
            
            # 显示账户信息
            account_info = broker.get_account_info()
            position = broker.get_position(symbol)
            
            logger.info(f"\n账户信息:")
            logger.info(f"  余额: {account_info['balance']:.2f} USDT")
            logger.info(f"  权益: {account_info['equity']:.2f} USDT")
            logger.info(f"  未实现盈亏: {account_info['total_unrealized_pnl']:.2f} USDT")
            logger.info(f"  总交易次数: {account_info['total_trades']}")
            
            logger.info(f"\n持仓信息:")
            logger.info(f"  持仓数量: {position['size']}")
            logger.info(f"  开仓均价: {position['entry_price']:.4f}")
            logger.info(f"  未实现盈亏: {position['unrealized_pnl']:.2f} USDT")
            
            # 等待下一次检查
            logger.info(f"\n等待 {check_interval_seconds} 秒...")
            time.sleep(check_interval_seconds)
            
    except KeyboardInterrupt:
        logger.info("\n用户中断，停止交易...")
    except Exception as e:
        logger.error(f"交易循环出错: {e}")
    finally:
        # 4. 停止交易引擎
        engine.stop()
        logger.info("交易引擎已停止")
        
        # 5. 生成最终报告
        logger.info(f"\n{'='*60}")
        logger.info("====== 最终交易报告 ======")
        
        account_info = broker.get_account_info()
        position = broker.get_position(symbol)
        
        logger.info(f"\n账户摘要:")
        logger.info(f"  初始余额: {account_info['initial_balance']:.2f} USDT")
        logger.info(f"  最终余额: {account_info['balance']:.2f} USDT")
        logger.info(f"  最终权益: {account_info['equity']:.2f} USDT")
        logger.info(f"  总盈亏: {account_info['equity'] - account_info['initial_balance']:.2f} USDT")
        logger.info(f"  收益率: {(account_info['equity'] / account_info['initial_balance'] - 1) * 100:.2f}%")
        logger.info(f"  总交易次数: {account_info['total_trades']}")
        
        logger.info(f"\n当前持仓:")
        logger.info(f"  持仓数量: {position['size']}")
        if position['size'] != 0:
            logger.info(f"  开仓均价: {position['entry_price']:.4f}")
            logger.info(f"  未实现盈亏: {position['unrealized_pnl']:.2f} USDT")
        
        if broker.trades:
            logger.info(f"\n交易历史:")
            for i, trade in enumerate(broker.trades, 1):
                logger.info(f"  交易 #{i}:")
                logger.info(f"    交易对: {trade['symbol']}")
                logger.info(f"    数量: {trade['amount']}")
                logger.info(f"    开仓价: {trade['entry_price']:.4f}")
                logger.info(f"    平仓价: {trade['exit_price']:.4f}")
                logger.info(f"    盈亏: {trade['pnl']:.2f} USDT")
                logger.info(f"    时间: {trade['timestamp']}")
        
        logger.info(f"\n{'='*60}")
        logger.info("====== 测试完成 ======")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="实时模拟交易测试")
    parser.add_argument("--symbol", type=str, default=SYMBOL, help="交易对")
    parser.add_argument("--balance", type=float, default=1000.0, help="初始余额")
    parser.add_argument("--leverage", type=int, default=1, help="杠杆倍数")
    parser.add_argument("--duration", type=int, default=5, help="运行时长（分钟）")
    parser.add_argument("--interval", type=int, default=60, help="检查间隔（秒）")
    parser.add_argument("--contract", type=str, default="SPOT", 
                       choices=["SPOT", "USDT_M", "COIN_M"], help="合约类型")
    
    args = parser.parse_args()
    
    # 转换合约类型
    contract_type_map = {
        "SPOT": ContractType.SPOT,
        "USDT_M": ContractType.USDT_M,
        "COIN_M": ContractType.COIN_M
    }
    
    run_paper_trading(
        symbol=args.symbol,
        initial_balance=args.balance,
        leverage=args.leverage,
        contract_type=contract_type_map[args.contract],
        duration_minutes=args.duration,
        check_interval_seconds=args.interval
    )
