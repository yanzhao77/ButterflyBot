"""
交易引擎 - 统一的信号处理和订单执行逻辑
"""

import logging
from typing import Callable, Dict, Optional
from ..broker.base import OrderSide, OrderType

logger = logging.getLogger(__name__)


class TradingEngine:
    """交易引擎，负责信号处理和订单执行"""
    
    def __init__(self, broker, risk_manager, symbol: str, get_signal_func: Callable):
        """初始化交易引擎
        
        Args:
            broker: Broker实例（BacktestBroker/PaperBroker/LiveBroker）
            risk_manager: 风险管理器实例
            symbol: 交易对
            get_signal_func: 信号生成函数
        """
        self.broker = broker
        self.risk_manager = risk_manager
        self.symbol = symbol
        self.get_signal = get_signal_func
        self.is_running = False
        
        logger.info(f"交易引擎初始化: 交易对={symbol}")

    def start(self):
        """启动交易引擎"""
        self.is_running = True
        logger.info("Trading engine started.")

    def stop(self):
        """停止交易引擎"""
        self.is_running = False
        logger.info("Trading engine stopped.")

    def execute_signal(
        self,
        signal: str,
        confidence: float,
        current_price: float,
        stop_loss_pct: Optional[float] = None,
        take_profit_pct: Optional[float] = None
    ) -> bool:
        """执行交易信号
        
        Args:
            signal: 交易信号（BUY/SELL/SHORT/COVER/HOLD）
            confidence: 信号置信度
            current_price: 当前价格
            stop_loss_pct: 止损百分比
            take_profit_pct: 止盈百分比
            
        Returns:
            是否成功执行
        """
        if not self.is_running:
            logger.warning("交易引擎未启动，忽略信号")
            return False
        
        try:
            # 更新风险管理器的余额信息
            current_balance = self.broker.get_balance()
            self.risk_manager.update_balance(current_balance)
            
            # 检查是否可以交易
            can_trade, reason = self.risk_manager.can_trade()
            if not can_trade:
                logger.warning(f"风险管理器禁止交易: {reason}")
                return False
            
            # 获取当前持仓
            position = self.broker.get_position(self.symbol)
            current_position_size = position.get('size', 0)
            
            # 标准化信号
            signal = signal.upper()
            
            # 处理不同的信号
            if signal == "BUY":
                return self._handle_buy_signal(
                    current_position_size,
                    current_price,
                    confidence,
                    stop_loss_pct,
                    take_profit_pct
                )
            elif signal == "SELL":
                return self._handle_sell_signal(
                    current_position_size,
                    current_price
                )
            elif signal == "SHORT":
                return self._handle_short_signal(
                    current_position_size,
                    current_price,
                    confidence,
                    stop_loss_pct,
                    take_profit_pct
                )
            elif signal == "COVER":
                return self._handle_cover_signal(
                    current_position_size,
                    current_price
                )
            elif signal == "HOLD":
                logger.debug(f"信号为HOLD，不执行任何操作")
                return True
            else:
                logger.warning(f"未知信号类型: {signal}")
                return False
                
        except Exception as e:
            logger.error(f"执行信号时发生错误: {e}", exc_info=True)
            return False
    
    def _handle_buy_signal(
        self,
        current_position_size: float,
        current_price: float,
        confidence: float,
        stop_loss_pct: Optional[float],
        take_profit_pct: Optional[float]
    ) -> bool:
        """处理买入信号"""
        if current_position_size > 0:
            logger.debug("已有多头持仓，忽略买入信号")
            return False
        
        if current_position_size < 0:
            logger.info("当前有空头持仓，先平仓")
            self.broker.close_position(self.symbol, current_price=current_price)
        
        # 计算开仓数量
        try:
            current_balance = self.broker.get_balance()
            amount = self.risk_manager.calculate_position_size(current_balance, current_price)
            
            if amount <= 0:
                logger.warning(f"计算的开仓数量无效: {amount}")
                return False
            
            # 下单
            logger.info(f"执行买入: 数量={amount}, 价格={current_price}, 置信度={confidence:.3f}")
            order = self.broker.place_order(
                self.symbol,
                OrderSide.BUY,
                amount,
                OrderType.MARKET,
                price=current_price
            )
            
            if order:
                logger.info(f"买入成功: 订单ID={order.get('order_id', 'N/A')}")
                return True
            else:
                logger.error("买入失败: 订单返回为空")
                return False
                
        except Exception as e:
            logger.error(f"买入操作失败: {e}", exc_info=True)
            return False
    
    def _handle_sell_signal(self, current_position_size: float, current_price: float) -> bool:
        """处理卖出信号"""
        if current_position_size <= 0:
            logger.debug("没有多头持仓，忽略卖出信号")
            return False
        
        try:
            logger.info(f"执行卖出平仓: 持仓数量={current_position_size}, 价格={current_price}")
            success = self.broker.close_position(self.symbol, current_price=current_price)
            
            if success:
                logger.info("卖出平仓成功")
                return True
            else:
                logger.error("卖出平仓失败")
                return False
                
        except Exception as e:
            logger.error(f"卖出操作失败: {e}", exc_info=True)
            return False
    
    def _handle_short_signal(
        self,
        current_position_size: float,
        current_price: float,
        confidence: float,
        stop_loss_pct: Optional[float],
        take_profit_pct: Optional[float]
    ) -> bool:
        """处理做空信号"""
        if current_position_size < 0:
            logger.debug("已有空头持仓，忽略做空信号")
            return False
        
        if current_position_size > 0:
            logger.info("当前有多头持仓，先平仓")
            self.broker.close_position(self.symbol, current_price=current_price)
        
        # 计算开仓数量
        try:
            current_balance = self.broker.get_balance()
            amount = self.risk_manager.calculate_position_size(current_balance, current_price)
            
            if amount <= 0:
                logger.warning(f"计算的开仓数量无效: {amount}")
                return False
            
            # 下单
            logger.info(f"执行做空: 数量={amount}, 价格={current_price}, 置信度={confidence:.3f}")
            order = self.broker.place_order(
                self.symbol,
                OrderSide.SELL,
                amount,
                OrderType.MARKET,
                price=current_price
            )
            
            if order:
                logger.info(f"做空成功: 订单ID={order.get('order_id', 'N/A')}")
                return True
            else:
                logger.error("做空失败: 订单返回为空")
                return False
                
        except Exception as e:
            logger.error(f"做空操作失败: {e}", exc_info=True)
            return False
    
    def _handle_cover_signal(self, current_position_size: float, current_price: float) -> bool:
        """处理平空信号"""
        if current_position_size >= 0:
            logger.debug("没有空头持仓，忽略平空信号")
            return False
        
        try:
            logger.info(f"执行平空: 持仓数量={current_position_size}, 价格={current_price}")
            success = self.broker.close_position(self.symbol, current_price=current_price)
            
            if success:
                logger.info("平空成功")
                return True
            else:
                logger.error("平空失败")
                return False
                
        except Exception as e:
            logger.error(f"平空操作失败: {e}", exc_info=True)
            return False
