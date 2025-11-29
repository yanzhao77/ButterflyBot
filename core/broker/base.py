# core/broker/base.py
"""
Broker抽象基类

统一回测、模拟盘、实盘的接口，方便切换和测试
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Optional
import pandas as pd


class OrderSide(Enum):
    """订单方向"""
    BUY = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    """订单类型"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"


class ContractType(Enum):
    """合约类型"""
    SPOT = "SPOT"              # 现货
    USDT_M = "USDT_M"          # USDT本位永续
    COIN_M = "COIN_M"          # 币本位永续


class PositionSide(Enum):
    """持仓方向（仅永续合约）"""
    LONG = "LONG"    # 多头
    SHORT = "SHORT"  # 空头
    BOTH = "BOTH"    # 双向持仓


class BaseBroker(ABC):
    """经纪商抽象基类
    
    所有Broker实现必须继承此类并实现所有抽象方法
    """
    
    def __init__(
        self,
        initial_balance: float = 1000.0,
        contract_type: ContractType = ContractType.SPOT
    ):
        self.initial_balance = initial_balance
        self.contract_type = contract_type
        self.leverage = 1
    
    @abstractmethod
    def get_balance(self, asset: str = "USDT") -> float:
        """获取余额
        
        Args:
            asset: 资产名称，默认USDT
            
        Returns:
            可用余额
        """
        pass
    
    @abstractmethod
    def get_position(self, symbol: str) -> Dict:
        """获取持仓信息
        
        Args:
            symbol: 交易对，如 "DOGE/USDT"
            
        Returns:
            {
                'size': float,           # 持仓数量（正=多，负=空，0=无持仓）
                'entry_price': float,    # 开仓均价
                'leverage': int,         # 杠杆倍数
                'unrealized_pnl': float, # 未实现盈亏
                'liquidation_price': float,  # 强平价格（仅永续）
                'margin': float,         # 占用保证金
            }
        """
        pass
    
    @abstractmethod
    def place_order(
        self,
        symbol: str,
        side: OrderSide,
        amount: float,
        order_type: OrderType = OrderType.MARKET,
        price: Optional[float] = None,
        reduce_only: bool = False
    ) -> Dict:
        """下单
        
        Args:
            symbol: 交易对
            side: 订单方向（BUY/SELL）
            amount: 数量
            order_type: 订单类型（MARKET/LIMIT）
            price: 限价单价格（仅LIMIT类型需要）
            reduce_only: 只减仓（仅永续合约）
            
        Returns:
            {
                'order_id': str,          # 订单ID
                'filled_price': float,    # 成交均价
                'filled_amount': float,   # 成交数量
                'fee': float,             # 手续费
                'timestamp': datetime,    # 成交时间
            }
        """
        pass
    
    @abstractmethod
    def get_klines(
        self,
        symbol: str,
        interval: str,
        limit: int = 500,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None
    ) -> pd.DataFrame:
        """获取K线数据
        
        Args:
            symbol: 交易对
            interval: K线周期（1m, 5m, 15m, 1h, 4h, 1d等）
            limit: 数量限制
            start_time: 开始时间（毫秒时间戳）
            end_time: 结束时间（毫秒时间戳）
            
        Returns:
            DataFrame with columns: [open, high, low, close, volume]
            Index: DatetimeIndex (UTC)
        """
        pass
    
    @abstractmethod
    def set_leverage(self, symbol: str, leverage: int):
        """设置杠杆倍数
        
        Args:
            symbol: 交易对
            leverage: 杠杆倍数（1-125，具体范围取决于交易所）
            
        Note:
            仅永续合约支持，现货会忽略此方法
        """
        pass
    
    @abstractmethod
    def close_position(self, symbol: str, position_side: Optional[PositionSide] = None):
        """平仓
        
        Args:
            symbol: 交易对
            position_side: 持仓方向（仅双向持仓模式需要）
            
        Note:
            现货：卖出全部持仓
            永续：平掉指定方向的全部持仓
        """
        pass
    
    @abstractmethod
    def get_account_info(self) -> Dict:
        """获取账户信息
        
        Returns:
            {
                'total_balance': float,      # 总权益
                'available_balance': float,  # 可用余额
                'margin_balance': float,     # 保证金余额（仅永续）
                'unrealized_pnl': float,     # 未实现盈亏
                'margin_ratio': float,       # 保证金率（仅永续）
            }
        """
        pass
    
    def get_current_price(self, symbol: str) -> float:
        """获取当前价格（便捷方法）
        
        Args:
            symbol: 交易对
            
        Returns:
            当前价格
        """
        df = self.get_klines(symbol, "1m", limit=1)
        return float(df['close'].iloc[-1])
    
    def calculate_position_value(self, symbol: str) -> float:
        """计算持仓价值
        
        Args:
            symbol: 交易对
            
        Returns:
            持仓价值（USDT）
        """
        position = self.get_position(symbol)
        size = position['size']
        
        if size == 0:
            return 0.0
        
        current_price = self.get_current_price(symbol)
        
        if self.contract_type == ContractType.SPOT:
            # 现货：持仓价值 = 数量 * 当前价格
            return abs(size) * current_price
        else:
            # 永续合约：持仓价值 = 数量 * 当前价格 * 杠杆
            return abs(size) * current_price
    
    def calculate_required_margin(self, symbol: str, amount: float) -> float:
        """计算所需保证金
        
        Args:
            symbol: 交易对
            amount: 开仓数量
            
        Returns:
            所需保证金（USDT）
        """
        current_price = self.get_current_price(symbol)
        
        if self.contract_type == ContractType.SPOT:
            # 现货：需要全额资金
            return amount * current_price
        else:
            # 永续合约：保证金 = 价值 / 杠杆
            return (amount * current_price) / self.leverage
