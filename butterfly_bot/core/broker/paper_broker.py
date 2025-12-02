"""
PaperBroker - 纸上交易模拟器
用于实时模拟交易，不涉及真实资金
"""

import logging
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime
from .base import BaseBroker, ContractType, OrderSide, OrderType, PositionSide

logger = logging.getLogger(__name__)


class PaperBroker(BaseBroker):
    """纸上交易模拟器，用于实时模拟交易"""
    
    def __init__(
        self,
        initial_balance: float,
        leverage: int = 1,
        contract_type: ContractType = ContractType.SPOT,
        commission_rate: float = 0.001
    ):
        """初始化纸上交易模拟器
        
        Args:
            initial_balance: 初始余额
            leverage: 杠杆倍数
            contract_type: 合约类型
            commission_rate: 手续费率
        """
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.leverage = leverage
        self.contract_type = contract_type
        self.commission_rate = commission_rate
        
        # 持仓和订单记录
        self.positions: Dict[str, Dict] = {}
        self.orders: List[Dict] = []
        self.trades: List[Dict] = []
        
        logger.info(f"PaperBroker初始化: 初始余额={initial_balance}, 杠杆={leverage}x, 合约类型={contract_type.name}")
    
    def get_balance(self) -> float:
        """获取当前余额"""
        return self.balance
    
    def get_position(self, symbol: str) -> Dict:
        """获取指定交易对的持仓信息
        
        Returns:
            {
                'symbol': str,
                'size': float,  # 持仓数量（正数=多头，负数=空头）
                'entry_price': float,  # 开仓均价
                'unrealized_pnl': float,  # 未实现盈亏
                'leverage': int
            }
        """
        if symbol not in self.positions:
            return {
                'symbol': symbol,
                'size': 0.0,
                'entry_price': 0.0,
                'unrealized_pnl': 0.0,
                'leverage': self.leverage
            }
        return self.positions[symbol].copy()
    
    def place_order(
        self,
        symbol: str,
        side: OrderSide,
        amount: float,
        order_type: OrderType = OrderType.MARKET,
        price: Optional[float] = None
    ) -> Dict:
        """下单
        
        Args:
            symbol: 交易对
            side: 买卖方向
            amount: 数量
            order_type: 订单类型
            price: 价格（市价单可选）
            
        Returns:
            订单信息字典
        """
        # 对于实时模拟，我们需要获取当前市场价格
        # 这里暂时使用传入的price参数
        if price is None:
            logger.warning(f"市价单需要提供当前价格，订单取消")
            return {}
        
        # 计算手续费
        commission = amount * price * self.commission_rate
        
        # 检查余额是否足够
        required_balance = amount * price / self.leverage + commission
        if self.balance < required_balance:
            logger.warning(f"余额不足: 需要{required_balance:.2f}, 当前{self.balance:.2f}")
            return {}
        
        # 创建订单
        order = {
            'order_id': f"PAPER_{len(self.orders) + 1}",
            'symbol': symbol,
            'side': side.name,
            'amount': amount,
            'price': price,
            'order_type': order_type.name,
            'status': 'FILLED',
            'timestamp': datetime.now(),
            'commission': commission
        }
        
        self.orders.append(order)
        
        # 更新持仓
        self._update_position(symbol, side, amount, price, commission)
        
        logger.info(f"订单已执行: {side.name} {amount} {symbol} @ {price}, 手续费={commission:.4f}")
        
        return order
    
    def _update_position(self, symbol: str, side: OrderSide, amount: float, price: float, commission: float):
        """更新持仓信息"""
        if symbol not in self.positions:
            self.positions[symbol] = {
                'symbol': symbol,
                'size': 0.0,
                'entry_price': 0.0,
                'unrealized_pnl': 0.0,
                'leverage': self.leverage
            }
        
        position = self.positions[symbol]
        
        # 计算新的持仓
        if side == OrderSide.BUY:
            new_size = position['size'] + amount
        else:  # SELL
            new_size = position['size'] - amount
        
        # 如果是开仓或加仓
        if (position['size'] >= 0 and side == OrderSide.BUY) or \
           (position['size'] <= 0 and side == OrderSide.SELL):
            # 计算新的平均开仓价
            total_cost = abs(position['size']) * position['entry_price'] + amount * price
            total_amount = abs(position['size']) + amount
            position['entry_price'] = total_cost / total_amount if total_amount > 0 else 0
        else:
            # 平仓或反向开仓
            if abs(new_size) < abs(position['size']):
                # 部分平仓，计算盈亏
                closed_amount = abs(position['size']) - abs(new_size)
                pnl = closed_amount * (price - position['entry_price']) * (1 if position['size'] > 0 else -1)
                self.balance += pnl
                
                # 记录交易
                self.trades.append({
                    'symbol': symbol,
                    'side': 'CLOSE',
                    'amount': closed_amount,
                    'entry_price': position['entry_price'],
                    'exit_price': price,
                    'pnl': pnl - commission,
                    'timestamp': datetime.now()
                })
                
                logger.info(f"部分平仓: {closed_amount} {symbol}, 盈亏={pnl:.2f}")
            else:
                # 完全平仓或反向开仓
                if position['size'] != 0:
                    pnl = abs(position['size']) * (price - position['entry_price']) * (1 if position['size'] > 0 else -1)
                    self.balance += pnl
                    
                    # 记录交易
                    self.trades.append({
                        'symbol': symbol,
                        'side': 'CLOSE',
                        'amount': abs(position['size']),
                        'entry_price': position['entry_price'],
                        'exit_price': price,
                        'pnl': pnl - commission,
                        'timestamp': datetime.now()
                    })
                    
                    logger.info(f"完全平仓: {abs(position['size'])} {symbol}, 盈亏={pnl:.2f}")
                
                # 如果有反向开仓
                if abs(new_size) > 0:
                    position['entry_price'] = price
        
        position['size'] = new_size
        
        # 扣除手续费
        self.balance -= commission
    
    def close_position(self, symbol: str, position_side: Optional['PositionSide'] = None, current_price: Optional[float] = None) -> bool:
        """平仓
        
        Args:
            symbol: 交易对
            current_price: 当前价格
            
        Returns:
            是否成功平仓
        """
        if symbol not in self.positions or self.positions[symbol]['size'] == 0:
            logger.warning(f"没有持仓: {symbol}")
            return False
        
        position = self.positions[symbol]
        
        if current_price is None:
            logger.warning(f"平仓需要提供当前价格")
            return False
        
        # 确定平仓方向
        if position['size'] > 0:
            side = OrderSide.SELL
        else:
            side = OrderSide.BUY
        
        # 执行平仓订单
        self.place_order(symbol, side, abs(position['size']), OrderType.MARKET, current_price)
        
        return True
    
    def get_account_info(self) -> Dict:
        """获取账户信息"""
        total_unrealized_pnl = 0.0
        for position in self.positions.values():
            total_unrealized_pnl += position['unrealized_pnl']
        
        return {
            'balance': self.balance,
            'initial_balance': self.initial_balance,
            'total_unrealized_pnl': total_unrealized_pnl,
            'equity': self.balance + total_unrealized_pnl,
            'leverage': self.leverage,
            'contract_type': self.contract_type.name,
            'total_trades': len(self.trades)
        }
    
    def update_unrealized_pnl(self, symbol: str, current_price: float):
        """更新未实现盈亏
        
        Args:
            symbol: 交易对
            current_price: 当前价格
        """
        if symbol in self.positions and self.positions[symbol]['size'] != 0:
            position = self.positions[symbol]
            unrealized_pnl = position['size'] * (current_price - position['entry_price'])
            position['unrealized_pnl'] = unrealized_pnl
    
    def get_klines(
        self,
        symbol: str,
        interval: str,
        limit: int = 500,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None
    ) -> pd.DataFrame:
        """获取K线数据（PaperBroker不支持，返回空DataFrame）
        
        Note:
            PaperBroker需要外部提供K线数据，此方法仅为满足接口要求
        """
        logger.warning("PaperBroker不支持直接获取K线数据，请使用RealtimeFetcher")
        return pd.DataFrame()
    
    def set_leverage(self, symbol: str, leverage: int):
        """设置杠杆倍数
        
        Args:
            symbol: 交易对
            leverage: 杠杆倍数
        """
        self.leverage = leverage
        logger.info(f"设置杠杆: {symbol} = {leverage}x")
