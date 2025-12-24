from .base import BaseBroker, OrderSide, OrderType, ContractType
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class BacktestBroker(BaseBroker):
    def __init__(self, initial_balance, leverage, contract_type, data, fee_rate=0.001):
        super().__init__(initial_balance, contract_type)
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.leverage = leverage
        self.data = data
        self.position = {"size": 0, "entry_price": 0.0}
        self.trades = []
        self.fee_rate = fee_rate  # 默认0.1%手续费
        
        logger.info(f"🔧 BacktestBroker初始化:")
        logger.info(f"   初始余额: {initial_balance}")
        logger.info(f"   杠杆: {leverage}x")
        logger.info(f"   手续费率: {fee_rate*100}%")
        logger.info(f"   合约类型: {contract_type}")

    def get_balance(self):
        """返回现金余额"""
        return self.balance
    
    def get_total_value(self, symbol=None):
        """返回总资产价值（现金 + 持仓价值）"""
        cash = self.balance
        position_value = 0.0
        
        if self.position["size"] > 0:
            current_price = self.get_current_price(symbol or "")
            position_value = self.position["size"] * current_price
        
        total = cash + position_value
        
        logger.debug(f"💰 总资产: cash={cash:.2f}, position={position_value:.2f}, total={total:.2f}")
        return total

    def get_position(self, symbol):
        return self.position

    def place_order(self, symbol, side, amount, order_type, price=0.0):
        current_price = self.get_current_price(symbol)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"📝 下单请求:")
        logger.info(f"   交易对: {symbol}")
        logger.info(f"   方向: {side}")
        logger.info(f"   数量: {amount}")
        logger.info(f"   当前价格: {current_price}")
        logger.info(f"   下单前余额: {self.balance:.2f}")
        logger.info(f"   下单前持仓: {self.position}")
        
        if side == OrderSide.BUY:
            # 买入
            cost = amount * current_price
            fee = cost * self.fee_rate
            total_cost = cost + fee
            
            logger.info(f"💵 买入计算:")
            logger.info(f"   成本: {cost:.2f}")
            logger.info(f"   手续费: {fee:.2f} ({self.fee_rate*100}%)")
            logger.info(f"   总成本: {total_cost:.2f}")
            
            if total_cost > self.balance:
                logger.error(f"❌ 余额不足: 需要{total_cost:.2f}, 只有{self.balance:.2f}")
                return {"status": "REJECTED", "reason": "Insufficient balance"}
            
            self.balance -= total_cost
            self.position["size"] = amount
            self.position["entry_price"] = current_price
            
            logger.info(f"✅ 买入成功:")
            logger.info(f"   新余额: {self.balance:.2f}")
            logger.info(f"   新持仓: size={self.position['size']:.2f}, entry={self.position['entry_price']:.4f}")
            logger.info(f"   总资产: {self.get_total_value(symbol):.2f}")
            
        elif side == OrderSide.SELL:
            # 卖出
            if self.position["size"] == 0:
                logger.warning(f"⚠️  没有持仓，无法卖出")
                return {"status": "REJECTED", "reason": "No position to sell"}
            
            revenue = self.position["size"] * current_price
            fee = revenue * self.fee_rate
            net_revenue = revenue - fee
            
            pnl = (current_price - self.position["entry_price"]) * self.position["size"] - fee
            pnl_pct = pnl / (self.position["entry_price"] * self.position["size"]) * 100
            
            logger.info(f"💵 卖出计算:")
            logger.info(f"   收入: {revenue:.2f}")
            logger.info(f"   手续费: {fee:.2f} ({self.fee_rate*100}%)")
            logger.info(f"   净收入: {net_revenue:.2f}")
            logger.info(f"   盈亏: {pnl:.2f} ({pnl_pct:.2f}%)")
            
            self.balance += net_revenue
            
            self.trades.append({
                "entry_price": self.position["entry_price"],
                "exit_price": current_price,
                "size": self.position["size"],
                "pnl": pnl,
                "pnl_pct": pnl_pct
            })
            
            logger.info(f"✅ 卖出成功:")
            logger.info(f"   新余额: {self.balance:.2f}")
            logger.info(f"   总资产: {self.get_total_value(symbol):.2f}")
            
            self.position = {"size": 0, "entry_price": 0.0}
        
        logger.info(f"{'='*60}\n")
        return {"status": "FILLED"}

    def close_position(self, symbol, current_price=None):
        """平仓
        
        Args:
            symbol: 交易对
            current_price: 当前价格（可选，如果提供则使用，否则从数据中获取）
        """
        if self.position["size"] > 0:
            return self.place_order(symbol, OrderSide.SELL, self.position["size"], OrderType.MARKET)
        return {"status": "NO_POSITION"}

    def get_current_price(self, symbol):
        # In backtesting, we assume the current price is the close of the current bar
        if not self.data.empty:
            return self.data.iloc[-1]["close"]
        return 0.0

    def get_account_info(self):
        return {
            "totalWalletBalance": self.balance,
            "totalPositionValue": self.position["size"] * self.get_current_price("") if self.position["size"] > 0 else 0.0,
            "totalValue": self.get_total_value()
        }

    def get_klines(self, symbol, timeframe, limit):
        return self.data.tail(limit)

    def set_leverage(self, symbol, leverage):
        self.leverage = leverage
