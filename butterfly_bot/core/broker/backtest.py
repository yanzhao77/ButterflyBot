from .base import BaseBroker, OrderSide, OrderType, ContractType
import pandas as pd

class BacktestBroker(BaseBroker):
    def __init__(self, initial_balance, leverage, contract_type, data):
        super().__init__(initial_balance, contract_type)
        self.balance = initial_balance
        self.leverage = leverage
        self.data = data
        self.position = {"size": 0, "entry_price": 0.0}
        self.trades = []

    def get_balance(self):
        return self.balance

    def get_position(self, symbol):
        return self.position

    def place_order(self, symbol, side, amount, order_type, price=0.0):
        current_price = self.get_current_price(symbol)
        if side == OrderSide.BUY:
            self.position["size"] = amount
            self.position["entry_price"] = current_price
            self.balance -= amount * current_price
        elif side == OrderSide.SELL:
            self.balance += self.position["size"] * current_price
            self.trades.append({
                "entry_price": self.position["entry_price"],
                "exit_price": current_price,
                "size": self.position["size"],
                "pnl": (current_price - self.position["entry_price"]) * self.position["size"]
            })
            self.position = {"size": 0, "entry_price": 0.0}
        return {"status": "FILLED"}

    def close_position(self, symbol):
        return self.place_order(symbol, OrderSide.SELL, 0, OrderType.MARKET)

    def get_current_price(self, symbol):
        # In backtesting, we assume the current price is the close of the current bar
        # This is a simplification. A more realistic backtester would handle this differently.
        if not self.data.empty:
            return self.data.iloc[-1]["close"]
        return 0.0

    def get_account_info(self):
        return {"totalWalletBalance": self.balance}

    def get_klines(self, symbol, timeframe, limit):
        return self.data.tail(limit)

    def set_leverage(self, symbol, leverage):
        self.leverage = leverage
