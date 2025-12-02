import logging

class TradingEngine:
    def __init__(self, broker, risk_manager, symbol, get_signal_func):
        self.broker = broker
        self.risk_manager = risk_manager
        self.symbol = symbol
        self.get_signal = get_signal_func
        self.is_running = False

    def start(self):
        self.is_running = True
        logging.info("Trading engine started.")

    def stop(self):
        self.is_running = False
        logging.info("Trading engine stopped.")

    def execute_signal(self, signal, confidence, current_price, stop_loss_pct, take_profit_pct):
        if not self.is_running:
            return

        self.risk_manager.update_balance(self.broker.get_balance())
        can_trade, reason = self.risk_manager.can_trade()
        if not can_trade:
            logging.warning(f"Risk manager disallows trading: {reason}")
            return

        position = self.broker.get_position(self.symbol)

        if signal.upper() == "BUY" and position['size'] == 0:
            amount = self.risk_manager.calculate_position_size(self.broker.get_balance(), current_price)
            self.broker.place_order(self.symbol, "BUY", amount, "MARKET")
        elif signal.upper() == "SELL" and position['size'] > 0:
            self.broker.close_position(self.symbol)
        elif signal.upper() == "SHORT" and position['size'] == 0:
            amount = self.risk_manager.calculate_position_size(self.broker.get_balance(), current_price)
            self.broker.place_order(self.symbol, "SELL", amount, "MARKET")
        elif signal.upper() == "COVER" and position['size'] < 0:
            self.broker.close_position(self.symbol)
