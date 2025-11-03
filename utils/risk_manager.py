# utils/risk_manager.py
import math

def kelly_fraction(win_rate: float, win_loss_ratio: float) -> float:
    """凯利公式：f = W - (1 - W) / R"""
    if win_loss_ratio <= 0:
        return 0.0
    f = win_rate - (1 - win_rate) / win_loss_ratio
    return max(0.0, min(f, 0.25))  # 限制最大仓位 25%

def calculate_position_size(account_value: float, entry_price: float,
                            stop_loss_price: float, win_rate: float,
                            win_loss_ratio: float) -> float:
    risk_per_trade = kelly_fraction(win_rate, win_loss_ratio)
    risk_amount = account_value * risk_per_trade
    risk_per_unit = abs(entry_price - stop_loss_price)
    if risk_per_unit == 0:
        return 0.0
    units = risk_amount / risk_per_unit
    return units * entry_price  # 返回 USDT 金额