# core/risk/risk_manager.py
"""
风险管理器

功能：
1. 账户回撤监控（硬性止损）
2. 单笔风险控制
3. 杠杆倍数限制
4. 持仓比例限制
5. 连续亏损保护
"""

import logging
from typing import Optional, Tuple
from datetime import datetime


logger = logging.getLogger(__name__)


class RiskManager:
    """风险管理器
    
    实现多层风险控制，确保交易安全
    """
    
    def __init__(
        self,
        initial_balance: float,
        max_drawdown_pct: float = 0.15,
        max_position_ratio: float = 0.25,
        max_leverage: int = 5,
        max_consecutive_losses: int = 5,
        max_daily_loss_pct: float = 0.05,
        max_risk_per_trade: float = 0.02,
        stop_loss_pct: float = 0.02,
        take_profit_pct: float = 0.03
    ):
        """初始化风险管理器
        
        Args:
            initial_balance: 初始资金
            max_drawdown_pct: 最大回撤百分比（触发硬性止损）
            max_position_ratio: 最大仓位比例
            max_leverage: 最大杠杆倍数
            max_consecutive_losses: 最大连续亏损次数
            max_daily_loss_pct: 单日最大亏损百分比
            max_risk_per_trade: 单笔最大风险百分比
        """
        # 资金管理
        self.initial_balance = initial_balance
        self.peak_balance = initial_balance
        self.current_balance = initial_balance
        self.daily_start_balance = initial_balance
        
        # 风控参数
        self.max_drawdown_pct = max_drawdown_pct
        self.max_position_ratio = max_position_ratio
        self.max_leverage = max_leverage
        self.max_consecutive_losses = max_consecutive_losses
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_risk_per_trade = max_risk_per_trade
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        
        # 交易统计
        self.consecutive_losses = 0
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        
        # 状态控制
        self.is_paused = False
        self.pause_reason = ""
        self.pause_time = None
        
        # 日期跟踪
        self.current_date = datetime.now().date()
        
        logger.info(f"风险管理器初始化: 初始资金={initial_balance}, 最大回撤={max_drawdown_pct:.1%}")
    
    def update_balance(self, balance: float):
        """更新余额
        
        Args:
            balance: 当前余额
        """
        self.current_balance = balance
        
        # 更新峰值
        if balance > self.peak_balance:
            self.peak_balance = balance
            logger.info(f"💰 新高！峰值余额: {self.peak_balance:.2f}")
        
        # 检查日期变化
        today = datetime.now().date()
        if today != self.current_date:
            self.daily_start_balance = balance
            self.current_date = today
            logger.info(f"📅 新的一天开始，起始余额: {balance:.2f}")
    
    def get_current_drawdown(self) -> float:
        """计算当前回撤
        
        Returns:
            回撤百分比（0.0-1.0）
        """
        if self.peak_balance <= 0:
            return 0.0
        return (self.peak_balance - self.current_balance) / self.peak_balance
    
    def get_daily_pnl_pct(self) -> float:
        """计算当日盈亏百分比
        
        Returns:
            盈亏百分比（可为负）
        """
        if self.daily_start_balance <= 0:
            return 0.0
        return (self.current_balance - self.daily_start_balance) / self.daily_start_balance
    
    def check_hard_stop(self) -> bool:
        """检查硬性止损（账户回撤）
        
        Returns:
            True: 触发硬性止损
            False: 未触发
        """
        drawdown = self.get_current_drawdown()
        
        if drawdown >= self.max_drawdown_pct:
            self.is_paused = True
            self.pause_reason = f"🚨 硬性止损触发！账户回撤{drawdown:.2%}超过限制{self.max_drawdown_pct:.2%}"
            self.pause_time = datetime.now()
            logger.error(self.pause_reason)
            return True
        
        return False
    
    def check_daily_loss(self) -> bool:
        """检查单日亏损限制
        
        Returns:
            True: 触发单日亏损限制
            False: 未触发
        """
        daily_pnl_pct = self.get_daily_pnl_pct()
        
        if daily_pnl_pct <= -self.max_daily_loss_pct:
            self.is_paused = True
            self.pause_reason = f"⚠️ 单日亏损限制触发！当日亏损{abs(daily_pnl_pct):.2%}超过限制{self.max_daily_loss_pct:.2%}"
            self.pause_time = datetime.now()
            logger.warning(self.pause_reason)
            return True
        
        return False
    
    def check_position_size(self, position_value: float, balance: float) -> Tuple[bool, str]:
        """检查仓位大小
        
        Args:
            position_value: 持仓价值
            balance: 当前余额
            
        Returns:
            (是否通过, 原因)
        """
        if balance <= 0:
            return False, "余额不足"
        
        ratio = position_value / balance
        
        if ratio > self.max_position_ratio:
            return False, f"仓位比例{ratio:.2%}超过限制{self.max_position_ratio:.2%}"
        
        return True, ""
    
    def check_leverage(self, leverage: int) -> Tuple[bool, str]:
        """检查杠杆倍数
        
        Args:
            leverage: 杠杆倍数
            
        Returns:
            (是否通过, 原因)
        """
        if leverage > self.max_leverage:
            return False, f"杠杆倍数{leverage}超过限制{self.max_leverage}"
        
        if leverage < 1:
            return False, f"杠杆倍数{leverage}无效（最小为1）"
        
        return True, ""
    
    def check_trade_risk(self, entry_price: float, stop_loss_price: float, amount: float) -> Tuple[bool, str]:
        """检查单笔交易风险
        
        Args:
            entry_price: 开仓价格
            stop_loss_price: 止损价格
            amount: 交易数量
            
        Returns:
            (是否通过, 原因)
        """
        # 计算单笔风险
        risk_per_unit = abs(entry_price - stop_loss_price)
        total_risk = risk_per_unit * amount
        risk_pct = total_risk / self.current_balance
        
        if risk_pct > self.max_risk_per_trade:
            return False, f"单笔风险{risk_pct:.2%}超过限制{self.max_risk_per_trade:.2%}"
        
        return True, ""
    
    def record_trade_result(self, pnl: float):
        """记录交易结果
        
        Args:
            pnl: 盈亏金额
        """
        self.total_trades += 1
        
        if pnl > 0:
            self.winning_trades += 1
            self.consecutive_losses = 0
            logger.info(f"✅ 盈利交易 #{self.total_trades}: +{pnl:.2f}")
        elif pnl < 0:
            self.losing_trades += 1
            self.consecutive_losses += 1
            logger.warning(f"❌ 亏损交易 #{self.total_trades}: {pnl:.2f} (连续亏损{self.consecutive_losses}次)")
        else:
            logger.info(f"⚪ 平局交易 #{self.total_trades}")
        
        # 检查连续亏损
        if self.consecutive_losses >= self.max_consecutive_losses:
            self.is_paused = True
            self.pause_reason = f"⚠️ 连续亏损{self.consecutive_losses}次，达到限制{self.max_consecutive_losses}次"
            self.pause_time = datetime.now()
            logger.error(self.pause_reason)
    
    def can_trade(self) -> Tuple[bool, str]:
        """检查是否可以交易
        
        Returns:
            (是否可以交易, 原因)
        """
        # 检查是否已暂停
        if self.is_paused:
            return False, self.pause_reason
        
        # 检查硬性止损
        if self.check_hard_stop():
            return False, self.pause_reason
        
        # 检查单日亏损
        if self.check_daily_loss():
            return False, self.pause_reason
        
        return True, ""
    
    def resume_trading(self, reason: str = "手动恢复"):
        """恢复交易
        
        Args:
            reason: 恢复原因
        """
        self.is_paused = False
        self.pause_reason = ""
        self.consecutive_losses = 0
        logger.info(f"🔄 交易已恢复: {reason}")
    
    def get_stats(self) -> dict:
        """获取统计信息
        
        Returns:
            统计数据字典
        """
        win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0
        
        return {
            'initial_balance': self.initial_balance,
            'current_balance': self.current_balance,
            'peak_balance': self.peak_balance,
            'current_drawdown': self.get_current_drawdown(),
            'daily_pnl_pct': self.get_daily_pnl_pct(),
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': win_rate,
            'consecutive_losses': self.consecutive_losses,
            'is_paused': self.is_paused,
            'pause_reason': self.pause_reason,
        }
    
    def __str__(self) -> str:
        """字符串表示"""
        stats = self.get_stats()
        return (
            f"RiskManager("
            f"余额={stats['current_balance']:.2f}, "
            f"回撤={stats['current_drawdown']:.2%}, "
            f"胜率={stats['win_rate']:.2%}, "
            f"状态={'暂停' if stats['is_paused'] else '运行'}"
            f")"
        )
