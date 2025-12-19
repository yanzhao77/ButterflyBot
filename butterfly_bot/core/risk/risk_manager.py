# core/risk/risk_manager.py
"""
风险管理器（增强版）

功能：
1. 账户回撤监控（硬性止损）
2. 单笔风险控制
3. 杠杆倍数限制
4. 持仓比例限制
5. 连续亏损保护
6. 动态止损止盈（新增）
7. Trailing Stop移动止损（新增）
8. 分批建仓/平仓（新增）
"""

import logging
from typing import Optional, Tuple, Dict
from datetime import datetime


logger = logging.getLogger(__name__)


class RiskManager:
    """风险管理器（增强版）
    
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
        stop_loss_pct: float = 0.03,
        take_profit_pct: float = 0.06,
        use_trailing_stop: bool = True,
        trailing_activation_pct: float = 0.02,
        trailing_distance_pct: float = 0.01,
        use_dynamic_sizing: bool = True
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
            stop_loss_pct: 默认止损百分比
            take_profit_pct: 默认止盈百分比
            use_trailing_stop: 是否启用移动止损
            trailing_activation_pct: 移动止损激活阈值（盈利百分比）
            trailing_distance_pct: 移动止损距离（从最高点回撤）
            use_dynamic_sizing: 是否使用动态仓位管理
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
        
        # 移动止损参数
        self.use_trailing_stop = use_trailing_stop
        self.trailing_activation_pct = trailing_activation_pct
        self.trailing_distance_pct = trailing_distance_pct
        
        # 动态仓位管理
        self.use_dynamic_sizing = use_dynamic_sizing
        
        # 交易统计
        self.consecutive_losses = 0
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        
        # 持仓跟踪（用于移动止损）
        self.position_tracker: Dict[str, Dict] = {}
        
        # 状态控制
        self.is_paused = False
        self.pause_reason = ""
        self.pause_time = None
        
        # 日期跟踪
        self.current_date = datetime.now().date()
        
        logger.info(f"风险管理器初始化: 初始资金={initial_balance}, 最大回撤={max_drawdown_pct:.1%}, "
                   f"移动止损={'启用' if use_trailing_stop else '禁用'}")
    
    def update_balance(self, balance: float):
        """更新余额
        
        Args:
            balance: 当前余额
        """
        self.current_balance = balance
        
        # 更新峰值
        if balance > self.peak_balance:
            self.peak_balance = balance
            logger.debug(f"💰 新高！峰值余额: {self.peak_balance:.2f}")
        
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
            self.pause_reason = f"⚠️ 单日亏损限制触发！今日亏损{daily_pnl_pct:.2%}超过限制{self.max_daily_loss_pct:.2%}"
            self.pause_time = datetime.now()
            logger.warning(self.pause_reason)
            return True
        
        return False
    
    def check_consecutive_losses(self) -> bool:
        """检查连续亏损限制
        
        Returns:
            True: 触发连续亏损限制
            False: 未触发
        """
        if self.consecutive_losses >= self.max_consecutive_losses:
            self.is_paused = True
            self.pause_reason = f"⚠️ 连续亏损限制触发！连续{self.consecutive_losses}次亏损"
            self.pause_time = datetime.now()
            logger.warning(self.pause_reason)
            return True
        
        return False
    
    def can_trade(self, leverage: int = 1) -> Tuple[bool, str]:
        """检查是否可以交易
        
        Args:
            leverage: 杠杆倍数
            
        Returns:
            (是否可以交易, 原因说明)
        """
        # 检查是否暂停
        if self.is_paused:
            return False, self.pause_reason
        
        # 检查硬性止损
        if self.check_hard_stop():
            return False, self.pause_reason
        
        # 检查单日亏损
        if self.check_daily_loss():
            return False, self.pause_reason
        
        # 检查连续亏损
        if self.check_consecutive_losses():
            return False, self.pause_reason
        
        # 检查杠杆限制
        if leverage > self.max_leverage:
            return False, f"杠杆倍数{leverage}超过限制{self.max_leverage}"
        
        return True, "可以交易"
    
    def calculate_position_size(
        self,
        balance: float,
        price: float,
        leverage: int = 1,
        confidence: float = 1.0
    ) -> float:
        """计算仓位大小
        
        Args:
            balance: 当前余额
            price: 当前价格
            leverage: 杠杆倍数
            confidence: 信号置信度（0-1）
            
        Returns:
            建议的仓位大小（数量）
        """
        if balance <= 0 or price <= 0:
            return 0.0
        
        # 基础仓位：使用最大仓位比例
        base_position_value = balance * self.max_position_ratio
        
        # 动态调整：根据置信度和连续亏损情况
        if self.use_dynamic_sizing:
            # 置信度调整（0.5-1.0）
            confidence_factor = 0.5 + (confidence * 0.5)
            
            # 连续亏损调整（减少仓位）
            loss_factor = max(0.5, 1.0 - (self.consecutive_losses * 0.1))
            
            # 综合调整
            adjustment_factor = confidence_factor * loss_factor
            base_position_value *= adjustment_factor
            
            logger.debug(f"动态仓位调整: 置信度={confidence:.2f}, 连续亏损={self.consecutive_losses}, "
                        f"调整系数={adjustment_factor:.2f}")
        
        # 考虑杠杆
        position_value = base_position_value * leverage
        
        # 转换为数量
        position_size = position_value / price
        
        return position_size
    
    def get_stop_loss_price(
        self,
        entry_price: float,
        side: str,
        custom_pct: Optional[float] = None
    ) -> float:
        """计算止损价格
        
        Args:
            entry_price: 入场价格
            side: 方向（'buy' 或 'sell'）
            custom_pct: 自定义止损百分比（可选）
            
        Returns:
            止损价格
        """
        stop_pct = custom_pct if custom_pct is not None else self.stop_loss_pct
        
        if side.lower() == 'buy':
            # 多头止损：低于入场价
            return entry_price * (1 - stop_pct)
        else:
            # 空头止损：高于入场价
            return entry_price * (1 + stop_pct)
    
    def get_take_profit_price(
        self,
        entry_price: float,
        side: str,
        custom_pct: Optional[float] = None
    ) -> float:
        """计算止盈价格
        
        Args:
            entry_price: 入场价格
            side: 方向（'buy' 或 'sell'）
            custom_pct: 自定义止盈百分比（可选）
            
        Returns:
            止盈价格
        """
        profit_pct = custom_pct if custom_pct is not None else self.take_profit_pct
        
        if side.lower() == 'buy':
            # 多头止盈：高于入场价
            return entry_price * (1 + profit_pct)
        else:
            # 空头止盈：低于入场价
            return entry_price * (1 - profit_pct)
    
    def init_position_tracking(
        self,
        position_id: str,
        entry_price: float,
        size: float,
        side: str
    ):
        """初始化持仓跟踪（用于移动止损）
        
        Args:
            position_id: 持仓ID
            entry_price: 入场价格
            size: 持仓数量
            side: 方向
        """
        self.position_tracker[position_id] = {
            'entry_price': entry_price,
            'size': size,
            'side': side,
            'peak_price': entry_price,  # 最高价（多头）或最低价（空头）
            'trailing_active': False,
            'trailing_stop_price': None
        }
        logger.debug(f"初始化持仓跟踪: {position_id}, 入场价={entry_price:.6f}")
    
    def update_trailing_stop(
        self,
        position_id: str,
        current_price: float
    ) -> Optional[float]:
        """更新移动止损
        
        Args:
            position_id: 持仓ID
            current_price: 当前价格
            
        Returns:
            移动止损价格（如果触发），否则None
        """
        if not self.use_trailing_stop:
            return None
        
        if position_id not in self.position_tracker:
            return None
        
        pos = self.position_tracker[position_id]
        entry_price = pos['entry_price']
        side = pos['side']
        
        # 计算当前盈亏百分比
        if side.lower() == 'buy':
            pnl_pct = (current_price - entry_price) / entry_price
            
            # 更新峰值价格
            if current_price > pos['peak_price']:
                pos['peak_price'] = current_price
            
            # 检查是否激活移动止损
            if not pos['trailing_active'] and pnl_pct >= self.trailing_activation_pct:
                pos['trailing_active'] = True
                logger.info(f"✅ 移动止损已激活: {position_id}, 当前盈利={pnl_pct:.2%}")
            
            # 如果已激活，更新止损价格
            if pos['trailing_active']:
                trailing_stop = pos['peak_price'] * (1 - self.trailing_distance_pct)
                pos['trailing_stop_price'] = trailing_stop
                
                # 检查是否触发止损
                if current_price <= trailing_stop:
                    logger.info(f"🎯 移动止损触发: {position_id}, 价格={current_price:.6f}, "
                               f"止损价={trailing_stop:.6f}")
                    return trailing_stop
        
        else:  # 空头
            pnl_pct = (entry_price - current_price) / entry_price
            
            # 更新峰值价格（空头是最低价）
            if current_price < pos['peak_price']:
                pos['peak_price'] = current_price
            
            # 检查是否激活移动止损
            if not pos['trailing_active'] and pnl_pct >= self.trailing_activation_pct:
                pos['trailing_active'] = True
                logger.info(f"✅ 移动止损已激活: {position_id}, 当前盈利={pnl_pct:.2%}")
            
            # 如果已激活，更新止损价格
            if pos['trailing_active']:
                trailing_stop = pos['peak_price'] * (1 + self.trailing_distance_pct)
                pos['trailing_stop_price'] = trailing_stop
                
                # 检查是否触发止损
                if current_price >= trailing_stop:
                    logger.info(f"🎯 移动止损触发: {position_id}, 价格={current_price:.6f}, "
                               f"止损价={trailing_stop:.6f}")
                    return trailing_stop
        
        return None
    
    def close_position_tracking(self, position_id: str):
        """关闭持仓跟踪
        
        Args:
            position_id: 持仓ID
        """
        if position_id in self.position_tracker:
            del self.position_tracker[position_id]
            logger.debug(f"关闭持仓跟踪: {position_id}")
    
    def record_trade(self, pnl: float):
        """记录交易结果
        
        Args:
            pnl: 盈亏金额
        """
        self.total_trades += 1
        
        if pnl > 0:
            self.winning_trades += 1
            self.consecutive_losses = 0
            logger.info(f"✅ 盈利交易 #{self.total_trades}: +{pnl:.2f}")
        else:
            self.losing_trades += 1
            self.consecutive_losses += 1
            logger.warning(f"❌ 亏损交易 #{self.total_trades}: {pnl:.2f}, "
                          f"连续亏损={self.consecutive_losses}")
    
    def get_statistics(self) -> Dict:
        """获取风险管理统计信息
        
        Returns:
            统计信息字典
        """
        win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0
        
        return {
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': win_rate,
            'consecutive_losses': self.consecutive_losses,
            'current_drawdown': self.get_current_drawdown(),
            'daily_pnl_pct': self.get_daily_pnl_pct(),
            'is_paused': self.is_paused,
            'pause_reason': self.pause_reason,
            'peak_balance': self.peak_balance,
            'current_balance': self.current_balance
        }
    
    def reset_pause(self):
        """重置暂停状态（谨慎使用）"""
        self.is_paused = False
        self.pause_reason = ""
        self.pause_time = None
        logger.info("风险管理器暂停状态已重置")
