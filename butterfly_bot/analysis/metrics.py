"""
性能指标计算器 - 计算回测和实盘的各种性能指标
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class PerformanceMetrics:
    """性能指标计算器"""
    
    def __init__(self, initial_balance: float = 1000.0, risk_free_rate: float = 0.03):
        """初始化
        
        Args:
            initial_balance: 初始资金
            risk_free_rate: 无风险利率（年化）
        """
        self.initial_balance = initial_balance
        self.risk_free_rate = risk_free_rate
    
    def calculate_all_metrics(
        self,
        equity_curve: pd.Series,
        trades: List[Dict],
        period_days: Optional[int] = None
    ) -> Dict:
        """计算所有性能指标
        
        Args:
            equity_curve: 权益曲线（时间序列）
            trades: 交易记录列表
            period_days: 回测周期天数（用于年化计算）
            
        Returns:
            包含所有指标的字典
        """
        metrics = {}
        
        # 收益指标
        metrics.update(self.calculate_returns(equity_curve, period_days))
        
        # 风险指标
        metrics.update(self.calculate_risk(equity_curve))
        
        # 风险调整收益指标
        metrics.update(self.calculate_risk_adjusted(equity_curve, period_days))
        
        # 交易统计
        if trades:
            metrics.update(self.calculate_trade_stats(trades))
        
        return metrics
    
    def calculate_returns(
        self,
        equity_curve: pd.Series,
        period_days: Optional[int] = None
    ) -> Dict:
        """计算收益率指标
        
        Args:
            equity_curve: 权益曲线
            period_days: 回测周期天数
            
        Returns:
            收益率指标字典
        """
        if len(equity_curve) == 0:
            return {
                'total_return': 0.0,
                'annualized_return': 0.0,
                'monthly_return': 0.0,
            }
        
        final_equity = equity_curve.iloc[-1]
        total_return = (final_equity - self.initial_balance) / self.initial_balance
        
        # 计算年化收益率
        if period_days is None:
            # 根据时间索引计算天数
            if isinstance(equity_curve.index, pd.DatetimeIndex):
                period_days = (equity_curve.index[-1] - equity_curve.index[0]).days
            else:
                period_days = len(equity_curve)  # 假设每个点是一天
        
        if period_days > 0:
            years = period_days / 365.0
            annualized_return = (1 + total_return) ** (1 / years) - 1
            monthly_return = (1 + total_return) ** (1 / (years * 12)) - 1
        else:
            annualized_return = 0.0
            monthly_return = 0.0
        
        return {
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'annualized_return': annualized_return,
            'annualized_return_pct': annualized_return * 100,
            'monthly_return': monthly_return,
            'monthly_return_pct': monthly_return * 100,
            'final_equity': final_equity,
            'total_profit': final_equity - self.initial_balance,
        }
    
    def calculate_risk(self, equity_curve: pd.Series) -> Dict:
        """计算风险指标
        
        Args:
            equity_curve: 权益曲线
            
        Returns:
            风险指标字典
        """
        if len(equity_curve) < 2:
            return {
                'max_drawdown': 0.0,
                'max_drawdown_pct': 0.0,
                'max_drawdown_duration': 0,
                'volatility': 0.0,
                'downside_volatility': 0.0,
            }
        
        # 计算回撤
        running_max = equity_curve.expanding().max()
        drawdown = (equity_curve - running_max) / running_max
        max_drawdown = drawdown.min()
        max_drawdown_pct = abs(max_drawdown * 100)
        
        # 计算最大回撤持续时间
        drawdown_duration = self._calculate_drawdown_duration(equity_curve)
        
        # 计算收益率
        returns = equity_curve.pct_change().dropna()
        
        # 计算波动率（年化）
        if len(returns) > 0:
            volatility = returns.std() * np.sqrt(252)  # 假设252个交易日
            
            # 计算下行波动率（只考虑负收益）
            negative_returns = returns[returns < 0]
            if len(negative_returns) > 0:
                downside_volatility = negative_returns.std() * np.sqrt(252)
            else:
                downside_volatility = 0.0
        else:
            volatility = 0.0
            downside_volatility = 0.0
        
        return {
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown_pct,
            'max_drawdown_duration': drawdown_duration,
            'volatility': volatility,
            'volatility_pct': volatility * 100,
            'downside_volatility': downside_volatility,
            'downside_volatility_pct': downside_volatility * 100,
        }
    
    def calculate_risk_adjusted(
        self,
        equity_curve: pd.Series,
        period_days: Optional[int] = None
    ) -> Dict:
        """计算风险调整收益指标
        
        Args:
            equity_curve: 权益曲线
            period_days: 回测周期天数
            
        Returns:
            风险调整收益指标字典
        """
        if len(equity_curve) < 2:
            return {
                'sharpe_ratio': 0.0,
                'sortino_ratio': 0.0,
                'calmar_ratio': 0.0,
            }
        
        # 计算收益率
        returns = equity_curve.pct_change().dropna()
        
        if len(returns) == 0:
            return {
                'sharpe_ratio': 0.0,
                'sortino_ratio': 0.0,
                'calmar_ratio': 0.0,
            }
        
        # 计算年化收益率
        return_metrics = self.calculate_returns(equity_curve, period_days)
        annualized_return = return_metrics['annualized_return']
        
        # 计算夏普比率
        excess_returns = returns - (self.risk_free_rate / 252)  # 日无风险利率
        if excess_returns.std() > 0:
            sharpe_ratio = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252)
        else:
            sharpe_ratio = 0.0
        
        # 计算索提诺比率
        negative_returns = returns[returns < 0]
        if len(negative_returns) > 0 and negative_returns.std() > 0:
            sortino_ratio = ((returns.mean() - self.risk_free_rate / 252) / negative_returns.std()) * np.sqrt(252)
        else:
            sortino_ratio = 0.0
        
        # 计算卡玛比率
        risk_metrics = self.calculate_risk(equity_curve)
        max_drawdown = abs(risk_metrics['max_drawdown'])
        if max_drawdown > 0:
            calmar_ratio = annualized_return / max_drawdown
        else:
            calmar_ratio = 0.0
        
        return {
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
        }
    
    def calculate_trade_stats(self, trades: List[Dict]) -> Dict:
        """计算交易统计
        
        Args:
            trades: 交易记录列表，每个交易包含 pnl, side 等信息
            
        Returns:
            交易统计字典
        """
        if not trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'win_rate_pct': 0.0,
                'avg_profit': 0.0,
                'avg_loss': 0.0,
                'profit_loss_ratio': 0.0,
                'max_consecutive_wins': 0,
                'max_consecutive_losses': 0,
                'total_fees': 0.0,
            }
        
        total_trades = len(trades)
        
        # 分离盈利和亏损交易
        profits = [t.get('pnl', 0) for t in trades if t.get('pnl', 0) > 0]
        losses = [t.get('pnl', 0) for t in trades if t.get('pnl', 0) < 0]
        
        winning_trades = len(profits)
        losing_trades = len(losses)
        
        # 计算胜率
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
        
        # 计算平均盈利和亏损
        avg_profit = np.mean(profits) if profits else 0.0
        avg_loss = abs(np.mean(losses)) if losses else 0.0
        
        # 计算盈亏比
        profit_loss_ratio = avg_profit / avg_loss if avg_loss > 0 else 0.0
        
        # 计算最大连续盈利和亏损
        max_consecutive_wins = self._calculate_max_consecutive(trades, 'win')
        max_consecutive_losses = self._calculate_max_consecutive(trades, 'loss')
        
        # 计算总手续费
        total_fees = sum(t.get('fee', 0) for t in trades)
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'win_rate_pct': win_rate * 100,
            'avg_profit': avg_profit,
            'avg_loss': avg_loss,
            'profit_loss_ratio': profit_loss_ratio,
            'max_consecutive_wins': max_consecutive_wins,
            'max_consecutive_losses': max_consecutive_losses,
            'total_fees': total_fees,
            'avg_trade_pnl': np.mean([t.get('pnl', 0) for t in trades]),
        }
    
    def _calculate_drawdown_duration(self, equity_curve: pd.Series) -> int:
        """计算最大回撤持续时间（天数或数据点数）"""
        try:
            running_max = equity_curve.expanding().max()
            drawdown = equity_curve - running_max
            
            # 找到最大回撤的位置
            max_dd_idx = drawdown.idxmin()
            
            # 从最大回撤位置往前找到峰值
            peak_idx = equity_curve[:max_dd_idx].idxmax()
            
            # 从最大回撤位置往后找到恢复点
            recovery_curve = equity_curve.loc[max_dd_idx:]
            peak_value = float(equity_curve.loc[peak_idx])
            
            # 使用values进行比较，避免索引问题
            recovery_mask = recovery_curve.values >= peak_value
            if recovery_mask.any():
                recovery_positions = np.where(recovery_mask)[0]
                recovery_idx = recovery_curve.index[recovery_positions[0]]
            else:
                recovery_idx = equity_curve.index[-1]
            
            # 计算持续时间
            if isinstance(equity_curve.index, pd.DatetimeIndex):
                duration = (recovery_idx - peak_idx).days
            else:
                # 使用位置索引计算
                peak_pos = equity_curve.index.get_loc(peak_idx)
                recovery_pos = equity_curve.index.get_loc(recovery_idx)
                duration = recovery_pos - peak_pos
            
            return max(0, duration)
        except Exception as e:
            logger.warning(f"计算回撤持续时间失败: {e}")
            return 0
    
    def _calculate_max_consecutive(self, trades: List[Dict], trade_type: str) -> int:
        """计算最大连续盈利或亏损次数"""
        if not trades:
            return 0
        
        max_consecutive = 0
        current_consecutive = 0
        
        for trade in trades:
            pnl = trade.get('pnl', 0)
            
            if trade_type == 'win' and pnl > 0:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            elif trade_type == 'loss' and pnl < 0:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive
    
    def format_metrics(self, metrics: Dict) -> str:
        """格式化指标输出
        
        Args:
            metrics: 指标字典
            
        Returns:
            格式化的字符串
        """
        output = []
        output.append("=" * 60)
        output.append("性能指标报告")
        output.append("=" * 60)
        
        # 收益指标
        output.append("\n【收益指标】")
        output.append(f"  总收益率: {metrics.get('total_return_pct', 0):.2f}%")
        output.append(f"  年化收益率: {metrics.get('annualized_return_pct', 0):.2f}%")
        output.append(f"  月均收益率: {metrics.get('monthly_return_pct', 0):.2f}%")
        output.append(f"  总盈亏: {metrics.get('total_profit', 0):.2f} USDT")
        output.append(f"  最终权益: {metrics.get('final_equity', 0):.2f} USDT")
        
        # 风险指标
        output.append("\n【风险指标】")
        output.append(f"  最大回撤: {metrics.get('max_drawdown_pct', 0):.2f}%")
        output.append(f"  回撤持续时间: {metrics.get('max_drawdown_duration', 0)} 天")
        output.append(f"  波动率: {metrics.get('volatility_pct', 0):.2f}%")
        output.append(f"  下行波动率: {metrics.get('downside_volatility_pct', 0):.2f}%")
        
        # 风险调整收益
        output.append("\n【风险调整收益】")
        output.append(f"  夏普比率: {metrics.get('sharpe_ratio', 0):.3f}")
        output.append(f"  索提诺比率: {metrics.get('sortino_ratio', 0):.3f}")
        output.append(f"  卡玛比率: {metrics.get('calmar_ratio', 0):.3f}")
        
        # 交易统计
        if 'total_trades' in metrics:
            output.append("\n【交易统计】")
            output.append(f"  总交易次数: {metrics.get('total_trades', 0)}")
            output.append(f"  盈利交易: {metrics.get('winning_trades', 0)}")
            output.append(f"  亏损交易: {metrics.get('losing_trades', 0)}")
            output.append(f"  胜率: {metrics.get('win_rate_pct', 0):.2f}%")
            output.append(f"  平均盈利: {metrics.get('avg_profit', 0):.2f} USDT")
            output.append(f"  平均亏损: {metrics.get('avg_loss', 0):.2f} USDT")
            output.append(f"  盈亏比: {metrics.get('profit_loss_ratio', 0):.2f}")
            output.append(f"  最大连续盈利: {metrics.get('max_consecutive_wins', 0)}")
            output.append(f"  最大连续亏损: {metrics.get('max_consecutive_losses', 0)}")
            output.append(f"  总手续费: {metrics.get('total_fees', 0):.2f} USDT")
        
        output.append("=" * 60)
        
        return "\n".join(output)
