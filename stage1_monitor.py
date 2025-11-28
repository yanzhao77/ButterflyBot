#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
阶段1实盘监控系统
实时监控交易表现，生成报告，风控检查
"""

import os
import sys
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np

# 阶段1风控参数
STAGE1_CONFIG = {
    # 资金管理
    'initial_capital': 100.0,
    'max_position_ratio': 0.20,  # 20%
    'min_trade_amount': 10.0,
    
    # 风控参数
    'daily_max_loss': 5.0,  # $5
    'daily_max_trades': 10,
    'weekly_max_loss': 10.0,  # $10
    'max_drawdown': 0.15,  # 15%
    'consecutive_loss_pause': 5,
    
    # 警戒线
    'equity_warning': 95.0,  # $95
    'equity_danger': 90.0,   # $90
    'winrate_warning': 0.45,
    'winrate_danger': 0.40,
    
    # 交易参数（调整后）
    'confidence_threshold': 0.08,  # 提高
    'stop_loss': 0.02,
    'take_profit': 0.025,  # 降低
    'time_stop': 15,  # 缩短
    'cooldown': 5,  # 延长
}

class Stage1Monitor:
    """阶段1实盘监控器"""
    
    def __init__(self, data_dir='stage1_data'):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        self.trades_file = self.data_dir / 'trades.json'
        self.equity_file = self.data_dir / 'equity.json'
        self.daily_file = self.data_dir / 'daily.json'
        
        self.trades = self.load_trades()
        self.equity_curve = self.load_equity()
        self.daily_stats = self.load_daily()
        
        self.config = STAGE1_CONFIG
        
    def load_trades(self):
        """加载交易记录"""
        if self.trades_file.exists():
            with open(self.trades_file, 'r') as f:
                return json.load(f)
        return []
    
    def load_equity(self):
        """加载权益曲线"""
        if self.equity_file.exists():
            with open(self.equity_file, 'r') as f:
                return json.load(f)
        return []
    
    def load_daily(self):
        """加载每日统计"""
        if self.daily_file.exists():
            with open(self.daily_file, 'r') as f:
                return json.load(f)
        return {}
    
    def save_trades(self):
        """保存交易记录"""
        with open(self.trades_file, 'w') as f:
            json.dump(self.trades, f, indent=2)
    
    def save_equity(self):
        """保存权益曲线"""
        with open(self.equity_file, 'w') as f:
            json.dump(self.equity_curve, f, indent=2)
    
    def save_daily(self):
        """保存每日统计"""
        with open(self.daily_file, 'w') as f:
            json.dump(self.daily_stats, f, indent=2)
    
    def add_trade(self, trade):
        """添加交易记录"""
        trade['timestamp'] = datetime.now().isoformat()
        self.trades.append(trade)
        self.save_trades()
    
    def update_equity(self, equity, cash, position):
        """更新权益曲线"""
        self.equity_curve.append({
            'timestamp': datetime.now().isoformat(),
            'equity': equity,
            'cash': cash,
            'position': position
        })
        self.save_equity()
    
    def get_current_stats(self):
        """获取当前统计数据"""
        if not self.trades:
            return {
                'total_trades': 0,
                'wins': 0,
                'losses': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0,
            }
        
        df = pd.DataFrame(self.trades)
        wins = df[df['pnl'] > 0]
        losses = df[df['pnl'] <= 0]
        
        total_win = wins['pnl'].sum() if len(wins) > 0 else 0
        total_loss = abs(losses['pnl'].sum()) if len(losses) > 0 else 0
        
        return {
            'total_trades': len(df),
            'wins': len(wins),
            'losses': len(losses),
            'win_rate': len(wins) / len(df) if len(df) > 0 else 0,
            'total_pnl': df['pnl'].sum(),
            'avg_win': wins['pnl'].mean() if len(wins) > 0 else 0,
            'avg_loss': losses['pnl'].mean() if len(losses) > 0 else 0,
            'profit_factor': total_win / total_loss if total_loss > 0 else 0,
        }
    
    def get_today_stats(self):
        """获取今日统计"""
        today = datetime.now().date().isoformat()
        today_trades = [t for t in self.trades if t['timestamp'][:10] == today]
        
        if not today_trades:
            return {
                'trades': 0,
                'pnl': 0,
                'wins': 0,
                'losses': 0,
            }
        
        df = pd.DataFrame(today_trades)
        return {
            'trades': len(df),
            'pnl': df['pnl'].sum(),
            'wins': len(df[df['pnl'] > 0]),
            'losses': len(df[df['pnl'] <= 0]),
        }
    
    def check_risk_control(self):
        """检查风控条件"""
        warnings = []
        dangers = []
        
        # 获取当前权益
        if self.equity_curve:
            current_equity = self.equity_curve[-1]['equity']
        else:
            current_equity = self.config['initial_capital']
        
        # 检查权益
        if current_equity < self.config['equity_danger']:
            dangers.append(f"账户权益低于危险线: ${current_equity:.2f} < ${self.config['equity_danger']:.2f}")
        elif current_equity < self.config['equity_warning']:
            warnings.append(f"账户权益低于警戒线: ${current_equity:.2f} < ${self.config['equity_warning']:.2f}")
        
        # 检查回撤
        if self.equity_curve:
            equity_series = pd.Series([e['equity'] for e in self.equity_curve])
            max_equity = equity_series.max()
            current_drawdown = (max_equity - current_equity) / max_equity
            
            if current_drawdown > self.config['max_drawdown']:
                dangers.append(f"回撤超过限制: {current_drawdown*100:.2f}% > {self.config['max_drawdown']*100:.2f}%")
        
        # 检查今日交易
        today_stats = self.get_today_stats()
        
        if today_stats['pnl'] < -self.config['daily_max_loss']:
            dangers.append(f"今日亏损超限: ${today_stats['pnl']:.2f} < -${self.config['daily_max_loss']:.2f}")
        
        if today_stats['trades'] >= self.config['daily_max_trades']:
            warnings.append(f"今日交易次数达到上限: {today_stats['trades']} >= {self.config['daily_max_trades']}")
        
        # 检查胜率
        stats = self.get_current_stats()
        if stats['total_trades'] >= 10:  # 至少10笔交易
            if stats['win_rate'] < self.config['winrate_danger']:
                dangers.append(f"胜率过低: {stats['win_rate']*100:.1f}% < {self.config['winrate_danger']*100:.1f}%")
            elif stats['win_rate'] < self.config['winrate_warning']:
                warnings.append(f"胜率偏低: {stats['win_rate']*100:.1f}% < {self.config['winrate_warning']*100:.1f}%")
        
        # 检查连续亏损
        if len(self.trades) >= self.config['consecutive_loss_pause']:
            recent_trades = self.trades[-self.config['consecutive_loss_pause']:]
            if all(t['pnl'] <= 0 for t in recent_trades):
                dangers.append(f"连续{self.config['consecutive_loss_pause']}次亏损，建议暂停交易")
        
        return {
            'warnings': warnings,
            'dangers': dangers,
            'should_pause': len(dangers) > 0
        }
    
    def generate_daily_report(self):
        """生成每日报告"""
        today = datetime.now().date().isoformat()
        today_stats = self.get_today_stats()
        overall_stats = self.get_current_stats()
        
        # 获取权益
        if self.equity_curve:
            current_equity = self.equity_curve[-1]['equity']
            initial_equity = self.config['initial_capital']
        else:
            current_equity = self.config['initial_capital']
            initial_equity = self.config['initial_capital']
        
        # 计算回撤
        if self.equity_curve:
            equity_series = pd.Series([e['equity'] for e in self.equity_curve])
            max_equity = equity_series.max()
            max_drawdown = ((max_equity - equity_series) / max_equity * 100).max()
        else:
            max_equity = initial_equity
            max_drawdown = 0
        
        # 风控检查
        risk_check = self.check_risk_control()
        
        report = f"""
{'='*80}
每日报告 - {today}
{'='*80}

📊 账户状态
  初始权益: ${initial_equity:.2f}
  当前权益: ${current_equity:.2f}
  当日盈亏: ${today_stats['pnl']:+.2f} ({today_stats['pnl']/initial_equity*100:+.2f}%)
  累计盈亏: ${current_equity - initial_equity:+.2f} ({(current_equity - initial_equity)/initial_equity*100:+.2f}%)
  最大回撤: {max_drawdown:.2f}%

📈 交易统计
  今日交易: {today_stats['trades']}次
  今日盈利: {today_stats['wins']}次
  今日亏损: {today_stats['losses']}次
  今日胜率: {today_stats['wins']/today_stats['trades']*100:.1f}% (如有交易)

  总交易: {overall_stats['total_trades']}次
  总盈利: {overall_stats['wins']}次 ({overall_stats['win_rate']*100:.1f}%)
  总亏损: {overall_stats['losses']}次
  
💰 盈亏分析
  平均盈利: ${overall_stats['avg_win']:.2f}
  平均亏损: ${overall_stats['avg_loss']:.2f}
  盈亏比: {abs(overall_stats['avg_win']/overall_stats['avg_loss']):.2f}:1 (如有亏损)
"""
        
        # 添加风险提示
        if risk_check['warnings'] or risk_check['dangers']:
            report += "\n⚠️ 风险提示\n"
            for warning in risk_check['warnings']:
                report += f"  ⚠️ {warning}\n"
            for danger in risk_check['dangers']:
                report += f"  ❌ {danger}\n"
            
            if risk_check['should_pause']:
                report += "\n  🛑 建议暂停交易！\n"
        else:
            report += "\n✅ 所有指标正常\n"
        
        report += f"\n{'='*80}\n"
        
        # 保存报告
        report_file = self.data_dir / f'daily_report_{today}.txt'
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        return report
    
    def generate_weekly_report(self):
        """生成每周报告"""
        # 获取本周数据
        today = datetime.now()
        week_start = today - timedelta(days=today.weekday())
        week_trades = [t for t in self.trades if datetime.fromisoformat(t['timestamp']) >= week_start]
        
        if not week_trades:
            return "本周暂无交易数据"
        
        df = pd.DataFrame(week_trades)
        
        # 按日期分组
        df['date'] = pd.to_datetime(df['timestamp']).dt.date
        daily = df.groupby('date').agg({
            'pnl': 'sum',
            'type': 'count'
        }).rename(columns={'type': 'trades'})
        
        # 计算胜率
        daily['wins'] = df[df['pnl'] > 0].groupby('date').size()
        daily['wins'] = daily['wins'].fillna(0)
        daily['win_rate'] = daily['wins'] / daily['trades'] * 100
        
        report = f"""
{'='*80}
周报 - {week_start.date()} 至 {today.date()}
{'='*80}

📊 每日表现
"""
        for date, row in daily.iterrows():
            status = "✅" if row['pnl'] > 0 else "❌"
            report += f"  {date}: ${row['pnl']:+6.2f} | {int(row['trades'])}笔 | 胜率{row['win_rate']:.0f}% {status}\n"
        
        report += f"""
📈 周度统计
  总交易: {len(df)}次
  总盈亏: ${df['pnl'].sum():+.2f}
  胜率: {len(df[df['pnl'] > 0])/len(df)*100:.1f}%
  盈利天数: {len(daily[daily['pnl'] > 0])}天
  亏损天数: {len(daily[daily['pnl'] <= 0])}天

💡 建议
"""
        
        # 分析并给出建议
        win_rate = len(df[df['pnl'] > 0])/len(df)
        if win_rate < 0.5:
            report += "  ⚠️ 胜率偏低，建议提高置信度阈值\n"
        elif win_rate > 0.65:
            report += "  ✅ 胜率优秀，可以考虑适当降低阈值增加交易\n"
        
        if df['pnl'].sum() < 0:
            report += "  ❌ 本周亏损，建议暂停交易并分析原因\n"
        elif df['pnl'].sum() > 10:
            report += "  🎉 本周盈利优秀，保持当前策略\n"
        
        report += f"\n{'='*80}\n"
        
        # 保存报告
        report_file = self.data_dir / f'weekly_report_{today.date()}.txt'
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        return report

def main():
    """主函数 - 演示用法"""
    monitor = Stage1Monitor()
    
    # 示例：添加交易
    # monitor.add_trade({
    #     'type': 'long',
    #     'entry': 0.385,
    #     'exit': 0.392,
    #     'pnl': 1.75,
    #     'return': 0.0182,
    #     'bars': 3,
    #     'reason': '信号反转'
    # })
    
    # 示例：更新权益
    # monitor.update_equity(equity=103.50, cash=78.50, position='long')
    
    # 生成报告
    print("生成每日报告...")
    daily_report = monitor.generate_daily_report()
    print(daily_report)
    
    # 风控检查
    print("\n风控检查...")
    risk_check = monitor.check_risk_control()
    if risk_check['should_pause']:
        print("⚠️ 建议暂停交易！")
        for danger in risk_check['dangers']:
            print(f"  ❌ {danger}")
    else:
        print("✅ 所有风控指标正常")

if __name__ == "__main__":
    main()
