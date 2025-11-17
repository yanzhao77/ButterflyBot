#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
策略对比测试 - 对比优化前后的策略效果
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 80)
print("ButterflyBot 策略优化效果对比")
print("=" * 80)

# 定义优化前后的参数
OLD_PARAMS = {
    "name": "原始策略",
    "CONFIDENCE_THRESHOLD": 0.47,
    "SELL_THRESHOLD": 0.45,
    "STOP_LOSS_PCT": 0.008,
    "TAKE_PROFIT_PCT": 0.01,
    "MAX_POSITION_RATIO": 0.30,
    "TIME_STOP_BARS": 20,
    "COOLDOWN_BARS": 8,
}

NEW_PARAMS = {
    "name": "优化策略",
    "CONFIDENCE_THRESHOLD": 0.62,
    "SELL_THRESHOLD": 0.40,
    "STOP_LOSS_PCT": 0.025,
    "TAKE_PROFIT_PCT": 0.05,
    "MAX_POSITION_RATIO": 0.25,
    "TIME_STOP_BARS": 50,
    "COOLDOWN_BARS": 5,
}

def calculate_expected_return(params):
    """计算策略的期望收益"""
    # 简化的数学模型
    
    # 1. 估算交易频率（基于阈值）
    threshold_gap = params["CONFIDENCE_THRESHOLD"] - params["SELL_THRESHOLD"]
    # 阈值差距越大，交易越少
    trades_per_100_bars = max(5, 50 / threshold_gap)
    
    # 2. 估算胜率（基于阈值高低）
    # 阈值越高，胜率越高
    base_win_rate = 0.45
    threshold_bonus = (params["CONFIDENCE_THRESHOLD"] - 0.5) * 0.3
    win_rate = min(0.65, base_win_rate + threshold_bonus)
    
    # 3. 计算盈亏比
    profit_loss_ratio = params["TAKE_PROFIT_PCT"] / params["STOP_LOSS_PCT"]
    
    # 4. 手续费影响
    commission = 0.002  # 0.2% 双向
    
    # 5. 计算期望收益
    avg_profit = params["TAKE_PROFIT_PCT"] - commission
    avg_loss = params["STOP_LOSS_PCT"] + commission
    
    expected_per_trade = win_rate * avg_profit - (1 - win_rate) * avg_loss
    
    # 6. 考虑仓位比例
    position_adjusted = expected_per_trade * params["MAX_POSITION_RATIO"]
    
    return {
        "trades_per_100": trades_per_100_bars,
        "win_rate": win_rate,
        "profit_loss_ratio": profit_loss_ratio,
        "expected_per_trade": expected_per_trade,
        "expected_per_100_bars": position_adjusted * trades_per_100_bars,
        "avg_profit": avg_profit,
        "avg_loss": avg_loss,
    }

def print_comparison():
    """打印对比结果"""
    print("\n" + "=" * 80)
    print("参数对比")
    print("=" * 80)
    
    print(f"\n{'参数':<25} {'原始策略':<20} {'优化策略':<20} {'变化':<15}")
    print("-" * 80)
    
    for key in ["CONFIDENCE_THRESHOLD", "SELL_THRESHOLD", "STOP_LOSS_PCT", 
                "TAKE_PROFIT_PCT", "MAX_POSITION_RATIO", "TIME_STOP_BARS", "COOLDOWN_BARS"]:
        old_val = OLD_PARAMS[key]
        new_val = NEW_PARAMS[key]
        
        if isinstance(old_val, float):
            change = f"{((new_val - old_val) / old_val * 100):+.1f}%"
            print(f"{key:<25} {old_val:<20.3f} {new_val:<20.3f} {change:<15}")
        else:
            change = f"{new_val - old_val:+d}"
            print(f"{key:<25} {old_val:<20} {new_val:<20} {change:<15}")
    
    print("\n" + "=" * 80)
    print("性能预测对比")
    print("=" * 80)
    
    old_metrics = calculate_expected_return(OLD_PARAMS)
    new_metrics = calculate_expected_return(NEW_PARAMS)
    
    print(f"\n{'指标':<30} {'原始策略':<20} {'优化策略':<20} {'改善':<15}")
    print("-" * 85)
    
    metrics = [
        ("每100根K线交易次数", "trades_per_100", "次"),
        ("预估胜率", "win_rate", "%"),
        ("盈亏比", "profit_loss_ratio", ":1"),
        ("平均盈利(扣费)", "avg_profit", "%"),
        ("平均亏损(含费)", "avg_loss", "%"),
        ("单次交易期望收益", "expected_per_trade", "%"),
        ("100根K线期望收益", "expected_per_100_bars", "%"),
    ]
    
    for label, key, unit in metrics:
        old_val = old_metrics[key]
        new_val = new_metrics[key]
        
        if unit == "%":
            old_str = f"{old_val*100:.2f}%"
            new_str = f"{new_val*100:.2f}%"
            change = f"{(new_val - old_val)*100:+.2f}pp"
        elif unit == "次":
            old_str = f"{old_val:.1f}"
            new_str = f"{new_val:.1f}"
            change = f"{new_val - old_val:+.1f}"
        elif unit == ":1":
            old_str = f"{old_val:.2f}:1"
            new_str = f"{new_val:.2f}:1"
            change = f"{((new_val - old_val) / old_val * 100):+.1f}%"
        else:
            old_str = f"{old_val:.4f}"
            new_str = f"{new_val:.4f}"
            change = f"{new_val - old_val:+.4f}"
        
        print(f"{label:<30} {old_str:<20} {new_str:<20} {change:<15}")
    
    print("\n" + "=" * 80)
    print("核心改进说明")
    print("=" * 80)
    
    improvements = [
        ("✅ 盈亏比提升", f"{OLD_PARAMS['TAKE_PROFIT_PCT']/OLD_PARAMS['STOP_LOSS_PCT']:.2f}:1 → {NEW_PARAMS['TAKE_PROFIT_PCT']/NEW_PARAMS['STOP_LOSS_PCT']:.2f}:1", 
         "从1.25:1提升至2:1，大幅改善风险收益比"),
        
        ("✅ 止损放宽", f"{OLD_PARAMS['STOP_LOSS_PCT']*100:.1f}% → {NEW_PARAMS['STOP_LOSS_PCT']*100:.1f}%",
         "避免被正常波动扫出，提高持仓稳定性"),
        
        ("✅ 止盈提高", f"{OLD_PARAMS['TAKE_PROFIT_PCT']*100:.1f}% → {NEW_PARAMS['TAKE_PROFIT_PCT']*100:.1f}%",
         "给予趋势充分发展空间，捕捉更大行情"),
        
        ("✅ 阈值优化", f"买入{OLD_PARAMS['CONFIDENCE_THRESHOLD']:.2f}→{NEW_PARAMS['CONFIDENCE_THRESHOLD']:.2f}",
         "提高开仓标准，只在高确定性时交易"),
        
        ("✅ 持仓延长", f"{OLD_PARAMS['TIME_STOP_BARS']}根→{NEW_PARAMS['TIME_STOP_BARS']}根K线",
         "从5小时延长至12.5小时，让利润奔跑"),
        
        ("✅ 交易频率", f"降低约{(1 - new_metrics['trades_per_100']/old_metrics['trades_per_100'])*100:.0f}%",
         "减少无效交易，降低手续费损耗"),
        
        ("✅ 期望收益", f"{old_metrics['expected_per_trade']*100:.3f}% → {new_metrics['expected_per_trade']*100:.3f}%",
         "单次交易期望收益转正" if new_metrics['expected_per_trade'] > 0 else "接近盈亏平衡"),
    ]
    
    for title, change, desc in improvements:
        print(f"\n{title}")
        print(f"  变化: {change}")
        print(f"  说明: {desc}")
    
    print("\n" + "=" * 80)
    print("结论")
    print("=" * 80)
    
    if new_metrics['expected_per_trade'] > 0 and old_metrics['expected_per_trade'] <= 0:
        print("\n🎉 优化成功！策略期望收益从负值转为正值，具备盈利基础。")
    elif new_metrics['expected_per_trade'] > old_metrics['expected_per_trade']:
        print(f"\n✅ 优化有效！期望收益提升 {(new_metrics['expected_per_trade'] - old_metrics['expected_per_trade'])*100:.2f}%")
    else:
        print("\n⚠️  优化效果有限，需要进一步调整")
    
    print("\n关键要点:")
    print("  1. 盈亏比从1.25:1提升至2:1，这是最关键的改进")
    print("  2. 交易频率降低，减少手续费侵蚀")
    print("  3. 更严格的开仓条件，提高胜率")
    print("  4. 更长的持仓时间，捕捉完整趋势")
    print("  5. 添加跟踪止盈功能，让利润奔跑")
    
    print("\n⚠️  风险提示:")
    print("  • 理论计算基于简化模型，实际效果取决于市场环境")
    print("  • 趋势市表现更好，震荡市可能仍然亏损")
    print("  • 建议先用小资金测试，验证后再扩大规模")
    print("  • 需要定期监控和调整参数")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    print_comparison()
    print("\n✅ 对比分析完成\n")
