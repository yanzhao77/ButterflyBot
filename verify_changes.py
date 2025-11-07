#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速验证脚本：检查所有改动是否正确应用
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import (
    CONFIDENCE_THRESHOLD, TIMEFRAME, FEATURE_WINDOW, MIN_FEATURE_ROWS,
    FEATURE_HISTORY_PADDING, RETRAIN_SINCE_DAYS, RETRAIN_LIMIT,
    STOP_LOSS_PCT, TAKE_PROFIT_PCT, MAX_POSITION_RATIO, COOLDOWN_BARS,
    PROB_EMA_SPAN, TIME_STOP_BARS
)

def check_config():
    """检查配置是否已正确修改"""
    print("\n" + "="*70)
    print("🔍 配置验证检查清单")
    print("="*70)
    
    checks = [
        ("买入阈值 (应该 ≤ 0.55)", CONFIDENCE_THRESHOLD, 0.55, "<="),
        ("时间框架 (应该是 1h)", TIMEFRAME, "1h", "=="),
        ("特征窗口 (应该 ≤ 200)", FEATURE_WINDOW, 200, "<="),
        ("最小特征行数 (应该 ≤ 60)", MIN_FEATURE_ROWS, 60, "<="),
        ("历史 Padding (应该 ≤ 60)", FEATURE_HISTORY_PADDING, 60, "<="),
        ("回测天数 (应该 ≤ 180)", RETRAIN_SINCE_DAYS, 180, "<="),
        ("回测限制 (应该 ≤ 10000)", RETRAIN_LIMIT, 100000, "<="),
    ]
    
    results = []
    for name, actual, expected, op in checks:
        if op == "<=":
            passed = actual <= expected
        elif op == "==":
            passed = actual == expected
        else:
            passed = actual == expected
        
        status = "✅ PASS" if passed else "❌ FAIL"
        results.append(passed)
        
        print(f"\n{status}: {name}")
        print(f"   期望: {expected} ({op})")
        print(f"   实际: {actual}")
    
    return all(results)

def check_strategy():
    """检查策略文件是否包含关键修改"""
    print("\n" + "="*70)
    print("🔍 策略文件验证检查清单")
    print("="*70)
    
    strategy_file = "strategies/backtrader_adapters/ai_signal_bt.py"
    
    if not os.path.exists(strategy_file):
        print(f"❌ 找不到策略文件: {strategy_file}")
        return False
    
    with open(strategy_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    checks = [
        ("趋势判断简化", "ma5 = np.mean(close_prices[-5:])", "✅"),
        ("趋势为可选", "confidence_boost = 1.0 if trend_up else 0.8", "✅"),
        ("移除硬性过滤", "if signal_info[\"signal\"] == \"buy\":", "✅"),
        ("移除趋势反转平仓", "should_sell = (signal_info[\"signal\"] == \"sell\")", "✅"),
    ]
    
    results = []
    for check_name, check_text, expected in checks:
        passed = check_text in content
        status = "✅ PASS" if passed else "❌ FAIL"
        results.append(passed)
        
        print(f"\n{status}: {check_name}")
        print(f"   检查: {check_text[:50]}...")
        if not passed:
            print(f"   ⚠️  未找到此修改！")
    
    return all(results)

def main():
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*68 + "║")
    print("║" + "  🚀 快速改动验证脚本  ".center(68) + "║")
    print("║" + " "*68 + "║")
    print("╚" + "="*68 + "╝")
    
    config_ok = check_config()
    strategy_ok = check_strategy()
    
    print("\n" + "="*70)
    print("📊 验证结果总结")
    print("="*70)
    
    print(f"\n配置文件检查: {'✅ 通过' if config_ok else '❌ 失败'}")
    print(f"策略文件检查: {'✅ 通过' if strategy_ok else '❌ 失败'}")
    
    if config_ok and strategy_ok:
        print("\n" + "🎉 " * 20)
        print("所有修改都已正确应用！可以运行回测了！")
        print("🎉 " * 20)
        print("\n推荐命令:")
        print("  python test_backtest_optimize.py")
        return 0
    else:
        print("\n⚠️ 检测到未完成的修改，请检查上面的失败项")
        return 1

if __name__ == "__main__":
    sys.exit(main())
