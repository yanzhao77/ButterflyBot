#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
回测优化验证脚本：运行回测并输出关键指标对比
"""
import sys
import os
import json
from datetime import datetime, timezone, timedelta

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import (
    TIMEFRAME, INITIAL_CASH, SYMBOL, RETRAIN_SINCE_DAYS, RETRAIN_LIMIT,
    CONFIDENCE_THRESHOLD, SELL_THRESHOLD, STOP_LOSS_PCT, TAKE_PROFIT_PCT,
    MAX_POSITION_RATIO, COOLDOWN_BARS
)

def main():
    print("=" * 70)
    print("🔄 AI 量化交易系统 - 优化回测测试")
    print("=" * 70)
    
    # 打印当前配置
    print("\n📋 当前优化配置：")
    print(f"  • 时间框架: {TIMEFRAME}")
    print(f"  • 初始资金: ${INITIAL_CASH:.2f}")
    print(f"  • 买入阈值: {CONFIDENCE_THRESHOLD}")
    print(f"  • 卖出阈值: {SELL_THRESHOLD}")
    print(f"  • 止损比例: {STOP_LOSS_PCT * 100}%")
    print(f"  • 止盈比例: {TAKE_PROFIT_PCT * 100}%")
    print(f"  • 最大仓位: {MAX_POSITION_RATIO * 100}%")
    print(f"  • 冷却周期: {COOLDOWN_BARS} 根K线")
    
    print("\n🚀 开始运行回测...")
    print("-" * 70)
    
    try:
        # 动态导入回测模块
        from backtest.run_backtest import run_backtest
        
        # 运行回测
        metrics = run_backtest()
        
        print("\n" + "=" * 70)
        print("✅ 回测完成！")
        print("=" * 70)
        
        # 打印详细结果
        print("\n📊 回测结果摘要：")
        print(f"  • 初始资金: ${INITIAL_CASH:,.2f}")
        print(f"  • 最终资金: ${metrics.get('final_value', 0):,.2f}")
        print(f"  • 收益率: {metrics.get('total_return_pct', 0):.2f}%")
        print(f"  • 总交易数: {metrics.get('total_trades', 0)}")
        print(f"  • 胜率: {metrics.get('win_rate', 0) * 100:.2f}%")
        print(f"  • 赢亏比: {metrics.get('win_loss_ratio', 0):.2f}")
        print(f"  • AUC 得分: {metrics.get('auc', 0):.4f}")
        print(f"  • 最大回撤: {metrics.get('max_drawdown', 0):.4f}")
        print(f"  • 平均每笔收益: ${metrics.get('avg_profit_per_trade', 0):.2f}")
        print(f"  • 总收益: ${metrics.get('total_profit', 0):.2f}")
        
        # 判断结果
        if metrics.get('total_return_pct', 0) > 0:
            print("\n🎉 恭喜！回测为盈利状态！")
        elif metrics.get('total_return_pct', 0) == 0:
            print("\n😐 回测收支平衡")
        else:
            print("\n⚠️ 回测仍为亏损状态，可能需要进一步优化")
        
        # 保存结果到文件
        result_file = "../backtest_result_latest.json"
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "config": {
                    "timeframe": TIMEFRAME,
                    "initial_cash": INITIAL_CASH,
                    "confidence_threshold": CONFIDENCE_THRESHOLD,
                    "stop_loss_pct": STOP_LOSS_PCT,
                    "take_profit_pct": TAKE_PROFIT_PCT,
                    "max_position_ratio": MAX_POSITION_RATIO,
                },
                "metrics": metrics
            }, f, indent=2)
        print(f"\n💾 结果已保存到: {result_file}")
        
        return metrics.get('total_return_pct', 0)
        
    except Exception as e:
        print(f"\n❌ 回测失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = main()
    sys.exit(0 if result and result > 0 else 1)
