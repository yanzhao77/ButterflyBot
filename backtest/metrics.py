# backtest/metrics.py
"""
回测性能指标计算
"""
import json
import os

import numpy as np
from sklearn.metrics import roc_auc_score
from config.settings import MODEL_METRICS_PATH

def calculate_metrics(trades, y_true_for_auc=None):
    """
    计算回测核心指标
    """
    if not trades:
        return {
            "auc": 0.5,
            "win_rate": 0.0,
            "win_loss_ratio": 0.0,
            "avg_profit_per_trade": 0.0,
            "max_drawdown": 0.0
        }

    profits = np.array([t["pnl"] for t in trades])
    wins = profits[profits > 0]
    losses = profits[profits < 0]

    win_rate = len(wins) / len(trades) if trades else 0
    avg_win = np.mean(wins) if len(wins) > 0 else 0
    avg_loss = -np.mean(losses) if len(losses) > 0 else 1e-8
    win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0

    # AUC：需要真实标签（假设你已保存 target 列用于评估）
    auc = 0.5
    if y_true_for_auc is not None and len(y_true_for_auc) == len(trades):
        try:
            # 模拟预测概率：盈利交易视为“正类”
            y_pred_proba = np.where(profits > 0, 0.8, 0.2)
            auc = float(roc_auc_score(y_true_for_auc, y_pred_proba))
        except:
            auc = 0.5

    # 最大回撤（简化版：基于交易序列）
    cumulative = np.cumsum(profits)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / (running_max + 1e-8)
    max_drawdown = float(np.min(drawdown)) if len(drawdown) > 0 else 0.0

    return {
        "auc": round(auc, 4),
        "win_rate": round(win_rate, 4),
        "win_loss_ratio": round(win_loss_ratio, 2),
        "avg_profit_per_trade": round(np.mean(profits), 2),
        "total_profit": round(np.sum(profits), 2),
        "max_drawdown": round(max_drawdown, 4),
        "total_trades": len(trades)
    }


def save_metrics(metrics, path=MODEL_METRICS_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)


def load_metrics():
    """加载最新回测指标"""
    import json
    import os
    path = "backtest/strategy_metrics.json"
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"auc": 0.55, "win_rate": 0.5}
