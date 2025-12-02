# utils/auto_retrain.py
from model.model_registry import (
    load_latest_model_path,
    get_model_metadata,
    find_best_model_by_auc,
    update_latest_model
)
from backtest.metrics import load_metrics
from model.train import train_and_evaluate
import os

def should_retrain_and_run(threshold_delta_auc: float = 0.02):
    metrics = load_metrics()
    backtest_auc = metrics.get("auc", 0.55)

    latest_path = load_latest_model_path()
    if latest_path is None:
        print("⚠️ 无模型，启动首次训练...")
        train_and_evaluate()
        return

    version = os.path.basename(latest_path).replace(".pkl", "")
    current_auc = get_model_metadata(version).get("auc", 0.5)

    print(f"🔍 当前模型 AUC: {current_auc:.4f} | 回测 AUC: {backtest_auc:.4f}")

    if backtest_auc > current_auc + threshold_delta_auc:
        print("🚀 启动自动重训练...")
        train_and_evaluate()
        best_v = find_best_model_by_auc()
        update_latest_model(best_v)
        print(f"🏆 最优模型更新为: {best_v}")
    else:
        print("✅ 无需重训练")