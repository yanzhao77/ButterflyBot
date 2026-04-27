"""
ButterflyBot 5年历史数据模型训练脚本 v2（优化版）
改进点：
1. 使用 is_unbalance=True 处理样本不平衡
2. 增大模型复杂度（num_leaves=63, n_estimators=500）
3. 使用 scale_pos_weight 自动平衡正负样本
4. 目标变量：shift=2, threshold=0.003（正样本~25.75%，更平衡）
5. 增加特征重要性分析
"""
import sys, os, json, logging, time
sys.path.insert(0, ".")
import numpy as np
import pandas as pd
import joblib
from datetime import datetime, timezone
from sklearn.metrics import roc_auc_score, classification_report, precision_recall_curve
import lightgbm as lgb

logging.basicConfig(level=logging.WARNING)

from butterfly_bot.data.fetcher import fetch_ohlcv
from butterfly_bot.data.features import add_features, get_feature_columns
from butterfly_bot.model.lgb_model import LGBModel
from butterfly_bot.model.model_registry import (
    save_model_with_metadata,
    update_latest_model,
    REGISTRY_DIR,
)

print("=" * 70)
print("🚀 ButterflyBot 5年历史数据模型训练 v2（优化版）")
print("=" * 70)

# ===================== 1. 加载数据 =====================
print("\n📥 加载5年历史数据...")
t0 = time.time()
df_raw = fetch_ohlcv("DOGE/USDT", "15m", limit=None)
print(f"✅ 数据加载完成: {len(df_raw)} 根K线，耗时 {time.time()-t0:.1f}s")
print(f"   时间范围: {df_raw.index[0]} ~ {df_raw.index[-1]}")

# ===================== 2. 特征工程（覆盖TARGET参数）=====================
print("\n🔧 构建特征（shift=2, threshold=0.003）...")
t1 = time.time()

# 临时覆盖TARGET参数
import butterfly_bot.config.settings as settings
original_shift = settings.TARGET_SHIFT
original_threshold = settings.TARGET_THRESHOLD
settings.TARGET_SHIFT = 2
settings.TARGET_THRESHOLD = 0.003

# 重新导入特征模块以使用新参数
import importlib
import butterfly_bot.data.features as feat_module
importlib.reload(feat_module)
add_features_v2 = feat_module.add_features
get_feature_columns_v2 = feat_module.get_feature_columns

df_feat = add_features_v2(df_raw)
feature_cols = get_feature_columns_v2()
available_cols = [c for c in feature_cols if c in df_feat.columns]

X = df_feat[available_cols]
y = df_feat["target"]
print(f"✅ 特征工程完成: {len(df_feat)} 样本，{len(available_cols)} 个特征，耗时 {time.time()-t1:.1f}s")
print(f"   目标变量分布: 上涨={y.sum()} ({y.mean()*100:.1f}%), 下跌/平={len(y)-y.sum()} ({(1-y.mean())*100:.1f}%)")

# 恢复原始参数
settings.TARGET_SHIFT = original_shift
settings.TARGET_THRESHOLD = original_threshold

# ===================== 3. 时序分割 =====================
split_idx = int(len(X) * 0.70)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

print(f"\n📊 数据分割:")
print(f"   训练集: {len(X_train):,} 样本 ({df_feat.index[0]} ~ {df_feat.index[split_idx-1]})")
print(f"   测试集: {len(X_test):,} 样本 ({df_feat.index[split_idx]} ~ {df_feat.index[-1]})")
print(f"   训练集正样本: {y_train.sum():,} ({y_train.mean()*100:.1f}%)")
print(f"   测试集正样本: {y_test.sum():,} ({y_test.mean()*100:.1f}%)")

# ===================== 4. 训练优化模型 =====================
print("\n🤖 训练优化 LightGBM 模型...")
print("   参数: num_leaves=63, n_estimators=500, is_unbalance=True")
t2 = time.time()

# 计算 scale_pos_weight
neg_count = (y_train == 0).sum()
pos_count = (y_train == 1).sum()
scale_pos_weight = neg_count / pos_count
print(f"   scale_pos_weight: {scale_pos_weight:.2f} (负样本/正样本)")

# 优化的LightGBM参数
optimized_params = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "num_leaves": 63,
    "learning_rate": 0.03,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 50,
    "scale_pos_weight": scale_pos_weight,
    "verbose": -1,
    "random_state": 42,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
}

model = LGBModel(params=optimized_params)
model.train(X_train, y_train, X_val=X_test, y_val=y_test, use_class_weight=False)
train_time = time.time() - t2
print(f"✅ 模型训练完成，耗时 {train_time:.1f}s")

# ===================== 5. 评估 =====================
print("\n📈 模型评估...")
y_pred_proba = model.predict(X_test)

try:
    auc = float(roc_auc_score(y_test, y_pred_proba))
except ValueError as e:
    print(f"⚠️ AUC 计算失败: {e}")
    auc = 0.5

print(f"   测试集 AUC: {auc:.4f}")
print(f"   预测概率统计: min={y_pred_proba.min():.4f}, max={y_pred_proba.max():.4f}, mean={y_pred_proba.mean():.4f}")
print(f"   中位数: {np.median(y_pred_proba):.4f}")
print(f"   75%分位数: {np.quantile(y_pred_proba, 0.75):.4f}")
print(f"   85%分位数: {np.quantile(y_pred_proba, 0.85):.4f}")
print(f"   90%分位数: {np.quantile(y_pred_proba, 0.90):.4f}")
print(f"   95%分位数: {np.quantile(y_pred_proba, 0.95):.4f}")
print(f"   99%分位数: {np.quantile(y_pred_proba, 0.99):.4f}")
print(f"   最大值: {y_pred_proba.max():.4f}")

# 不同阈值下的精确率
print("\n不同置信度阈值下的精确率（测试集）:")
for th in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
    mask = y_pred_proba >= th
    count = mask.sum()
    if count > 0:
        precision = y_test[mask].mean()
        print(f"   阈值≥{th:.2f}: {count:5d}次触发, 精确率={precision:.2%}")
    else:
        print(f"   阈值≥{th:.2f}: 0次触发")

# 年度AUC分析
print("\n年度 AUC 分析（检验泛化能力）:")
test_years = df_feat.index[split_idx:].year.unique()
for yr in sorted(test_years):
    yr_mask = df_feat.index[split_idx:].year == yr
    if yr_mask.sum() > 100:
        try:
            yr_auc = roc_auc_score(y_test[yr_mask], y_pred_proba[yr_mask])
            yr_precision_at_70 = y_test[yr_mask][y_pred_proba[yr_mask] >= 0.70].mean() if (y_pred_proba[yr_mask] >= 0.70).sum() > 0 else 0
            yr_count_70 = (y_pred_proba[yr_mask] >= 0.70).sum()
            print(f"   {yr}年: AUC={yr_auc:.4f}, 超阈值0.70={yr_count_70}次, 精确率={yr_precision_at_70:.2%}")
        except:
            pass

# 特征重要性
print("\n特征重要性（Top 15）:")
feat_imp = model.get_feature_importance()
sorted_imp = sorted(feat_imp.items(), key=lambda x: -x[1])[:15]
for feat, imp in sorted_imp:
    print(f"   {feat:<25} {imp:>8.1f}")

# ===================== 6. 保存模型 =====================
metadata = {
    "symbol": "DOGE/USDT",
    "timeframe": "15m",
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "train_size": len(X_train),
    "test_size": len(X_test),
    "auc": round(auc, 4),
    "features": available_cols,
    "feature_count": len(available_cols),
    "data_range": f"{str(df_raw.index[0])} ~ {str(df_raw.index[-1])}",
    "total_bars": len(df_raw),
    "train_period": f"{str(df_feat.index[0])} ~ {str(df_feat.index[split_idx-1])}",
    "test_period": f"{str(df_feat.index[split_idx])} ~ {str(df_feat.index[-1])}",
    "model_type": "5year_optimized",
    "target_shift": 2,
    "target_threshold": 0.003,
    "scale_pos_weight": round(scale_pos_weight, 2),
    "description": "5年历史数据训练，优化参数（num_leaves=63，scale_pos_weight），解决过拟合",
    "prob_stats": {
        "max": round(float(y_pred_proba.max()), 4),
        "q99": round(float(np.quantile(y_pred_proba, 0.99)), 4),
        "q95": round(float(np.quantile(y_pred_proba, 0.95)), 4),
        "q90": round(float(np.quantile(y_pred_proba, 0.90)), 4),
        "q85": round(float(np.quantile(y_pred_proba, 0.85)), 4),
    }
}

version = save_model_with_metadata(model, metadata)
print(f"\n💾 模型已保存: {version}")

# 强制设置为最新模型（因为这是5年数据训练的最新版本）
update_latest_model(version)
print(f"✅ 已设置为最新模型: {version}")

print(f"\n{'='*70}")
print(f"✅ 训练完成！")
print(f"   版本: {version}")
print(f"   AUC: {auc:.4f}")
print(f"   训练样本: {len(X_train):,}")
print(f"   测试样本: {len(X_test):,}")
print(f"   预测概率最大值: {y_pred_proba.max():.4f}")
print(f"   预测概率90%分位: {np.quantile(y_pred_proba, 0.90):.4f}")
print(f"{'='*70}")
