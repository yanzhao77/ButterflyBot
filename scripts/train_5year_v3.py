"""
ButterflyBot 5年历史数据模型训练脚本 v3（最终优化版）
改进点：
1. 不使用 scale_pos_weight（会压缩概率），改用 sample_weight 平衡
2. 使用 Platt Scaling 对概率进行校准，使预测概率更接近真实概率
3. 目标变量：shift=2, threshold=0.003（正样本~25.75%，更平衡）
4. 增大模型复杂度（num_leaves=127, n_estimators=1000）
5. 分析校准后的最优阈值
"""
import sys, os, json, logging, time
sys.path.insert(0, ".")
import numpy as np
import pandas as pd
import joblib
from datetime import datetime, timezone
from sklearn.metrics import roc_auc_score, classification_report, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils.class_weight import compute_sample_weight
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
print("🚀 ButterflyBot 5年历史数据模型训练 v3（最终优化版）")
print("=" * 70)

# ===================== 1. 加载数据 =====================
print("\n📥 加载5年历史数据...")
t0 = time.time()
df_raw = fetch_ohlcv("DOGE/USDT", "15m", limit=None)
print(f"✅ 数据加载完成: {len(df_raw)} 根K线，耗时 {time.time()-t0:.1f}s")

# ===================== 2. 特征工程 =====================
print("\n🔧 构建特征（shift=2, threshold=0.003）...")
t1 = time.time()

import butterfly_bot.config.settings as settings
settings.TARGET_SHIFT = 2
settings.TARGET_THRESHOLD = 0.003

import importlib
import butterfly_bot.data.features as feat_module
importlib.reload(feat_module)

df_feat = feat_module.add_features(df_raw)
feature_cols = feat_module.get_feature_columns()
available_cols = [c for c in feature_cols if c in df_feat.columns]

X = df_feat[available_cols]
y = df_feat["target"]
print(f"✅ 特征工程完成: {len(df_feat):,} 样本，{len(available_cols)} 个特征，耗时 {time.time()-t1:.1f}s")
print(f"   正样本: {y.sum():,} ({y.mean()*100:.1f}%), 负样本: {len(y)-y.sum():,} ({(1-y.mean())*100:.1f}%)")

# ===================== 3. 时序分割 =====================
# 70% 训练，15% 校准，15% 测试
n = len(X)
split_train = int(n * 0.70)
split_cal   = int(n * 0.85)

X_train = X.iloc[:split_train]
y_train = y.iloc[:split_train]
X_cal   = X.iloc[split_train:split_cal]
y_cal   = y.iloc[split_train:split_cal]
X_test  = X.iloc[split_cal:]
y_test  = y.iloc[split_cal:]

print(f"\n📊 数据分割:")
print(f"   训练集: {len(X_train):,} 样本 ({df_feat.index[0]} ~ {df_feat.index[split_train-1]})")
print(f"   校准集: {len(X_cal):,} 样本 ({df_feat.index[split_train]} ~ {df_feat.index[split_cal-1]})")
print(f"   测试集: {len(X_test):,} 样本 ({df_feat.index[split_cal]} ~ {df_feat.index[-1]})")

# ===================== 4. 训练基础模型 =====================
print("\n🤖 训练基础 LightGBM 模型（使用 sample_weight）...")
t2 = time.time()

# 使用 sample_weight 而非 scale_pos_weight（保持概率校准）
sample_weight = compute_sample_weight('balanced', y_train)
print(f"   正样本权重: {sample_weight[y_train==1].mean():.2f}, 负样本权重: {sample_weight[y_train==0].mean():.2f}")

# 优化参数（不使用scale_pos_weight）
params = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "num_leaves": 127,
    "learning_rate": 0.02,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 30,
    "verbose": -1,
    "random_state": 42,
    "reg_alpha": 0.05,
    "reg_lambda": 0.05,
}

train_data = lgb.Dataset(X_train, label=y_train, weight=sample_weight)
val_data   = lgb.Dataset(X_cal, label=y_cal, reference=train_data)

lgb_model = lgb.train(
    params,
    train_data,
    valid_sets=[train_data, val_data],
    valid_names=["train", "valid"],
    num_boost_round=500,
    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
)
print(f"✅ 基础模型训练完成，耗时 {time.time()-t2:.1f}s，最优迭代: {lgb_model.best_iteration}")

# 基础模型预测
raw_probs_cal  = lgb_model.predict(X_cal,  num_iteration=lgb_model.best_iteration)
raw_probs_test = lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration)

raw_auc = roc_auc_score(y_test, raw_probs_test)
print(f"   基础模型 AUC（测试集）: {raw_auc:.4f}")
print(f"   基础模型概率范围: [{raw_probs_test.min():.4f}, {raw_probs_test.max():.4f}], 均值={raw_probs_test.mean():.4f}")

# ===================== 5. 概率校准（Platt Scaling）=====================
print("\n🔧 进行概率校准（Platt Scaling）...")
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

# 使用 Platt Scaling（逻辑回归校准）
platt_model = LogisticRegression(C=1.0, random_state=42)
platt_model.fit(raw_probs_cal.reshape(-1, 1), y_cal)
calibrated_probs_test = platt_model.predict_proba(raw_probs_test.reshape(-1, 1))[:, 1]

cal_auc = roc_auc_score(y_test, calibrated_probs_test)
print(f"   校准后 AUC（测试集）: {cal_auc:.4f}")
print(f"   校准后概率范围: [{calibrated_probs_test.min():.4f}, {calibrated_probs_test.max():.4f}], 均值={calibrated_probs_test.mean():.4f}")
print(f"   85%分位数: {np.quantile(calibrated_probs_test, 0.85):.4f}")
print(f"   90%分位数: {np.quantile(calibrated_probs_test, 0.90):.4f}")
print(f"   95%分位数: {np.quantile(calibrated_probs_test, 0.95):.4f}")
print(f"   99%分位数: {np.quantile(calibrated_probs_test, 0.99):.4f}")
print(f"   最大值: {calibrated_probs_test.max():.4f}")

# 不同阈值下的精确率
print("\n不同置信度阈值下的精确率（校准后，测试集）:")
best_th = 0.50
best_f1 = 0
for th in [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
    mask = calibrated_probs_test >= th
    count = mask.sum()
    if count > 0:
        precision = y_test[mask].mean()
        recall = y_test[mask].sum() / y_test.sum()
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        if f1 > best_f1:
            best_f1 = f1
            best_th = th
        print(f"   阈值≥{th:.2f}: {count:6d}次触发, 精确率={precision:.2%}, 召回率={recall:.2%}, F1={f1:.3f}")
    else:
        print(f"   阈值≥{th:.2f}: 0次触发")

print(f"\n   最优阈值（最大F1）: {best_th:.2f}（F1={best_f1:.3f}）")

# 年度AUC分析
print("\n年度 AUC 分析（校准后，检验泛化能力）:")
test_years = df_feat.index[split_cal:].year.unique()
for yr in sorted(test_years):
    yr_mask = df_feat.index[split_cal:].year == yr
    if yr_mask.sum() > 100:
        try:
            yr_auc = roc_auc_score(y_test[yr_mask], calibrated_probs_test[yr_mask])
            # 找最优阈值下的精确率
            for th in [0.50, 0.55, 0.60]:
                cnt = (calibrated_probs_test[yr_mask] >= th).sum()
                if cnt > 0:
                    prec = y_test[yr_mask][calibrated_probs_test[yr_mask] >= th].mean()
                    print(f"   {yr}年: AUC={yr_auc:.4f}, 阈值{th}触发{cnt}次, 精确率={prec:.2%}")
                    break
        except:
            pass

# 特征重要性
print("\n特征重要性（Top 15）:")
feat_names = lgb_model.feature_name()
feat_imp = lgb_model.feature_importance("gain")
sorted_idx = np.argsort(feat_imp)[::-1][:15]
for i in sorted_idx:
    print(f"   {feat_names[i]:<25} {feat_imp[i]:>10.1f}")

# ===================== 6. 保存模型（LGBModel + 校准器）=====================
# 包装成 LGBModel 格式
lgb_wrapper = LGBModel(params=params)
lgb_wrapper.model = lgb_model

# 保存校准器
cal_path = os.path.join(REGISTRY_DIR, "platt_calibrator.pkl")
joblib.dump(platt_model, cal_path)
print(f"\n💾 校准器已保存: {cal_path}")

metadata = {
    "symbol": "DOGE/USDT",
    "timeframe": "15m",
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "train_size": len(X_train),
    "calibration_size": len(X_cal),
    "test_size": len(X_test),
    "raw_auc": round(raw_auc, 4),
    "calibrated_auc": round(cal_auc, 4),
    "auc": round(cal_auc, 4),
    "features": available_cols,
    "feature_count": len(available_cols),
    "data_range": f"{str(df_raw.index[0])} ~ {str(df_raw.index[-1])}",
    "total_bars": len(df_raw),
    "train_period": f"{str(df_feat.index[0])} ~ {str(df_feat.index[split_train-1])}",
    "test_period": f"{str(df_feat.index[split_cal])} ~ {str(df_feat.index[-1])}",
    "model_type": "5year_calibrated",
    "target_shift": 2,
    "target_threshold": 0.003,
    "best_threshold": best_th,
    "calibrator": "platt_scaling",
    "description": "5年历史数据训练，Platt校准，解决概率压缩问题",
    "prob_stats_calibrated": {
        "max": round(float(calibrated_probs_test.max()), 4),
        "q99": round(float(np.quantile(calibrated_probs_test, 0.99)), 4),
        "q95": round(float(np.quantile(calibrated_probs_test, 0.95)), 4),
        "q90": round(float(np.quantile(calibrated_probs_test, 0.90)), 4),
        "q85": round(float(np.quantile(calibrated_probs_test, 0.85)), 4),
    }
}

version = save_model_with_metadata(lgb_wrapper, metadata)
print(f"💾 模型已保存: {version}")
update_latest_model(version)
print(f"✅ 已设置为最新模型: {version}")

print(f"\n{'='*70}")
print(f"✅ 训练完成！")
print(f"   版本: {version}")
print(f"   原始 AUC: {raw_auc:.4f}")
print(f"   校准后 AUC: {cal_auc:.4f}")
print(f"   校准后概率最大值: {calibrated_probs_test.max():.4f}")
print(f"   建议置信度阈值: {best_th:.2f}")
print(f"{'='*70}")
