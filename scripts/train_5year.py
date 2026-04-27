"""
ButterflyBot 5年历史数据模型训练脚本
使用 198,140 根 K 线（2020-04-28 ~ 2025-12-23）训练 LightGBM 模型
解决过拟合问题：覆盖牛市、熊市、横盘等多种市场状态
"""
import sys, os, json, logging, time
sys.path.insert(0, ".")
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import TimeSeriesSplit

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

from butterfly_bot.data.fetcher import fetch_ohlcv
from butterfly_bot.data.features import add_features, get_feature_columns
from butterfly_bot.model.lgb_model import LGBModel
from butterfly_bot.model.model_registry import (
    save_model_with_metadata,
    find_best_model_by_auc,
    update_latest_model,
)

print("=" * 70)
print("🚀 ButterflyBot 5年历史数据模型训练")
print("=" * 70)

# ===================== 1. 加载数据 =====================
print("\n📥 加载5年历史数据...")
t0 = time.time()
df_raw = fetch_ohlcv("DOGE/USDT", "15m", limit=None)
print(f"✅ 数据加载完成: {len(df_raw)} 根K线，耗时 {time.time()-t0:.1f}s")
print(f"   时间范围: {df_raw.index[0]} ~ {df_raw.index[-1]}")

# ===================== 2. 特征工程 =====================
print("\n🔧 构建特征...")
t1 = time.time()
df_feat = add_features(df_raw)
feature_cols = get_feature_columns()
available_cols = [c for c in feature_cols if c in df_feat.columns]
missing_cols = set(feature_cols) - set(df_feat.columns)
if missing_cols:
    print(f"⚠️  缺少特征列: {missing_cols}")

X = df_feat[available_cols]
y = df_feat["target"]
print(f"✅ 特征工程完成: {len(df_feat)} 样本，{len(available_cols)} 个特征，耗时 {time.time()-t1:.1f}s")
print(f"   目标变量分布: 上涨={y.sum()} ({y.mean()*100:.1f}%), 下跌/平={len(y)-y.sum()} ({(1-y.mean())*100:.1f}%)")

# ===================== 3. 时序分割 =====================
# 使用 70% 训练，30% 测试（时间顺序）
split_idx = int(len(X) * 0.70)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

print(f"\n📊 数据分割:")
print(f"   训练集: {len(X_train)} 样本 ({df_feat.index[0]} ~ {df_feat.index[split_idx-1]})")
print(f"   测试集: {len(X_test)} 样本 ({df_feat.index[split_idx]} ~ {df_feat.index[-1]})")
print(f"   训练集目标分布: 上涨={y_train.sum()} ({y_train.mean()*100:.1f}%)")
print(f"   测试集目标分布: 上涨={y_test.sum()} ({y_test.mean()*100:.1f}%)")

# ===================== 4. 训练模型 =====================
print("\n🤖 开始训练 LightGBM 模型...")
t2 = time.time()
model = LGBModel()
model.train(X_train, y_train, X_val=X_test, y_val=y_test)
train_time = time.time() - t2
print(f"✅ 模型训练完成，耗时 {train_time:.1f}s")

# ===================== 5. 评估 =====================
print("\n📈 模型评估...")
y_pred_proba = model.predict(X_test)
y_pred_binary = (y_pred_proba >= 0.5).astype(int)

try:
    auc = float(roc_auc_score(y_test, y_pred_proba))
except ValueError as e:
    print(f"⚠️ AUC 计算失败: {e}")
    auc = 0.5

print(f"   测试集 AUC: {auc:.4f}")
print(f"   预测概率统计: min={y_pred_proba.min():.4f}, max={y_pred_proba.max():.4f}, mean={y_pred_proba.mean():.4f}")
print(f"   85%分位数: {np.quantile(y_pred_proba, 0.85):.4f}")
print(f"   90%分位数: {np.quantile(y_pred_proba, 0.90):.4f}")
print(f"   95%分位数: {np.quantile(y_pred_proba, 0.95):.4f}")

# 分类报告
print("\n分类报告（阈值=0.50）:")
print(classification_report(y_test, y_pred_binary, target_names=["下跌/平", "上涨"]))

# 不同阈值下的精确率
print("不同置信度阈值下的精确率:")
for th in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
    mask = y_pred_proba >= th
    count = mask.sum()
    if count > 0:
        precision = y_test[mask].mean()
        print(f"   阈值≥{th:.2f}: {count}次触发, 精确率={precision:.2%}")
    else:
        print(f"   阈值≥{th:.2f}: 0次触发")

# 年度AUC分析（检验泛化能力）
print("\n年度 AUC 分析（检验泛化能力）:")
test_years = df_feat.index[split_idx:].year.unique()
for yr in sorted(test_years):
    yr_mask = df_feat.index[split_idx:].year == yr
    if yr_mask.sum() > 100:
        try:
            yr_auc = roc_auc_score(y_test[yr_mask], y_pred_proba[yr_mask])
            print(f"   {yr}年: AUC={yr_auc:.4f} ({yr_mask.sum()}样本)")
        except:
            pass

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
    "data_range": f"{df_raw.index[0]} ~ {df_raw.index[-1]}",
    "total_bars": len(df_raw),
    "train_period": f"{df_feat.index[0]} ~ {df_feat.index[split_idx-1]}",
    "test_period": f"{df_feat.index[split_idx]} ~ {df_feat.index[-1]}",
    "model_type": "5year_full_data",
    "description": "使用5年历史数据训练，覆盖牛市/熊市/横盘，解决过拟合问题",
}

version = save_model_with_metadata(model, metadata)
print(f"\n💾 模型已保存: {version}")

# 更新最优模型（如果AUC更好）
try:
    best_version = find_best_model_by_auc()
    update_latest_model(best_version)
    print(f"🏆 当前最优模型: {best_version}")
    # 读取最优模型的AUC
    best_json = f"models/registry/{best_version}.json"
    if os.path.exists(best_json):
        with open(best_json) as f:
            best_meta = json.load(f)
        print(f"   最优模型AUC: {best_meta.get('auc', 'N/A')}")
except Exception as e:
    print(f"⚠️ 更新最优模型失败: {e}")
    # 手动设置为最新版本
    update_latest_model(version)
    print(f"✅ 已手动设置最新模型: {version}")

print(f"\n✅ 训练完成！新模型版本: {version}")
print(f"   AUC: {auc:.4f}")
print(f"   训练样本: {len(X_train):,}")
print(f"   测试样本: {len(X_test):,}")
print("=" * 70)
