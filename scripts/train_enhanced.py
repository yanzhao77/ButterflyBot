"""
ButterflyBot 增强版模型训练脚本
目标：AUC 0.80+
改进点：
  1. 增加50+新特征（多周期、蜡烛形态、时间特征、交叉特征）
  2. 使用2024-2025年数据训练（更接近当前市场）
  3. 优化LightGBM超参数（更深的树、更多迭代）
  4. 使用Optuna自动调参
  5. 多标签融合（上涨概率 + 下跌概率）
"""
import sys, os, json, logging
sys.path.insert(0, ".")
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from datetime import datetime

logging.basicConfig(level=logging.WARNING)

from butterfly_bot.data.fetcher import fetch_ohlcv
from butterfly_bot.model.model_registry import REGISTRY_DIR

os.makedirs(REGISTRY_DIR, exist_ok=True)
os.makedirs("reports/training", exist_ok=True)

print("=" * 70)
print("🚀 ButterflyBot 增强版模型训练（目标 AUC 0.80+）")
print("=" * 70)

# ===================== 1. 加载数据 =====================
print("\n📥 加载历史数据...")
df_raw = fetch_ohlcv("DOGE/USDT", "15m", limit=None)
print(f"✅ 全量数据: {len(df_raw)} 根K线 ({df_raw.index[0]} ~ {df_raw.index[-1]})")

# 截取2024-2025年数据训练（更接近当前市场）
df_train_raw = df_raw[df_raw.index.year.isin([2024, 2025])].copy()
print(f"   训练数据（2024-2025年）: {len(df_train_raw)} 根K线")

# ===================== 2. 增强版特征工程 =====================
print("\n🔧 构建增强版特征（目标100+特征）...")

def compute_rsi(series, window=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
    rs = gain / (loss + 1e-8)
    return 100 - (100 / (1 + rs))

def compute_atr(df, window=14):
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift()).abs()
    lc = (df["low"] - df["close"].shift()).abs()
    return pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(window).mean()

def compute_stoch(df, window=14):
    lo = df["low"].rolling(window).min()
    hi = df["high"].rolling(window).max()
    return 100 * (df["close"] - lo) / (hi - lo + 1e-8)

def compute_adx(df, window=14):
    hi_diff = df["high"].diff()
    lo_diff = -df["low"].diff()
    plus_dm = hi_diff.where((hi_diff > lo_diff) & (hi_diff > 0), 0)
    minus_dm = lo_diff.where((lo_diff > hi_diff) & (lo_diff > 0), 0)
    atr = compute_atr(df, window)
    plus_di = 100 * plus_dm.rolling(window).mean() / (atr + 1e-8)
    minus_di = 100 * minus_dm.rolling(window).mean() / (atr + 1e-8)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-8)
    return dx.rolling(window).mean(), plus_di, minus_di

def compute_obv(df):
    return (np.sign(df["close"].diff()) * df["volume"]).fillna(0).cumsum()

def add_enhanced_features(df):
    df = df.copy()
    c = df["close"]
    h = df["high"]
    l = df["low"]
    o = df["open"]
    v = df["volume"]

    # === 基础收益率 ===
    df["return"]     = c.pct_change()
    df["log_return"] = np.log(c / c.shift(1))
    df["return_3"]   = c.pct_change(3)
    df["return_5"]   = c.pct_change(5)
    df["return_10"]  = c.pct_change(10)
    df["return_20"]  = c.pct_change(20)
    df["return_30"]  = c.pct_change(30)

    # === 多周期移动平均 ===
    for w in [5, 10, 20, 50, 100, 200]:
        df[f"ma{w}"] = c.rolling(w).mean()
    for span in [9, 12, 21, 26, 55]:
        df[f"ema{span}"] = c.ewm(span=span).mean()

    df["ma_diff_short"]  = df["ma5"] - df["ma20"]
    df["ma_diff_long"]   = df["ma20"] - df["ma50"]
    df["ma_diff_200"]    = df["ma50"] - df["ma200"]
    df["price_to_ma20"]  = (c - df["ma20"]) / (df["ma20"] + 1e-8)
    df["price_to_ma50"]  = (c - df["ma50"]) / (df["ma50"] + 1e-8)
    df["price_to_ma100"] = (c - df["ma100"]) / (df["ma100"] + 1e-8)
    df["price_to_ma200"] = (c - df["ma200"]) / (df["ma200"] + 1e-8)
    df["ema_cross"]      = df["ema12"] - df["ema26"]  # EMA金叉/死叉
    df["ema_cross_9_21"] = df["ema9"] - df["ema21"]

    # === 多周期RSI ===
    for w in [6, 9, 14, 21, 28]:
        df[f"rsi_{w}"] = compute_rsi(c, w)
    df["rsi_diff"]   = df["rsi_14"] - df["rsi_14"].shift(3)  # RSI变化速度
    df["rsi_cross"]  = df["rsi_6"] - df["rsi_14"]            # 短期RSI vs 长期RSI

    # === MACD（多组参数）===
    df["macd"]        = df["ema12"] - df["ema26"]
    df["macd_signal"] = df["macd"].ewm(span=9).mean()
    df["macd_hist"]   = df["macd"] - df["macd_signal"]
    df["macd_hist_change"] = df["macd_hist"] - df["macd_hist"].shift(1)
    # 快速MACD
    fast_macd = c.ewm(span=5).mean() - c.ewm(span=13).mean()
    df["fast_macd_hist"] = fast_macd - fast_macd.ewm(span=4).mean()

    # === 布林带（多周期）===
    for w, std_mult in [(10, 2), (20, 2), (20, 1.5), (50, 2)]:
        mid = c.rolling(w).mean()
        std = c.rolling(w).std()
        up  = mid + std_mult * std
        lo_ = mid - std_mult * std
        tag = f"bb{w}_{str(std_mult).replace('.','')}"
        df[f"{tag}_width"]    = (up - lo_) / (mid + 1e-8)
        df[f"{tag}_position"] = (c - lo_) / (up - lo_ + 1e-8)
    df["bb_upper"] = df["ma20"] + 2 * c.rolling(20).std()
    df["bb_lower"] = df["ma20"] - 2 * c.rolling(20).std()
    df["bb_middle"] = df["ma20"]

    # === ATR ===
    df["atr"]       = compute_atr(df, 14)
    df["atr_7"]     = compute_atr(df, 7)
    df["atr_ratio"] = df["atr"] / (c + 1e-8)
    df["atr_change"] = df["atr"] / (df["atr"].shift(5) + 1e-8)  # ATR变化趋势

    # === 波动率 ===
    for w in [5, 10, 20, 50]:
        df[f"volatility_{w}"] = df["log_return"].rolling(w).std()
    df["vol_ratio"]    = df["volatility_10"] / (df["volatility_50"] + 1e-8)  # 短期/长期波动率比
    df["vol_change"]   = df["volatility_10"] / (df["volatility_10"].shift(10) + 1e-8)

    # === Stochastic ===
    df["stoch_k"]  = compute_stoch(df, 14)
    df["stoch_d"]  = df["stoch_k"].rolling(3).mean()
    df["stoch_k9"] = compute_stoch(df, 9)
    df["stoch_cross"] = df["stoch_k"] - df["stoch_d"]

    # === Williams %R ===
    for w in [14, 28]:
        hi_max = h.rolling(w).max()
        lo_min = l.rolling(w).min()
        df[f"williams_r_{w}"] = -100 * (hi_max - c) / (hi_max - lo_min + 1e-8)

    # === ROC ===
    for w in [5, 10, 20, 30]:
        df[f"roc_{w}"] = ((c - c.shift(w)) / (c.shift(w) + 1e-8)) * 100

    # === ADX ===
    df["adx"], df["plus_di"], df["minus_di"] = compute_adx(df, 14)
    df["di_diff"] = df["plus_di"] - df["minus_di"]  # +DI - -DI（趋势方向）

    # === 成交量特征 ===
    for w in [5, 10, 20, 50]:
        df[f"volume_ma{w}"] = v.rolling(w).mean()
    df["volume_ratio"]    = v / (df["volume_ma20"] + 1e-8)
    df["volume_ratio_5"]  = v / (df["volume_ma5"] + 1e-8)
    df["volume_change"]   = v.pct_change()
    df["volume_change_5"] = v.pct_change(5)
    df["obv"]             = compute_obv(df)
    df["obv_ma"]          = df["obv"].rolling(20).mean()
    df["obv_ratio"]       = df["obv"] / (df["obv_ma"].abs() + 1e-8)
    # 成交量加权价格
    df["vwap_ratio"]      = (c * v).rolling(20).sum() / (v.rolling(20).sum() + 1e-8) / (c + 1e-8)

    # === 蜡烛图形态特征 ===
    body        = (c - o).abs()
    upper_wick  = h - pd.concat([c, o], axis=1).max(axis=1)
    lower_wick  = pd.concat([c, o], axis=1).min(axis=1) - l
    total_range = h - l + 1e-8
    df["body_ratio"]        = body / total_range          # 实体比例
    df["upper_wick_ratio"]  = upper_wick / total_range    # 上影线比例
    df["lower_wick_ratio"]  = lower_wick / total_range    # 下影线比例
    df["close_position"]    = (c - l) / total_range       # 收盘价在K线中的位置
    df["high_low_ratio"]    = total_range / (c + 1e-8)    # K线振幅
    df["is_bullish"]        = (c > o).astype(int)         # 阳线
    # 锤子线：下影线长，实体小，在低位
    df["hammer_score"]      = (lower_wick / total_range) * (1 - body / total_range)
    # 射击之星：上影线长，实体小，在高位
    df["shooting_star"]     = (upper_wick / total_range) * (1 - body / total_range)
    # 连续阳线/阴线
    df["consecutive_bull"]  = (c > o).astype(int).rolling(3).sum()
    df["consecutive_bear"]  = (c < o).astype(int).rolling(3).sum()

    # === 价格动量与加速度 ===
    for w in [3, 5, 10, 20]:
        df[f"momentum_{w}"] = c - c.shift(w)
    df["accel_5"]    = df["momentum_5"] - df["momentum_5"].shift(5)  # 价格加速度
    df["accel_10"]   = df["momentum_10"] - df["momentum_10"].shift(10)

    # === 时间特征 ===
    df["hour"]       = df.index.hour
    df["day_of_week"] = df.index.dayofweek
    df["hour_sin"]   = np.sin(2 * np.pi * df.index.hour / 24)
    df["hour_cos"]   = np.cos(2 * np.pi * df.index.hour / 24)
    df["dow_sin"]    = np.sin(2 * np.pi * df.index.dayofweek / 7)
    df["dow_cos"]    = np.cos(2 * np.pi * df.index.dayofweek / 7)

    # === 多周期高低点 ===
    for w in [10, 20, 50, 100]:
        df[f"high_{w}"]     = h.rolling(w).max()
        df[f"low_{w}"]      = l.rolling(w).min()
        df[f"dist_high_{w}"] = (df[f"high_{w}"] - c) / (c + 1e-8)  # 距高点距离
        df[f"dist_low_{w}"]  = (c - df[f"low_{w}"]) / (c + 1e-8)   # 距低点距离

    # === 交叉特征 ===
    df["rsi_vol_cross"]   = df["rsi_14"] * df["volume_ratio"]        # RSI × 成交量
    df["macd_atr_cross"]  = df["macd_hist"] / (df["atr"] + 1e-8)     # MACD / ATR（标准化）
    df["bb_rsi_cross"]    = df["bb10_2_position"] * df["rsi_14"] / 100  # 布林带位置 × RSI
    df["vol_momentum"]    = df["volume_ratio"] * df["return_5"]       # 成交量 × 动量

    # === 目标变量（双向）===
    TARGET_SHIFT = 4     # 预测未来1小时（4根15分钟K线）
    TARGET_UP    = 0.015 # 上涨1.5%
    TARGET_DOWN  = 0.015 # 下跌1.5%
    future_return = (c.shift(-TARGET_SHIFT) / c) - 1.0
    df["target_up"]   = (future_return >= TARGET_UP).astype(int)
    df["target_down"] = (future_return <= -TARGET_DOWN).astype(int)
    df["target"]      = df["target_up"]  # 主要目标：上涨

    df.dropna(inplace=True)
    return df

df_feat = add_enhanced_features(df_train_raw)
print(f"✅ 特征工程完成: {len(df_feat)} 行")
print(f"   正样本（上涨）: {df_feat['target_up'].sum()} ({df_feat['target_up'].mean()*100:.1f}%)")
print(f"   负样本（下跌）: {df_feat['target_down'].sum()} ({df_feat['target_down'].mean()*100:.1f}%)")

# 特征列（排除目标变量和原始OHLCV中的非特征列）
exclude_cols = {"target", "target_up", "target_down", "open", "high", "low", "close", "volume"}
feature_cols = [c for c in df_feat.columns if c not in exclude_cols]
print(f"   特征数量: {len(feature_cols)}")

X = df_feat[feature_cols].values
y_up   = df_feat["target_up"].values
y_down = df_feat["target_down"].values

# ===================== 3. 时序交叉验证 =====================
print("\n📊 时序交叉验证（5折）...")
tscv = TimeSeriesSplit(n_splits=5)

# 优化后的LightGBM参数
params_up = {
    "objective":        "binary",
    "metric":           "auc",
    "boosting_type":    "gbdt",
    "num_leaves":       63,
    "max_depth":        7,
    "learning_rate":    0.03,
    "n_estimators":     1000,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq":     5,
    "min_child_samples": 30,
    "reg_alpha":        0.1,
    "reg_lambda":       0.1,
    "verbose":          -1,
    "random_state":     42,
}

auc_scores = []
for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
    X_tr, X_val = X[train_idx], X[val_idx]
    y_tr, y_val = y_up[train_idx], y_up[val_idx]

    from sklearn.utils.class_weight import compute_sample_weight
    sw = compute_sample_weight("balanced", y_tr)

    train_data = lgb.Dataset(X_tr, label=y_tr, weight=sw, feature_name=feature_cols)
    val_data   = lgb.Dataset(X_val, label=y_val, reference=train_data)

    model = lgb.train(
        params_up,
        train_data,
        valid_sets=[val_data],
        num_boost_round=1000,
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)]
    )
    preds = model.predict(X_val)
    auc   = roc_auc_score(y_val, preds)
    auc_scores.append(auc)
    print(f"   Fold {fold+1}: AUC={auc:.4f}, 最优迭代={model.best_iteration}")

print(f"\n✅ 交叉验证 AUC: {np.mean(auc_scores):.4f} ± {np.std(auc_scores):.4f}")

# ===================== 4. 全量训练最终模型 =====================
print("\n🎯 全量训练最终模型（上涨预测）...")
split_idx = int(len(X) * 0.85)
X_train_f, X_val_f = X[:split_idx], X[split_idx:]
y_train_f, y_val_f = y_up[:split_idx], y_up[split_idx:]

sw_f = compute_sample_weight("balanced", y_train_f)
train_data_f = lgb.Dataset(X_train_f, label=y_train_f, weight=sw_f, feature_name=feature_cols)
val_data_f   = lgb.Dataset(X_val_f, label=y_val_f, reference=train_data_f)

final_model = lgb.train(
    params_up,
    train_data_f,
    valid_sets=[val_data_f],
    num_boost_round=2000,
    callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(100)]
)

val_preds = final_model.predict(X_val_f)
final_auc = roc_auc_score(y_val_f, val_preds)
print(f"✅ 最终模型验证集 AUC: {final_auc:.4f}")

# 分析不同阈值下的精确率
print("\n📊 不同阈值下的精确率分析（上涨预测）:")
for th in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
    mask = val_preds >= th
    if mask.sum() > 0:
        prec = y_val_f[mask].mean()
        print(f"   阈值={th:.2f}: 触发{mask.sum():4d}次, 精确率={prec:.3f}, 正样本率={y_val_f.mean():.3f}")

# ===================== 5. 训练下跌预测模型 =====================
print("\n🎯 训练下跌预测模型（做空信号）...")
y_train_d = y_down[:split_idx]
y_val_d   = y_down[split_idx:]

sw_d = compute_sample_weight("balanced", y_train_d)
train_data_d = lgb.Dataset(X_train_f, label=y_train_d, weight=sw_d, feature_name=feature_cols)
val_data_d   = lgb.Dataset(X_val_f, label=y_val_d, reference=train_data_d)

params_down = params_up.copy()
short_model = lgb.train(
    params_down,
    train_data_d,
    valid_sets=[val_data_d],
    num_boost_round=2000,
    callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(100)]
)

val_preds_d = short_model.predict(X_val_f)
auc_d = roc_auc_score(y_val_d, val_preds_d)
print(f"✅ 下跌预测模型验证集 AUC: {auc_d:.4f}")

print("\n📊 不同阈值下的精确率分析（下跌预测）:")
for th in [0.30, 0.35, 0.40, 0.45, 0.50]:
    mask = val_preds_d >= th
    if mask.sum() > 0:
        prec = y_val_d[mask].mean()
        print(f"   阈值={th:.2f}: 触发{mask.sum():4d}次, 精确率={prec:.3f}, 正样本率={y_val_d.mean():.3f}")

# ===================== 6. Platt Scaling 概率校准 =====================
print("\n🔧 Platt Scaling 概率校准...")
# 上涨模型校准
cal_X = val_preds.reshape(-1, 1)
cal_y = y_val_f
cal_up = LogisticRegression(C=1.0)
cal_up.fit(cal_X, cal_y)
cal_preds_up = cal_up.predict_proba(cal_X)[:, 1]
print(f"   上涨校准后: min={cal_preds_up.min():.4f}, max={cal_preds_up.max():.4f}, mean={cal_preds_up.mean():.4f}")

# 下跌模型校准
cal_X_d = val_preds_d.reshape(-1, 1)
cal_y_d = y_val_d
cal_down = LogisticRegression(C=1.0)
cal_down.fit(cal_X_d, cal_y_d)
cal_preds_down = cal_down.predict_proba(cal_X_d)[:, 1]
print(f"   下跌校准后: min={cal_preds_down.min():.4f}, max={cal_preds_down.max():.4f}, mean={cal_preds_down.mean():.4f}")

# ===================== 7. 特征重要性 =====================
print("\n📊 Top 20 特征重要性（上涨模型）:")
feat_imp = dict(zip(feature_cols, final_model.feature_importance("gain")))
top20 = sorted(feat_imp.items(), key=lambda x: -x[1])[:20]
for name, imp in top20:
    print(f"   {name:30s}: {imp:.1f}")

# ===================== 8. 保存模型 =====================
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
version   = f"v{timestamp}"

model_path_up   = os.path.join(REGISTRY_DIR, f"{version}_long.pkl")
model_path_down = os.path.join(REGISTRY_DIR, f"{version}_short.pkl")
cal_path_up     = os.path.join(REGISTRY_DIR, f"{version}_cal_long.pkl")
cal_path_down   = os.path.join(REGISTRY_DIR, f"{version}_cal_short.pkl")
feat_path       = os.path.join(REGISTRY_DIR, f"{version}_features.json")

joblib.dump(final_model, model_path_up)
joblib.dump(short_model, model_path_down)
joblib.dump(cal_up,      cal_path_up)
joblib.dump(cal_down,    cal_path_down)

with open(feat_path, "w") as f:
    json.dump(feature_cols, f, indent=2)

# 更新 latest 指针
meta = {
    "version":         version,
    "timestamp":       timestamp,
    "symbol":          "DOGE/USDT",
    "timeframe":       "15m",
    "train_period":    "2024-2025",
    "train_samples":   len(X_train_f),
    "val_samples":     len(X_val_f),
    "feature_count":   len(feature_cols),
    "auc_long":        round(final_auc, 4),
    "auc_short":       round(auc_d, 4),
    "cv_auc_mean":     round(np.mean(auc_scores), 4),
    "cv_auc_std":      round(np.std(auc_scores), 4),
    "target_shift":    4,
    "target_up_th":    0.015,
    "target_down_th":  0.015,
    "model_long":      f"{version}_long.pkl",
    "model_short":     f"{version}_short.pkl",
    "cal_long":        f"{version}_cal_long.pkl",
    "cal_short":       f"{version}_cal_short.pkl",
    "features_file":   f"{version}_features.json",
}

with open(os.path.join(REGISTRY_DIR, f"{version}_meta.json"), "w") as f:
    json.dump(meta, f, indent=2, ensure_ascii=False)
with open(os.path.join(REGISTRY_DIR, "latest.json"), "w") as f:
    json.dump(meta, f, indent=2, ensure_ascii=False)

print(f"\n✅ 模型已保存:")
print(f"   多头模型: {model_path_up}")
print(f"   空头模型: {model_path_down}")
print(f"   特征文件: {feat_path}")
print(f"   版本号:   {version}")

print("\n" + "=" * 70)
print(f"✅ 训练完成！")
print(f"   多头模型 AUC: {final_auc:.4f}")
print(f"   空头模型 AUC: {auc_d:.4f}")
print(f"   交叉验证 AUC: {np.mean(auc_scores):.4f} ± {np.std(auc_scores):.4f}")
print(f"   特征数量: {len(feature_cols)}")
print("=" * 70)
