# train.py
"""
量化模型训练入口（V2+ 融合版）：
- 工程化结构（argparse + config）
- 健壮性检查（数据量、目标变量、AUC容错）
- 验证集训练 + 自动注册最优模型
"""

import argparse
import os

from sklearn.metrics import roc_auc_score

from config.settings import SYMBOL, TIMEFRAME, TRAIN_TEST_SPLIT_RATIO, REGISTRY_DIR
from data.features import add_features, get_feature_columns
from data.fetcher import fetch_ohlcv
from model.lgb_model import LGBModel
from model.model_registry import (
    save_model_with_metadata,
    find_best_model_by_auc,
    update_latest_model
)


def main(symbol: str, timeframe: str, limit: int = 2000, since_days: int = None):
    print(f"🔧 开始训练模型 | 交易对: {symbol} | 周期: {timeframe} | K线数: {limit}")

    # === 1. 获取原始数据 ===
    since = None
    if since_days is not None:
        from datetime import datetime, timedelta
        dt_since = datetime.utcnow() - timedelta(days=since_days)
        since = int(dt_since.timestamp() * 1000)
        print(f"⏳ 拉取自 {dt_since.strftime('%Y-%m-%d')} 以来的所有K线数据")
    df_raw = fetch_ohlcv(symbol=symbol, timeframe=timeframe, limit=limit, since=since)

    # --- 健壮性检查：数据量 ---
    if len(df_raw) < 200:
        raise ValueError(f"❌ 数据量不足（仅 {len(df_raw)} 根K线），至少需要 200 根")

    print(f"✅ 获取 {len(df_raw)} 根K线，正在构建特征...")

    # === 2. 构建特征 ===
    df_feat = add_features(df_raw)

    # --- 健壮性检查：特征工程后是否为空 ---
    if len(df_feat) == 0:
        raise ValueError("❌ 特征工程后无有效数据（可能全为 NaN）")

    # --- 健壮性检查：目标变量有效性 ---
    if "target" not in df_feat.columns:
        raise ValueError("❌ 缺少目标变量 'target'，请检查 data/features.py")

    y = df_feat["target"]
    if y.nunique() < 2:
        raise ValueError("❌ 目标变量无变化（全涨或全跌），无法训练分类模型")

    feature_cols = get_feature_columns()
    missing_features = set(feature_cols) - set(df_feat.columns)
    if missing_features:
        raise ValueError(f"❌ 缺少特征列: {missing_features}")

    print(f"🧩 特征维度: {len(feature_cols)} | 有效样本: {len(df_feat)}")

    # === 3. 准备训练数据（时序分割）===
    X = df_feat[feature_cols]
    split_idx = int(len(X) * TRAIN_TEST_SPLIT_RATIO)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    if len(X_train) == 0 or len(X_test) == 0:
        raise ValueError("❌ 训练集或测试集为空，请增加数据量或调整 split_ratio")

    print(f"📊 训练集: {len(X_train)} | 测试集: {len(X_test)}")

    # === 4. 训练模型（带验证集）===
    model = LGBModel()
    model.train(X_train, y_train, X_val=X_test, y_val=y_test)

    # === 5. 评估 AUC（带容错）===
    y_pred_proba = model.predict(X_test)
    try:
        auc = float(roc_auc_score(y_test, y_pred_proba))
    except ValueError as e:
        print(f"⚠️ AUC 计算失败: {e}，使用默认值 0.5")
        auc = 0.5

    print(f"📈 测试集 AUC: {auc:.4f}")

    # === 6. 保存模型与元数据 ===
    metadata = {
        "symbol": symbol,
        "timeframe": timeframe,
        "timestamp": datetime.now().isoformat(),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "auc": round(auc, 4),
        "features": feature_cols,
        "limit": limit,
        "split_ratio": TRAIN_TEST_SPLIT_RATIO
    }

    version = save_model_with_metadata(model, metadata)
    print(f"💾 模型已保存为版本: {version}")

    # === 7. 更新最优模型 ===
    try:
        best_version = find_best_model_by_auc()
        update_latest_model(best_version)
        print(f"🏆 当前最优模型: {best_version}")
    except Exception as e:
        print(f"⚠️ 无法更新最优模型: {e}")

    return version, auc


def train_and_evaluate(symbol: str = None, timeframe: str = None, limit: int = 2000, since_days: int = None):
    """向外暴露的便捷接口，兼容外部调用（如 API / 自动重训练）。

    若 symbol/timeframe 未提供则使用 config 中的默认值。
    返回 (version, auc)
    """
    from config.settings import SYMBOL as CFG_SYMBOL, TIMEFRAME as CFG_TIMEFRAME

    symbol = symbol or CFG_SYMBOL
    timeframe = timeframe or CFG_TIMEFRAME

    return main(symbol=symbol, timeframe=timeframe, limit=limit, since_days=since_days)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练量化交易模型（V2+ 融合版）")
    parser.add_argument("--symbol", type=str, default=SYMBOL, help="交易对，如 BTC/USDT")
    parser.add_argument("--timeframe", type=str, default=TIMEFRAME, help="K线周期，如 1h, 15m")
    parser.add_argument("--limit", type=int, default=100000, help="获取K线数量（建议 ≥1000）")
    parser.add_argument("--since_days", type=int, default=365, help="拉取过去 N 天的数据（如 365 表示一年）")

    args = parser.parse_args()

    # 确保模型注册目录存在
    os.makedirs(REGISTRY_DIR, exist_ok=True)
    try:
        version, auc = main(
            symbol=args.symbol,
            timeframe=args.timeframe,
            limit=args.limit,
            since_days=args.since_days
        )
        print(f"\n✅ 训练成功！版本: {version} | AUC: {auc:.4f}")
    except Exception as e:
        print(f"\n❌ 训练失败: {e}")
        exit(1)