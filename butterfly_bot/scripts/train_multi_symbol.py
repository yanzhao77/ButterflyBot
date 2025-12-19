#!/usr/bin/env python3
"""
多币种模型训练脚本
使用多个交易对的数据训练模型，提升泛化能力
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timezone

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from sklearn.metrics import roc_auc_score, classification_report
from butterfly_bot.config.settings import (
    TIMEFRAME,
    TRAIN_TEST_SPLIT_RATIO,
)
from butterfly_bot.data.features import add_features, get_feature_columns
from butterfly_bot.data.fetcher import fetch_ohlcv
from butterfly_bot.model.lgb_model import LGBModel
from butterfly_bot.model.model_registry import save_model_with_metadata, update_latest_model


def main():
    """使用多个币种训练模型"""
    
    # 定义多个交易对
    symbols = ["DOGE/USDT", "BTC/USDT", "ETH/USDT"]
    timeframe = TIMEFRAME
    limit = 3000  # 每个币种获取3000根K线
    
    print(f"🔧 开始多币种模型训练")
    print(f"交易对: {', '.join(symbols)}")
    print(f"周期: {timeframe}")
    print(f"每个币种K线数: {limit}")
    print("=" * 60)
    
    # 收集所有币种的数据
    all_data = []
    
    for symbol in symbols:
        print(f"\n📥 获取 {symbol} 数据...")
        try:
            df_raw = fetch_ohlcv(symbol=symbol, timeframe=timeframe, limit=limit)
            
            if len(df_raw) < 200:
                print(f"⚠️  {symbol} 数据量不足（{len(df_raw)}条），跳过")
                continue
            
            print(f"✅ 获取 {len(df_raw)} 根K线")
            
            # 构建特征
            df_feat = add_features(df_raw)
            
            if len(df_feat) == 0:
                print(f"⚠️  {symbol} 特征工程后无有效数据，跳过")
                continue
            
            # 添加币种标识（可选，用于分析）
            df_feat['symbol'] = symbol
            
            all_data.append(df_feat)
            print(f"✅ {symbol} 有效样本: {len(df_feat)}")
            
        except Exception as e:
            print(f"❌ {symbol} 数据获取失败: {e}")
            continue
    
    if len(all_data) == 0:
        raise ValueError("❌ 没有有效的训练数据")
    
    # 合并所有数据
    print(f"\n🔄 合并 {len(all_data)} 个币种的数据...")
    df_combined = pd.concat(all_data, ignore_index=True)
    
    # 移除symbol列（不用于训练）
    if 'symbol' in df_combined.columns:
        df_combined = df_combined.drop('symbol', axis=1)
    
    print(f"✅ 合并后总样本数: {len(df_combined)}")
    
    # 检查目标变量
    if "target" not in df_combined.columns:
        raise ValueError("❌ 缺少目标变量 'target'")
    
    y = df_combined["target"]
    if y.nunique() < 2:
        raise ValueError("❌ 目标变量无变化，无法训练分类模型")
    
    # 准备特征
    feature_cols = get_feature_columns()
    missing_features = set(feature_cols) - set(df_combined.columns)
    if missing_features:
        raise ValueError(f"❌ 缺少特征列: {missing_features}")
    
    X = df_combined[feature_cols]
    
    print(f"🧩 特征维度: {len(feature_cols)}")
    print(f"📊 正样本比例: {y.mean():.2%}")
    
    # 时序分割（保持时间顺序）
    split_idx = int(len(X) * TRAIN_TEST_SPLIT_RATIO)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    if len(X_train) == 0 or len(X_test) == 0:
        raise ValueError("❌ 训练集或测试集为空")
    
    print(f"📊 训练集: {len(X_train)} | 测试集: {len(X_test)}")
    print("=" * 60)
    
    # 训练模型
    print("\n🚀 开始训练模型...")
    model = LGBModel()
    model.train(X_train, y_train, X_val=X_test, y_val=y_test)
    
    # 评估模型
    print("\n📈 评估模型性能...")
    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    try:
        auc = float(roc_auc_score(y_test, y_pred_proba))
        print(f"✅ 测试集 AUC: {auc:.4f}")
    except ValueError as e:
        print(f"⚠️  AUC 计算失败: {e}")
        auc = 0.5
    
    # 打印分类报告
    print("\n📋 分类报告:")
    print(classification_report(y_test, y_pred, target_names=['下跌', '上涨']))
    
    # 保存模型
    print("\n💾 保存模型...")
    metadata = {
        "symbols": symbols,
        "timeframe": timeframe,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "auc": auc,
        "feature_count": len(feature_cols),
        "model_type": "multi_symbol",
    }
    
    version = save_model_with_metadata(model, metadata)
    print(f"✅ 模型已保存为版本: {version}")
    
    # 更新为最新模型
    update_latest_model(version, auc)
    print(f"✅ 已更新为最新模型")
    
    print("\n" + "=" * 60)
    print(f"🎉 多币种模型训练完成！")
    print(f"版本: {version}")
    print(f"AUC: {auc:.4f}")
    print(f"特征数: {len(feature_cols)}")
    print(f"训练样本: {len(X_train)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
