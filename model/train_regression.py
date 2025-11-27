#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练回归模型预测未来收益率
用于双向交易策略
"""

import sys
import os
import argparse
from datetime import datetime, timezone, timedelta
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import SYMBOL, TIMEFRAME, BASE_PATH
from data.fetcher import fetch_ohlcv
from data.features import add_features

# 预测参数
PREDICTION_WINDOW = 4  # 预测未来4根K线的收益率
MIN_RETURN_THRESHOLD = 0.015  # 最小收益率阈值1.5%

def prepare_regression_data(df, prediction_window=PREDICTION_WINDOW):
    """
    准备回归训练数据
    
    目标：预测未来N根K线的收益率
    """
    print(f"📊 准备回归数据...")
    print(f"  预测窗口: {prediction_window}根K线")
    
    # 计算未来收益率
    df['future_return'] = (df['close'].shift(-prediction_window) - df['close']) / df['close']
    
    # 删除无法计算未来收益的行
    df = df.dropna(subset=['future_return'])
    
    # 统计信息
    positive_samples = (df['future_return'] > MIN_RETURN_THRESHOLD).sum()
    negative_samples = (df['future_return'] < -MIN_RETURN_THRESHOLD).sum()
    neutral_samples = len(df) - positive_samples - negative_samples
    
    print(f"\n  样本分布:")
    print(f"    上涨样本 (>{MIN_RETURN_THRESHOLD*100:.1f}%): {positive_samples} ({positive_samples/len(df)*100:.1f}%)")
    print(f"    下跌样本 (<-{MIN_RETURN_THRESHOLD*100:.1f}%): {negative_samples} ({negative_samples/len(df)*100:.1f}%)")
    print(f"    震荡样本: {neutral_samples} ({neutral_samples/len(df)*100:.1f}%)")
    
    print(f"\n  收益率统计:")
    print(f"    均值: {df['future_return'].mean()*100:.3f}%")
    print(f"    标准差: {df['future_return'].std()*100:.3f}%")
    print(f"    最小值: {df['future_return'].min()*100:.3f}%")
    print(f"    最大值: {df['future_return'].max()*100:.3f}%")
    print(f"    中位数: {df['future_return'].median()*100:.3f}%")
    
    return df

def train_regression_model(limit=35000, since_days=365):
    """训练回归模型"""
    
    print("=" * 80)
    print("训练回归模型 - 预测未来收益率")
    print("=" * 80)
    print(f"\n配置:")
    print(f"  交易对: {SYMBOL}")
    print(f"  周期: {TIMEFRAME}")
    print(f"  数据量: {limit}条")
    print(f"  时间范围: 最近{since_days}天")
    
    # 获取数据
    print(f"\n⏳ 获取历史数据...")
    since_date = datetime.now(timezone.utc) - timedelta(days=since_days)
    df = fetch_ohlcv(SYMBOL, TIMEFRAME, limit=limit, since=since_date)
    
    if df is None or df.empty:
        print("❌ 数据获取失败")
        return None
    
    print(f"✅ 获取 {len(df)} 根K线")
    
    # 添加特征
    print(f"\n🧩 构建特征...")
    df = add_features(df)
    
    if df.empty:
        print("❌ 特征构建失败")
        return None
    
    # 准备回归数据
    df = prepare_regression_data(df, PREDICTION_WINDOW)
    
    # 特征列
    feature_cols = [
        'open', 'high', 'low', 'close', 'volume',
        'return', 'log_return',
        'ma20', 'ma50', 'ma_diff',
        'rsi', 'macd', 'macd_signal', 'macd_hist',
        'volatility', 'volume_ratio'
    ]
    
    # 检查特征
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        print(f"❌ 缺少特征: {missing_cols}")
        return None
    
    # 删除包含NaN的行
    df = df.dropna(subset=feature_cols + ['future_return'])
    
    print(f"\n🧩 特征维度: {len(feature_cols)} | 有效样本: {len(df)}")
    
    # 准备训练数据
    X = df[feature_cols].values
    y = df['future_return'].values
    
    # 分割训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, shuffle=False
    )
    
    print(f"📊 训练集: {len(X_train)} | 测试集: {len(X_test)}")
    
    # 训练LightGBM回归模型
    print(f"\n🚀 训练LightGBM回归模型...")
    
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'seed': 42
    }
    
    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[test_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=100)
        ]
    )
    
    # 评估模型
    print(f"\n📈 评估模型性能...")
    
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # 训练集指标
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    train_mae = mean_absolute_error(y_train, y_pred_train)
    train_r2 = r2_score(y_train, y_pred_train)
    
    # 测试集指标
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_mae = mean_absolute_error(y_test, y_pred_test)
    test_r2 = r2_score(y_test, y_pred_test)
    
    print(f"\n训练集:")
    print(f"  RMSE: {train_rmse*100:.3f}%")
    print(f"  MAE: {train_mae*100:.3f}%")
    print(f"  R²: {train_r2:.4f}")
    
    print(f"\n测试集:")
    print(f"  RMSE: {test_rmse*100:.3f}%")
    print(f"  MAE: {test_mae*100:.3f}%")
    print(f"  R²: {test_r2:.4f}")
    
    # 方向准确率（预测涨跌方向是否正确）
    train_direction_acc = np.mean((y_pred_train > 0) == (y_train > 0))
    test_direction_acc = np.mean((y_pred_test > 0) == (y_test > 0))
    
    print(f"\n方向准确率:")
    print(f"  训练集: {train_direction_acc*100:.2f}%")
    print(f"  测试集: {test_direction_acc*100:.2f}%")
    
    # 特征重要性
    print(f"\n📊 特征重要性 (Top 10):")
    importance = model.feature_importance(importance_type='gain')
    feature_importance = sorted(
        zip(feature_cols, importance),
        key=lambda x: x[1],
        reverse=True
    )
    for i, (feat, imp) in enumerate(feature_importance[:10], 1):
        print(f"  {i}. {feat}: {imp:.0f}")
    
    # 保存模型
    model_dir = BASE_PATH / 'models' / 'registry'
    model_dir.mkdir(parents=True, exist_ok=True)
    
    version = datetime.now().strftime('v%Y%m%d_%H%M')
    model_path = model_dir / f'{version}_regression.pkl'
    metadata_path = model_dir / f'{version}_regression.json'
    
    joblib.dump(model, model_path)
    
    metadata = {
        'version': version,
        'type': 'regression',
        'symbol': SYMBOL,
        'timeframe': TIMEFRAME,
        'prediction_window': PREDICTION_WINDOW,
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'test_rmse': float(test_rmse),
        'test_mae': float(test_mae),
        'test_r2': float(test_r2),
        'test_direction_acc': float(test_direction_acc),
        'features': feature_cols,
        'created_at': datetime.now(timezone.utc).isoformat()
    }
    
    import json
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # 更新latest指针
    latest_path = model_dir / 'latest_regression.txt'
    with open(latest_path, 'w') as f:
        f.write(version)
    
    print(f"\n💾 模型已保存:")
    print(f"  版本: {version}")
    print(f"  路径: {model_path}")
    print(f"  元数据: {metadata_path}")
    
    print(f"\n✅ 训练成功!")
    print(f"  测试集R²: {test_r2:.4f}")
    print(f"  方向准确率: {test_direction_acc*100:.2f}%")
    
    return model_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='训练回归模型')
    parser.add_argument('--limit', type=int, default=35000, help='数据量')
    parser.add_argument('--since_days', type=int, default=365, help='时间范围（天）')
    
    args = parser.parse_args()
    
    model_path = train_regression_model(
        limit=args.limit,
        since_days=args.since_days
    )
    
    sys.exit(0 if model_path else 1)
