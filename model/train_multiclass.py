#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练三分类模型：做多 / 观望 / 做空
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
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import SYMBOL, TIMEFRAME, BASE_PATH
from data.fetcher import fetch_ohlcv
from data.features import add_features

# 预测参数
PREDICTION_WINDOW = 4  # 预测未来4根K线的收益率
LONG_THRESHOLD = 0.012  # 做多阈值：预期涨幅>1.2%
SHORT_THRESHOLD = -0.012  # 做空阈值：预期跌幅<-1.2%

def prepare_multiclass_data(df, prediction_window=PREDICTION_WINDOW):
    """
    准备三分类训练数据
    
    类别定义：
    - 0: 做空（未来收益率 < SHORT_THRESHOLD）
    - 1: 观望（SHORT_THRESHOLD <= 未来收益率 <= LONG_THRESHOLD）
    - 2: 做多（未来收益率 > LONG_THRESHOLD）
    """
    print(f"📊 准备三分类数据...")
    print(f"  预测窗口: {prediction_window}根K线")
    print(f"  做多阈值: >{LONG_THRESHOLD*100:.1f}%")
    print(f"  做空阈值: <{SHORT_THRESHOLD*100:.1f}%")
    
    # 计算未来收益率
    df['future_return'] = (df['close'].shift(-prediction_window) - df['close']) / df['close']
    
    # 定义类别
    def classify(ret):
        if ret > LONG_THRESHOLD:
            return 2  # 做多
        elif ret < SHORT_THRESHOLD:
            return 0  # 做空
        else:
            return 1  # 观望
    
    df['target'] = df['future_return'].apply(classify)
    
    # 删除无法计算的行
    df = df.dropna(subset=['future_return', 'target'])
    
    # 统计信息
    class_counts = df['target'].value_counts().sort_index()
    
    print(f"\n  样本分布:")
    print(f"    做空 (0): {class_counts.get(0, 0)} ({class_counts.get(0, 0)/len(df)*100:.1f}%)")
    print(f"    观望 (1): {class_counts.get(1, 0)} ({class_counts.get(1, 0)/len(df)*100:.1f}%)")
    print(f"    做多 (2): {class_counts.get(2, 0)} ({class_counts.get(2, 0)/len(df)*100:.1f}%)")
    
    print(f"\n  收益率统计:")
    for cls, label in [(0, '做空'), (1, '观望'), (2, '做多')]:
        if cls in class_counts.index:
            cls_returns = df[df['target'] == cls]['future_return']
            print(f"    {label}: 均值{cls_returns.mean()*100:.3f}%, "
                  f"标准差{cls_returns.std()*100:.3f}%")
    
    return df

def train_multiclass_model(limit=35000, since_days=365):
    """训练三分类模型"""
    
    print("=" * 80)
    print("训练三分类模型 - 做多/观望/做空")
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
    
    # 准备三分类数据
    df = prepare_multiclass_data(df, PREDICTION_WINDOW)
    
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
    df = df.dropna(subset=feature_cols + ['target'])
    
    print(f"\n🧩 特征维度: {len(feature_cols)} | 有效样本: {len(df)}")
    
    # 准备训练数据
    X = df[feature_cols].values
    y = df['target'].values
    
    # 分割训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, shuffle=False
    )
    
    print(f"📊 训练集: {len(X_train)} | 测试集: {len(X_test)}")
    
    # 训练LightGBM多分类模型
    print(f"\n🚀 训练LightGBM三分类模型...")
    
    params = {
        'objective': 'multiclass',
        'num_class': 3,
        'metric': 'multi_logloss',
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
    
    y_pred_train = model.predict(X_train).argmax(axis=1)
    y_pred_test = model.predict(X_test).argmax(axis=1)
    
    # 训练集准确率
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    
    print(f"\n准确率:")
    print(f"  训练集: {train_acc*100:.2f}%")
    print(f"  测试集: {test_acc*100:.2f}%")
    
    # 混淆矩阵
    print(f"\n混淆矩阵 (测试集):")
    cm = confusion_matrix(y_test, y_pred_test)
    print("           预测做空  预测观望  预测做多")
    print(f"实际做空    {cm[0][0]:6d}    {cm[0][1]:6d}    {cm[0][2]:6d}")
    print(f"实际观望    {cm[1][0]:6d}    {cm[1][1]:6d}    {cm[1][2]:6d}")
    print(f"实际做多    {cm[2][0]:6d}    {cm[2][1]:6d}    {cm[2][2]:6d}")
    
    # 分类报告
    print(f"\n分类报告 (测试集):")
    target_names = ['做空', '观望', '做多']
    print(classification_report(y_test, y_pred_test, target_names=target_names, zero_division=0))
    
    # 预测分布
    print(f"\n预测分布 (测试集):")
    pred_counts = pd.Series(y_pred_test).value_counts().sort_index()
    print(f"  做空: {pred_counts.get(0, 0)} ({pred_counts.get(0, 0)/len(y_pred_test)*100:.1f}%)")
    print(f"  观望: {pred_counts.get(1, 0)} ({pred_counts.get(1, 0)/len(y_pred_test)*100:.1f}%)")
    print(f"  做多: {pred_counts.get(2, 0)} ({pred_counts.get(2, 0)/len(y_pred_test)*100:.1f}%)")
    
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
    model_path = model_dir / f'{version}_multiclass.pkl'
    metadata_path = model_dir / f'{version}_multiclass.json'
    
    joblib.dump(model, model_path)
    
    metadata = {
        'version': version,
        'type': 'multiclass',
        'num_classes': 3,
        'class_labels': {0: '做空', 1: '观望', 2: '做多'},
        'symbol': SYMBOL,
        'timeframe': TIMEFRAME,
        'prediction_window': PREDICTION_WINDOW,
        'long_threshold': LONG_THRESHOLD,
        'short_threshold': SHORT_THRESHOLD,
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'test_accuracy': float(test_acc),
        'features': feature_cols,
        'created_at': datetime.now(timezone.utc).isoformat()
    }
    
    import json
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # 更新latest指针
    latest_path = model_dir / 'latest_multiclass.txt'
    with open(latest_path, 'w') as f:
        f.write(version)
    
    print(f"\n💾 模型已保存:")
    print(f"  版本: {version}")
    print(f"  路径: {model_path}")
    print(f"  元数据: {metadata_path}")
    
    print(f"\n✅ 训练成功!")
    print(f"  测试集准确率: {test_acc*100:.2f}%")
    
    return model_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='训练三分类模型')
    parser.add_argument('--limit', type=int, default=35000, help='数据量')
    parser.add_argument('--since_days', type=int, default=365, help='时间范围（天）')
    
    args = parser.parse_args()
    
    model_path = train_multiclass_model(
        limit=args.limit,
        since_days=args.since_days
    )
    
    sys.exit(0 if model_path else 1)
