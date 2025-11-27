#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练平衡的二分类模型
- 降低阈值至0.8%
- 删除震荡样本
- 平衡上涨和下跌样本
"""

import sys
import os
import argparse
from datetime import datetime, timezone, timedelta
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import SYMBOL, TIMEFRAME, BASE_PATH
from data.fetcher import fetch_ohlcv
from data.features import add_features

# 平衡的目标定义
PREDICTION_WINDOW = 4  # 预测未来4根K线
UP_THRESHOLD = 0.008  # 上涨阈值：0.8%
DOWN_THRESHOLD = -0.008  # 下跌阈值：-0.8%

def prepare_balanced_data(df, prediction_window=PREDICTION_WINDOW):
    """
    准备平衡的训练数据
    
    目标定义：
    - 1 (上涨): 未来收益率 > 0.8%
    - 0 (下跌): 未来收益率 < -0.8%
    - 删除: -0.8% <= 未来收益率 <= 0.8% (震荡)
    """
    print(f"📊 准备平衡数据...")
    print(f"  预测窗口: {prediction_window}根K线")
    print(f"  上涨阈值: >{UP_THRESHOLD*100:.1f}%")
    print(f"  下跌阈值: <{DOWN_THRESHOLD*100:.1f}%")
    
    # 计算未来收益率
    df['future_return'] = (df['close'].shift(-prediction_window) - df['close']) / df['close']
    
    # 删除无法计算的行
    df = df.dropna(subset=['future_return'])
    
    total_before = len(df)
    
    # 定义目标并删除震荡样本
    def classify(ret):
        if ret > UP_THRESHOLD:
            return 1  # 上涨
        elif ret < DOWN_THRESHOLD:
            return 0  # 下跌
        else:
            return None  # 震荡，将被删除
    
    df['target'] = df['future_return'].apply(classify)
    
    # 删除震荡样本
    df = df.dropna(subset=['target'])
    df['target'] = df['target'].astype(int)
    
    total_after = len(df)
    removed = total_before - total_after
    
    print(f"\n  样本处理:")
    print(f"    原始样本: {total_before}")
    print(f"    删除震荡: {removed} ({removed/total_before*100:.1f}%)")
    print(f"    保留样本: {total_after} ({total_after/total_before*100:.1f}%)")
    
    # 统计信息
    class_counts = df['target'].value_counts().sort_index()
    
    print(f"\n  样本分布:")
    print(f"    下跌 (0): {class_counts.get(0, 0)} ({class_counts.get(0, 0)/len(df)*100:.1f}%)")
    print(f"    上涨 (1): {class_counts.get(1, 0)} ({class_counts.get(1, 0)/len(df)*100:.1f}%)")
    
    # 计算平衡度
    if len(class_counts) == 2:
        balance_ratio = min(class_counts) / max(class_counts)
        print(f"    平衡度: {balance_ratio:.2f} (1.0为完全平衡)")
    
    # 收益率统计
    print(f"\n  收益率统计:")
    for cls, label in [(0, '下跌'), (1, '上涨')]:
        if cls in class_counts.index:
            cls_returns = df[df['target'] == cls]['future_return']
            print(f"    {label}: 均值{cls_returns.mean()*100:.3f}%, "
                  f"标准差{cls_returns.std()*100:.3f}%, "
                  f"范围[{cls_returns.min()*100:.2f}%, {cls_returns.max()*100:.2f}%]")
    
    return df

def train_balanced_model(limit=35000, since_days=365):
    """训练平衡的二分类模型"""
    
    print("=" * 80)
    print("训练平衡的二分类模型")
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
    
    # 准备平衡数据
    df = prepare_balanced_data(df, PREDICTION_WINDOW)
    
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
    
    # 统计训练集和测试集的分布
    train_dist = pd.Series(y_train).value_counts().sort_index()
    test_dist = pd.Series(y_test).value_counts().sort_index()
    
    print(f"\n训练集分布:")
    print(f"  下跌: {train_dist.get(0, 0)} ({train_dist.get(0, 0)/len(y_train)*100:.1f}%)")
    print(f"  上涨: {train_dist.get(1, 0)} ({train_dist.get(1, 0)/len(y_train)*100:.1f}%)")
    
    print(f"\n测试集分布:")
    print(f"  下跌: {test_dist.get(0, 0)} ({test_dist.get(0, 0)/len(y_test)*100:.1f}%)")
    print(f"  上涨: {test_dist.get(1, 0)} ({test_dist.get(1, 0)/len(y_test)*100:.1f}%)")
    
    # 训练LightGBM二分类模型
    print(f"\n🚀 训练LightGBM平衡模型...")
    
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'seed': 42,
        'is_unbalance': True  # 处理可能的轻微不平衡
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
    
    # AUC
    train_auc = roc_auc_score(y_train, y_pred_train)
    test_auc = roc_auc_score(y_test, y_pred_test)
    
    print(f"\nAUC:")
    print(f"  训练集: {train_auc:.4f}")
    print(f"  测试集: {test_auc:.4f}")
    
    # 准确率（使用0.5阈值）
    train_acc = accuracy_score(y_train, (y_pred_train > 0.5).astype(int))
    test_acc = accuracy_score(y_test, (y_pred_test > 0.5).astype(int))
    
    print(f"\n准确率 (阈值0.5):")
    print(f"  训练集: {train_acc*100:.2f}%")
    print(f"  测试集: {test_acc*100:.2f}%")
    
    # 预测概率分布
    print(f"\n预测概率分布 (测试集):")
    print(f"  最小值: {y_pred_test.min():.4f}")
    print(f"  最大值: {y_pred_test.max():.4f}")
    print(f"  均值: {y_pred_test.mean():.4f}")
    print(f"  中位数: {np.median(y_pred_test):.4f}")
    print(f"  标准差: {y_pred_test.std():.4f}")
    
    # 分位数
    print(f"\n分位数:")
    for p in [5, 25, 50, 75, 95]:
        print(f"  {p:2d}%: {np.percentile(y_pred_test, p):.4f}")
    
    # 预测分布
    print(f"\n预测分布 (阈值0.5):")
    pred_down = (y_pred_test < 0.5).sum()
    pred_up = (y_pred_test >= 0.5).sum()
    print(f"  预测下跌 (<0.5): {pred_down} ({pred_down/len(y_pred_test)*100:.1f}%)")
    print(f"  预测上涨 (>=0.5): {pred_up} ({pred_up/len(y_pred_test)*100:.1f}%)")
    
    # 混淆矩阵
    print(f"\n混淆矩阵 (测试集, 阈值0.5):")
    y_pred_binary = (y_pred_test > 0.5).astype(int)
    cm = confusion_matrix(y_test, y_pred_binary)
    print("           预测下跌  预测上涨")
    print(f"实际下跌    {cm[0][0]:6d}    {cm[0][1]:6d}")
    print(f"实际上涨    {cm[1][0]:6d}    {cm[1][1]:6d}")
    
    # 分类报告
    print(f"\n分类报告 (测试集):")
    target_names = ['下跌', '上涨']
    print(classification_report(y_test, y_pred_binary, target_names=target_names, zero_division=0))
    
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
    model_path = model_dir / f'{version}_balanced.pkl'
    metadata_path = model_dir / f'{version}_balanced.json'
    
    joblib.dump(model, model_path)
    
    metadata = {
        'version': version,
        'type': 'balanced_binary',
        'symbol': SYMBOL,
        'timeframe': TIMEFRAME,
        'prediction_window': PREDICTION_WINDOW,
        'up_threshold': UP_THRESHOLD,
        'down_threshold': DOWN_THRESHOLD,
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'test_auc': float(test_auc),
        'test_accuracy': float(test_acc),
        'pred_prob_min': float(y_pred_test.min()),
        'pred_prob_max': float(y_pred_test.max()),
        'pred_prob_mean': float(y_pred_test.mean()),
        'pred_prob_std': float(y_pred_test.std()),
        'features': feature_cols,
        'created_at': datetime.now(timezone.utc).isoformat()
    }
    
    import json
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # 更新latest指针
    latest_path = model_dir / 'latest_balanced.txt'
    with open(latest_path, 'w') as f:
        f.write(version)
    
    print(f"\n💾 模型已保存:")
    print(f"  版本: {version}")
    print(f"  路径: {model_path}")
    print(f"  元数据: {metadata_path}")
    
    print(f"\n✅ 训练成功!")
    print(f"  测试集AUC: {test_auc:.4f}")
    print(f"  预测概率范围: [{y_pred_test.min():.4f}, {y_pred_test.max():.4f}]")
    
    # 验证双向预测能力
    if y_pred_test.max() > 0.5 and y_pred_test.min() < 0.5:
        print(f"\n✅ 模型具备双向预测能力！")
        print(f"  可以预测上涨 (概率>0.5)")
        print(f"  可以预测下跌 (概率<0.5)")
    else:
        print(f"\n⚠️  模型可能仍然偏向单侧")
        if y_pred_test.max() <= 0.5:
            print(f"  所有预测都<0.5，仍然只预测下跌")
        if y_pred_test.min() >= 0.5:
            print(f"  所有预测都>0.5，只预测上涨")
    
    return model_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='训练平衡的二分类模型')
    parser.add_argument('--limit', type=int, default=35000, help='数据量')
    parser.add_argument('--since_days', type=int, default=365, help='时间范围（天）')
    
    args = parser.parse_args()
    
    model_path = train_balanced_model(
        limit=args.limit,
        since_days=args.since_days
    )
    
    sys.exit(0 if model_path else 1)
