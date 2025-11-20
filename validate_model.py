#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证训练好的模型性能
"""

import sys
import os
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import BASE_PATH, SYMBOL, TIMEFRAME
from data.features import add_features, get_feature_columns
from model.model_registry import load_model_by_version, load_latest_model_path
import joblib

print("=" * 80)
print("模型性能验证")
print("=" * 80)

def load_test_data():
    """加载测试数据"""
    cache_dir = BASE_PATH / 'cached_data'
    filename = f"binance_{SYMBOL.replace('/', '_')}_{TIMEFRAME}.csv"
    cache_path = cache_dir / filename
    
    if not cache_path.exists():
        print(f"❌ 数据文件不存在: {cache_path}")
        return None
    
    print(f"\n📂 加载数据: {cache_path}")
    
    df = pd.read_csv(cache_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df.set_index('timestamp', inplace=True)
    
    print(f"✅ 加载成功: {len(df)} 条数据")
    
    # 添加特征
    print("🧩 构建特征...")
    df_feat = add_features(df)
    
    print(f"✅ 特征构建完成: {len(df_feat)} 条有效数据")
    
    return df_feat

def validate_model(model, df_feat):
    """验证模型性能"""
    
    feature_cols = get_feature_columns()
    
    # 使用后30%的数据作为验证集
    split_idx = int(len(df_feat) * 0.7)
    df_test = df_feat.iloc[split_idx:]
    
    print(f"\n📊 验证集大小: {len(df_test)} 条")
    
    X_test = df_test[feature_cols]
    y_test = df_test['target']
    
    # 预测
    print("\n🔮 开始预测...")
    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    # 计算各项指标
    print("\n" + "=" * 80)
    print("性能指标")
    print("=" * 80)
    
    # AUC
    try:
        auc = roc_auc_score(y_test, y_pred_proba)
        print(f"\n📈 AUC: {auc:.4f}")
        if auc >= 0.7:
            print("   ✅ 优秀 (≥0.7)")
        elif auc >= 0.6:
            print("   ⚠️  良好 (0.6-0.7)")
        else:
            print("   ❌ 较差 (<0.6)")
    except Exception as e:
        print(f"⚠️  AUC计算失败: {e}")
        auc = None
    
    # 准确率
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n🎯 准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # 精确率、召回率、F1
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    print(f"🔍 精确率: {precision:.4f} ({precision*100:.2f}%)")
    print(f"🔍 召回率: {recall:.4f} ({recall*100:.2f}%)")
    print(f"🔍 F1分数: {f1:.4f}")
    
    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print(f"\n📊 混淆矩阵:")
    print(f"   真阴性(TN): {tn:5d}  |  假阳性(FP): {fp:5d}")
    print(f"   假阴性(FN): {fn:5d}  |  真阳性(TP): {tp:5d}")
    
    # 预测分布
    print(f"\n📊 预测分布:")
    print(f"   预测上涨: {y_pred.sum()} ({y_pred.sum()/len(y_pred)*100:.1f}%)")
    print(f"   预测下跌: {len(y_pred)-y_pred.sum()} ({(len(y_pred)-y_pred.sum())/len(y_pred)*100:.1f}%)")
    
    print(f"\n📊 实际分布:")
    print(f"   实际上涨: {y_test.sum()} ({y_test.sum()/len(y_test)*100:.1f}%)")
    print(f"   实际下跌: {len(y_test)-y_test.sum()} ({(len(y_test)-y_test.sum())/len(y_test)*100:.1f}%)")
    
    # 概率分布
    print(f"\n📊 预测概率分布:")
    print(f"   最小值: {y_pred_proba.min():.4f}")
    print(f"   25分位: {np.percentile(y_pred_proba, 25):.4f}")
    print(f"   中位数: {np.median(y_pred_proba):.4f}")
    print(f"   75分位: {np.percentile(y_pred_proba, 75):.4f}")
    print(f"   最大值: {y_pred_proba.max():.4f}")
    print(f"   平均值: {y_pred_proba.mean():.4f}")
    print(f"   标准差: {y_pred_proba.std():.4f}")
    
    # 高置信度预测
    high_conf_threshold = 0.7
    low_conf_threshold = 0.3
    
    high_conf_up = (y_pred_proba >= high_conf_threshold).sum()
    high_conf_down = (y_pred_proba <= low_conf_threshold).sum()
    
    print(f"\n📊 高置信度预测:")
    print(f"   高置信上涨 (≥{high_conf_threshold}): {high_conf_up} ({high_conf_up/len(y_pred_proba)*100:.1f}%)")
    print(f"   高置信下跌 (≤{low_conf_threshold}): {high_conf_down} ({high_conf_down/len(y_pred_proba)*100:.1f}%)")
    
    if high_conf_up > 0:
        high_conf_up_mask = y_pred_proba >= high_conf_threshold
        high_conf_up_accuracy = accuracy_score(y_test[high_conf_up_mask], y_pred[high_conf_up_mask])
        print(f"   高置信上涨准确率: {high_conf_up_accuracy:.4f} ({high_conf_up_accuracy*100:.2f}%)")
    
    if high_conf_down > 0:
        high_conf_down_mask = y_pred_proba <= low_conf_threshold
        high_conf_down_accuracy = accuracy_score(y_test[high_conf_down_mask], y_pred[high_conf_down_mask])
        print(f"   高置信下跌准确率: {high_conf_down_accuracy:.4f} ({high_conf_down_accuracy*100:.2f}%)")
    
    # 评估
    print("\n" + "=" * 80)
    print("模型评估")
    print("=" * 80)
    
    if auc and auc >= 0.7:
        print("\n✅ 模型性能优秀！")
        print("   • AUC ≥ 0.7，具有良好的区分能力")
        print("   • 可以用于实际交易策略")
    elif auc and auc >= 0.6:
        print("\n⚠️  模型性能良好，但有改进空间")
        print("   • AUC 在 0.6-0.7 之间")
        print("   • 建议谨慎使用，小资金测试")
    else:
        print("\n❌ 模型性能不足")
        print("   • AUC < 0.6")
        print("   • 建议重新训练或调整特征")
    
    print("\n" + "=" * 80)
    
    return {
        'auc': auc,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'high_conf_up': high_conf_up,
        'high_conf_down': high_conf_down,
    }

if __name__ == "__main__":
    try:
        # 加载模型
        print("\n🤖 加载最新模型...")
        model_path = load_latest_model_path()
        if not model_path:
            print("❌ 未找到训练好的模型")
            sys.exit(1)
        model = joblib.load(model_path)
        print(f"✅ 模型加载成功: {model_path}")
        
        # 加载数据
        df_feat = load_test_data()
        if df_feat is None:
            sys.exit(1)
        
        # 验证模型
        metrics = validate_model(model, df_feat)
        
        print("\n✅ 验证完成！")
        sys.exit(0)
        
    except Exception as e:
        print(f"\n💥 验证失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
