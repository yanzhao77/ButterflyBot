#!/usr/bin/env python
"""分析模型预测概率分布"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path

# 添加项目路径
import sys
sys.path.insert(0, str(Path(__file__).parent))

from butterfly_bot.data.fetcher import fetch_ohlcv
from butterfly_bot.data.features import add_features, get_feature_columns

# 加载最新模型
model_path = Path(__file__).parent / "models/registry/v20251222_031926.pkl"
print(f'📁 模型路径: {model_path}')

with open(model_path, 'rb') as f:
    model = pickle.load(f)
print('✅ 模型已加载')

# 获取数据
print('\n📥 获取历史数据...')
df = fetch_ohlcv('DOGE/USDT', '15m', limit=500)
df = add_features(df)
df = df.dropna()
print(f'✅ 数据准备完成: {len(df)}行')

# 获取特征
feature_cols = get_feature_columns()
X = df[feature_cols].values

# 预测
print('\n🔮 执行预测...')
probs = model.predict(X)
print(f'✅ 预测完成: {len(probs)}个样本')

# 分析概率分布
print(f'\n📊 预测概率分布分析:')
print(f'=' * 60)
print(f'样本数: {len(probs)}')
print(f'最小值: {probs.min():.4f}')
print(f'最大值: {probs.max():.4f}')
print(f'平均值: {probs.mean():.4f}')
print(f'中位数: {np.median(probs):.4f}')
print(f'标准差: {probs.std():.4f}')

print(f'\n分位数:')
for q in [0.25, 0.50, 0.75, 0.90, 0.95, 0.99]:
    print(f'  {q*100:>3.0f}%: {np.quantile(probs, q):.4f}')

print(f'\n阈值分析:')
print(f'{"阈值":<8} {"数量":>6} {"百分比":>8}')
print(f'-' * 25)
for threshold in [0.10, 0.20, 0.30, 0.40, 0.50, 0.55, 0.60, 0.70]:
    count = (probs >= threshold).sum()
    pct = count / len(probs) * 100
    print(f'>= {threshold:.2f}  {count:>6}  {pct:>7.1f}%')

print(f'\n当前策略阈值: 0.55')
print(f'满足阈值的样本: {(probs >= 0.55).sum()}个 ({(probs >= 0.55).sum() / len(probs) * 100:.1f}%)')

print(f'\n建议:')
if (probs >= 0.55).sum() < 10:
    print(f'⚠️  满足当前阈值(0.55)的样本太少!')
    print(f'   建议降低CONFIDENCE_THRESHOLD到0.30-0.40')
    suggested_threshold = np.quantile(probs, 0.75)
    print(f'   或使用75分位数: {suggested_threshold:.4f}')
else:
    print(f'✅ 当前阈值(0.55)合理')

print(f'=' * 60)
