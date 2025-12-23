# model/lgb_model.py
"""
LightGBM 模型封装：训练与预测接口（支持类别权重平衡）
"""

import lightgbm as lgb
import numpy as np
from sklearn.utils.class_weight import compute_sample_weight


class LGBModel:
    def __init__(self, params: dict = None):
        self.model = None
        self.params = params or {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.8,
            "verbose": -1,
            "random_state": 42
        }

    def train(self, X_train, y_train, X_val=None, y_val=None, use_class_weight=False):
        """训练模型
        
        Args:
            X_train: 训练特征
            y_train: 训练标签
            X_val: 验证特征
            y_val: 验证标签
            use_class_weight: 是否使用类别权重平衡
        """
        # 计算类别权重
        sample_weight = None
        if use_class_weight:
            sample_weight = compute_sample_weight('balanced', y_train)
            print(f"✅ 启用类别权重平衡")
            print(f"   正样本平均权重: {sample_weight[y_train==1].mean():.2f}")
            print(f"   负样本平均权重: {sample_weight[y_train==0].mean():.2f}")
        
        train_data = lgb.Dataset(X_train, label=y_train, weight=sample_weight)
        valid_sets = [train_data]
        valid_names = ["train"]

        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            valid_sets.append(val_data)
            valid_names.append("valid")

        self.model = lgb.train(
            self.params,
            train_data,
            valid_sets=valid_sets,
            valid_names=valid_names,
            num_boost_round=200,
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
        )

    def predict(self, X):
        """返回上涨概率（正类概率）"""
        if self.model is None:
            raise RuntimeError("模型尚未训练！")
        return self.model.predict(X, num_iteration=self.model.best_iteration)

    def get_feature_importance(self):
        """获取特征重要性"""
        if self.model is None:
            return {}
        return dict(zip(self.model.feature_name(), self.model.feature_importance("gain")))
