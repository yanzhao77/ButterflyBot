import lightgbm as lgb
import joblib

class EnsembleModel:
    def __init__(self, model_version, timeframe):
        self.model = joblib.load(model_version)

    def predict(self, features):
        """预测概率
        
        Args:
            features: DataFrame或numpy array，可以是单行或多行
            
        Returns:
            float or array: 单行输入返回float，多行输入返回array
        """
        # LGBModel.predict already returns probabilities
        if hasattr(self.model, 'predict'):
            probs = self.model.predict(features)
            # 如果只有一行，返回标量值
            if len(probs) == 1:
                return float(probs[0])
            # 否则返回最后一个值（用于实时预测）
            return float(probs[-1])
        else:
            # Fallback for sklearn-like models
            probs = self.model.predict_proba(features)[:, 1]
            if len(probs) == 1:
                return float(probs[0])
            return float(probs[-1])
