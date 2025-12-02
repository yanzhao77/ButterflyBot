import lightgbm as lgb
import joblib

class EnsembleModel:
    def __init__(self, model_version, timeframe):
        self.model = joblib.load(model_version)

    def predict(self, features):
        # LGBModel.predict already returns probabilities
        if hasattr(self.model, 'predict'):
            return self.model.predict(features)
        else:
            # Fallback for sklearn-like models
            return self.model.predict_proba(features)[:, 1]
