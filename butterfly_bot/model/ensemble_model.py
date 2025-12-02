import lightgbm as lgb
import joblib

class EnsembleModel:
    def __init__(self, model_version, timeframe):
        self.model = joblib.load(model_version)

    def predict(self, features):
        return self.model.predict_proba(features)[:, 1]
