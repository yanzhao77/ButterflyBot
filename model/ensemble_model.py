# model/ensemble_model.py
import os
import numpy as np
import pandas as pd
import joblib
import re
from utils.timeframe_utils import timeframe_to_pandas_freq
from backtest.metrics import load_metrics
from config.settings import REGISTRY_DIR

class EnsembleModel:
    def __init__(self, model_path: str = None, model_version: str = None, timeframe: str = "1h"):
        """支持传入 model_path（完整 .pkl 路径）或 model_version（如 vYYYYMMDD_HHMM）。

        如果传入 model_version，将会尝试从 REGISTRY_DIR 中拼接出对应的 .pkl 路径。
        timeframe 为字符串，如 "1h"，用于控制 Prophet 启用/频率计算。
        """
        # 解析并定位模型文件
        resolved_path = None
        if model_path:
            resolved_path = model_path
        elif model_version:
            v = os.path.basename(model_version)
            if v.endswith('.pkl'):
                resolved_path = model_version
            else:
                resolved_path = os.path.join(REGISTRY_DIR, f"{v}.pkl")
        else:
            raise ValueError("必须提供 model_path 或 model_version")

        # 加载模型（若缺少依赖如 lightgbm，降级为 dummy predictor 以便 smoke-test）
        try:
            self.lgb_model = joblib.load(resolved_path)
        except ModuleNotFoundError as e:
            # 依赖缺失，创建一个返回中性概率的 dummy 模型
            print(f"⚠️ 加载模型时缺少依赖: {e}. 将使用 DummyLGB 作为替代（预测返回 0.5）。")

            class DummyLGB:
                def predict(self, X):
                    import numpy as _np
                    # 返回与样本数相同的中性概率
                    n = len(X) if hasattr(X, '__len__') else 1
                    return _np.array([0.5] * n)

            self.lgb_model = DummyLGB()
        except Exception as e:
            raise RuntimeError(f"加载模型失败: {resolved_path} -> {e}")

        self.base_timeframe = timeframe.lower()
        self.freq = timeframe_to_pandas_freq(timeframe)
        self.use_prophet = self._should_use_prophet()

        metrics = load_metrics()
        self.auc = metrics.get("auc", 0.55)
        self.weights = self._compute_weights()
        print(f"⚙️  Ensemble 初始化 | AUC={self.auc:.4f} | 权重: {self.weights}")

    def _should_use_prophet(self) -> bool:
        tf = self.base_timeframe
        match = re.match(r'^(\d+)([mhdw])$', tf)
        if not match:
            return True
        n, unit = int(match.group(1)), match.group(2)
        return (unit != 'm') or (n >= 15)

    def _compute_weights(self):
        auc = self.auc
        if auc >= 0.7:
            lgb_w = 0.6
        elif auc >= 0.6:
            lgb_w = 0.5
        elif auc >= 0.55:
            lgb_w = 0.4
        else:
            lgb_w = 0.3

        if self.use_prophet:
            prophet_w = 0.3 if auc >= 0.55 else 0.4
            rule_w = 1.0 - lgb_w - prophet_w
        else:
            prophet_w = 0.0
            rule_w = 1.0 - lgb_w

        rule_w = max(rule_w, 0.1)
        lgb_w = 1.0 - prophet_w - rule_w
        return {'lgb': round(lgb_w, 2), 'prophet': round(prophet_w, 2), 'rule': round(rule_w, 2)}

    def predict_prophet_trend(self, df: pd.DataFrame) -> float:
        # 延迟导入 prophet，若不可用则返回中性分数 0.5
        try:
            from prophet import Prophet
        except Exception:
            return 0.5

        try:
            d = df[['close']].copy().reset_index()
            d.columns = ['ds', 'y']
            d['ds'] = pd.to_datetime(d['ds'])
            m = Prophet(daily_seasonality=False, weekly_seasonality=False, yearly_seasonality=False)
            m.fit(d)
            future = m.make_future_dataframe(periods=1, freq=self.freq)
            pred = m.predict(future)['yhat'].iloc[-1]
            return 1.0 if pred > df['close'].iloc[-1] else 0.0
        except Exception:
            return 0.5

    def predict_rule_based(self, df: pd.DataFrame) -> float:
        if 'rsi' not in df.columns or 'macd' not in df.columns:
            return 0.5
        rsi = df['rsi'].iloc[-1]
        macd = df['macd'].iloc[-1]
        macd_prev = df['macd'].iloc[-2] if len(df) > 1 else macd - 0.01
        if rsi < 30 and macd > macd_prev:
            return 0.85
        elif rsi > 70 and macd < macd_prev:
            return 0.15
        return 0.5

    def predict(self, df: pd.DataFrame) -> float:
        if len(df) < 30:
            cols = [c for c in df.columns if c != 'target']
            return float(self.lgb_model.predict(df[cols])[0])

        cols = [c for c in df.columns if c != 'target']
        lgb_prob = self.lgb_model.predict(df[cols])[0]
        prophet_score = self.predict_prophet_trend(df) if self.use_prophet else 0.5
        rule_score = self.predict_rule_based(df)

        final = (
            self.weights['lgb'] * lgb_prob +
            self.weights['prophet'] * prophet_score +
            self.weights['rule'] * rule_score
        )
        return float(np.clip(final, 0.0, 1.0))