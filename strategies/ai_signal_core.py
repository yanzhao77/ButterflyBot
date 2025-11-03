# strategies/ai_signal_core.py
"""
AI 信号核心逻辑：与框架无关的纯策略
"""

import pandas as pd
from typing import Dict, Any

from data.features import add_features, get_feature_columns
from model.ensemble_model import EnsembleModel
from model.model_registry import load_latest_model_path


class AISignalCore:
    def __init__(
        self,
        symbol: str = "BTC/USDT",
        timeframe: str = "1h",
        confidence_threshold: float = 0.6,
        cooldown_bars: int = 3,
        trend_filter: bool = True,
    ):
        self.symbol = symbol
        self.timeframe = timeframe
        self.confidence_threshold = confidence_threshold
        self.cooldown_bars = cooldown_bars
        self.trend_filter = trend_filter

        # 加载最新模型
        model_path = load_latest_model_path()
        if model_path is None:
            raise RuntimeError("❌ 未找到已注册模型！请先运行 `python model/train.py`")
        self.model = EnsembleModel(model_version=model_path, timeframe=timeframe)

        self._feature_cols = get_feature_columns()
        self._last_signal_bar = -1  # 用于冷却期

    def generate_signal(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        输入完整K线DataFrame，输出标准化信号字典
        """
        n_bars = len(df)
        if n_bars < 50:
            return self._hold_signal("数据不足")

        # 构建特征
        try:
            df_feat = add_features(df)
        except Exception as e:
            return self._hold_signal(f"特征构建失败: {e}")

        # 检查特征完整性
        if df_feat[self._feature_cols].isnull().any().any():
            return self._hold_signal("特征含缺失值")

        # 模型预测
        try:
            prob = float(self.model.predict(df_feat))
        except Exception as e:
            return self._hold_signal(f"模型预测失败: {e}")

        current_bar = n_bars - 1

        # 冷却期检查
        if current_bar - self._last_signal_bar <= self.cooldown_bars:
            return self._hold_signal("冷却期中", prob)

        # 趋势过滤（仅做多）
        if self.trend_filter:
            close = df["close"].iloc[-1]
            ma50 = df_feat.get("ma50", pd.Series([close])).iloc[-1]
            if pd.isna(ma50):
                ma50 = close
            if prob > 0.5 and close < ma50:
                return self._hold_signal("趋势过滤（价格 < MA50）", prob)
            if prob < 0.5 and close > ma50:
                return self._hold_signal("趋势过滤（不做空）", prob)

        # 生成信号
        if prob >= self.confidence_threshold:
            self._last_signal_bar = current_bar
            return {
                "signal": "buy",
                "confidence": prob,
                "reason": f"AI 看涨 (p={prob:.3f})",
                "timestamp": pd.Timestamp.now()
            }
        elif prob <= (1 - self.confidence_threshold):
            self._last_signal_bar = current_bar
            return {
                "signal": "sell",
                "confidence": prob,
                "reason": f"AI 看跌 (p={prob:.3f})",
                "timestamp": pd.Timestamp.now()
            }
        else:
            return self._hold_signal(f"置信度不足 ({prob:.3f})", prob)

    def _hold_signal(self, reason: str, confidence: float = 0.0) -> Dict[str, Any]:
        return {
            "signal": "hold",
            "confidence": confidence,
            "reason": reason,
            "timestamp": pd.Timestamp.now()
        }

    def reset(self):
        """重置状态（用于回测每轮开始）"""
        self._last_signal_bar = -1