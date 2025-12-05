import logging
from typing import Dict, Any

import pandas as pd

from ..data.features import add_features, get_feature_columns
from ..model.ensemble_model import EnsembleModel
from ..model.model_registry import load_latest_model_path
from ..config.settings import (
    CONFIDENCE_THRESHOLD,
    SELL_THRESHOLD,
    PROB_EMA_SPAN,
    USE_QUANTILE_THRESH,
    PROB_Q_HIGH,
    PROB_Q_LOW,
    PROB_WINDOW,
    COOLDOWN_BARS,
    TREND_FILTER,
    REQUIRE_P_EMA_UP,
    P_EMA_MOMENTUM_BARS,
)


class AISignalCore:
    def __init__(
            self,
            symbol: str = "BTC/USDT",
            timeframe: str = "1h",
            confidence_threshold: float = CONFIDENCE_THRESHOLD,
            cooldown_bars: int = COOLDOWN_BARS,
            trend_filter: bool = TREND_FILTER,
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
        self._prob_ema = None
        self._prob_hist = []
        self._pema_hist = []

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

        # 模型预测（只使用训练时的特征列）
        try:
            X = df_feat[self._feature_cols]
            prob = float(self.model.predict(X))
        except Exception as e:
            return self._hold_signal(f"模型预测失败: {e}")

        current_bar = n_bars - 1

        # 冷却期检查
        if current_bar - self._last_signal_bar <= self.cooldown_bars:
            return self._hold_signal("冷却期中", prob)

        # 概率EMA
        alpha = 2.0 / (float(PROB_EMA_SPAN) + 1.0)
        self._prob_ema = prob if self._prob_ema is None else (alpha * prob + (1 - alpha) * self._prob_ema)
        p_eval = float(self._prob_ema)
        self._pema_hist.append(p_eval)

        # 维护概率历史并计算自适应阈值
        self._prob_hist.append(prob)
        window_len = int(PROB_WINDOW) if int(PROB_WINDOW) > 10 else 10
        hist_window = self._prob_hist[-window_len:] if len(self._prob_hist) >= window_len else self._prob_hist
        buy_th = float(CONFIDENCE_THRESHOLD)
        sell_th = float(SELL_THRESHOLD)
        if USE_QUANTILE_THRESH and len(hist_window) >= max(30, int(window_len * 0.5)):
            import numpy as np
            buy_th = float(np.quantile(hist_window, float(PROB_Q_HIGH)))
            sell_th = float(np.quantile(hist_window, float(PROB_Q_LOW)))

        # 趋势过滤（仅做多）
        if self.trend_filter:
            close = df["close"].iloc[-1]
            ma50 = df_feat.get("ma50", pd.Series([close])).iloc[-1]
            if pd.isna(ma50):
                ma50 = close
            if p_eval > 0.5 and close < ma50:
                return self._hold_signal("趋势过滤（价格 < MA50）", prob)
            if p_eval < 0.5 and close > ma50:
                return self._hold_signal("趋势过滤（不做空）", prob)

        # 动量过滤：要求 p_ema 连续上升或近期均值抬升
        momentum_ok = True
        m = int(P_EMA_MOMENTUM_BARS) if int(P_EMA_MOMENTUM_BARS) > 1 else 2
        if REQUIRE_P_EMA_UP and len(self._pema_hist) >= m:
            recent = self._pema_hist[-m:]
            # 简单判断：最后一个大于第一个，或相邻增量之和>0
            momentum_ok = (recent[-1] > recent[0]) and (sum([recent[i] - recent[i-1] for i in range(1, len(recent))]) > 0)

        logging.info(f'p_eval: {p_eval:.3f}, buy_th: {buy_th:.3f}, sell_th: {sell_th:.3f}, momentum_ok: {momentum_ok}')

        # 生成信号
        if p_eval >= buy_th and momentum_ok:
            self._last_signal_bar = current_bar
            return {
                "signal": "buy",
                "confidence": p_eval,
                "reason": f"AI 看涨 (p_ema={p_eval:.3f}, th={buy_th:.3f}, mom={momentum_ok})",
                "timestamp": pd.Timestamp.now()
            }
        elif p_eval <= sell_th:
            self._last_signal_bar = current_bar
            return {
                "signal": "sell",
                "confidence": p_eval,
                "reason": f"AI 看跌 (p_ema={p_eval:.3f}, th={sell_th:.3f})",
                "timestamp": pd.Timestamp.now()
            }
        else:
            return self._hold_signal(f"置信度不足 (p_ema={p_eval:.3f}, th=({sell_th:.3f},{buy_th:.3f}))", p_eval)

    def _hold_signal(self, reason: str, confidence: float = 0.0) -> Dict[str, Any]:
        return {
            "signal": "hold",
            "confidence": confidence,
            "reason": reason,
            "timestamp": pd.Timestamp.now()
        }

    def get_signal(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        兼容 TradingEngine 的入口
        """
        return self.generate_signal(df)

    def reset(self):
        """重置状态（用于回测每轮开始）"""
        self._last_signal_bar = -1
        self._prob_ema = None
        self._prob_hist = []
