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

logger = logging.getLogger(__name__)


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
        
        logger.info(f"🔧 AISignalCore初始化:")
        logger.info(f"   symbol: {symbol}")
        logger.info(f"   timeframe: {timeframe}")
        logger.info(f"   confidence_threshold: {confidence_threshold}")
        logger.info(f"   cooldown_bars: {cooldown_bars}")
        logger.info(f"   trend_filter: {trend_filter}")

        # 加载最新模型
        model_path = load_latest_model_path()
        if model_path is None:
            raise RuntimeError("❌ 未找到已注册模型！请先运行 `python model/train.py`")
        logger.info(f"✅ 加载模型: {model_path}")
        self.model = EnsembleModel(model_version=model_path, timeframe=timeframe)

        self._feature_cols = get_feature_columns()
        logger.info(f"✅ 特征列数: {len(self._feature_cols)}")
        
        self._last_signal_bar = -1  # 用于冷却期
        self._prob_ema = None
        self._prob_hist = []
        self._pema_hist = []
        self._signal_count = 0

    def generate_signal(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        输入完整K线DataFrame，输出标准化信号字典
        """
        self._signal_count += 1
        n_bars = len(df)
        
        logger.debug(f"\n{'='*80}")
        logger.debug(f"🔍 信号生成 #{self._signal_count}")
        logger.debug(f"   K线数量: {n_bars}")
        logger.debug(f"   最后价格: {df['close'].iloc[-1]:.4f}")
        logger.debug(f"   最后时间: {df.index[-1]}")
        
        if n_bars < 50:
            logger.debug(f"❌ 数据不足 (需要>=50根K线)")
            return self._hold_signal("数据不足")

        # 构建特征
        try:
            df_feat = add_features(df)
            logger.debug(f"✅ 特征构建成功")
        except Exception as e:
            logger.error(f"❌ 特征构建失败: {e}")
            return self._hold_signal(f"特征构建失败: {e}")

        # 检查特征完整性
        missing_features = df_feat[self._feature_cols].isnull().sum()
        if missing_features.any():
            logger.error(f"❌ 特征含缺失值:")
            for feat, count in missing_features[missing_features > 0].items():
                logger.error(f"   {feat}: {count}个缺失")
            return self._hold_signal("特征含缺失值")
        logger.debug(f"✅ 特征完整性检查通过")

        # 模型预测（只使用训练时的特征列）
        try:
            X = df_feat[self._feature_cols]
            prob = float(self.model.predict(X))
            logger.debug(f"✅ 模型预测成功: prob={prob:.4f}")
        except Exception as e:
            logger.error(f"❌ 模型预测失败: {e}")
            return self._hold_signal(f"模型预测失败: {e}")

        current_bar = n_bars - 1

        # 冷却期检查
        bars_since_last = current_bar - self._last_signal_bar
        if bars_since_last <= self.cooldown_bars:
            logger.debug(f"⏸️  冷却期中 (距上次信号{bars_since_last}根K线，需要>{self.cooldown_bars})")
            return self._hold_signal("冷却期中", prob)
        logger.debug(f"✅ 冷却期检查通过 (距上次信号{bars_since_last}根K线)")

        # 概率EMA
        alpha = 2.0 / (float(PROB_EMA_SPAN) + 1.0)
        self._prob_ema = prob if self._prob_ema is None else (alpha * prob + (1 - alpha) * self._prob_ema)
        p_eval = float(self._prob_ema)
        self._pema_hist.append(p_eval)
        logger.debug(f"📊 概率平滑: prob={prob:.4f} -> p_ema={p_eval:.4f}")

        # 维护概率历史并计算自适应阈值
        self._prob_hist.append(prob)
        window_len = int(PROB_WINDOW) if int(PROB_WINDOW) > 10 else 10
        hist_window = self._prob_hist[-window_len:] if len(self._prob_hist) >= window_len else self._prob_hist
        buy_th = float(CONFIDENCE_THRESHOLD)
        sell_th = float(SELL_THRESHOLD)
        
        logger.debug(f"📊 阈值设置:")
        logger.debug(f"   USE_QUANTILE_THRESH: {USE_QUANTILE_THRESH}")
        logger.debug(f"   固定阈值: buy={buy_th:.4f}, sell={sell_th:.4f}")
        
        if USE_QUANTILE_THRESH and len(hist_window) >= max(30, int(window_len * 0.5)):
            import numpy as np
            buy_th = float(np.quantile(hist_window, float(PROB_Q_HIGH)))
            sell_th = float(np.quantile(hist_window, float(PROB_Q_LOW)))
            logger.debug(f"   分位数阈值: buy={buy_th:.4f} (Q{PROB_Q_HIGH}), sell={sell_th:.4f} (Q{PROB_Q_LOW})")

        # 趋势过滤（仅做多）
        if self.trend_filter:
            close = df["close"].iloc[-1]
            ma50 = df_feat.get("ma50", pd.Series([close])).iloc[-1]
            if pd.isna(ma50):
                ma50 = close
            logger.debug(f"📈 趋势过滤: close={close:.4f}, ma50={ma50:.4f}")
            
            if p_eval > 0.5 and close < ma50:
                logger.debug(f"❌ 趋势过滤阻止买入 (价格 < MA50)")
                return self._hold_signal("趋势过滤（价格 < MA50）", prob)
            if p_eval < 0.5 and close > ma50:
                logger.debug(f"❌ 趋势过滤阻止卖出 (不做空)")
                return self._hold_signal("趋势过滤（不做空）", prob)
            logger.debug(f"✅ 趋势过滤通过")

        # 动量过滤：要求 p_ema 连续上升或近期均值抬升
        momentum_ok = True
        m = int(P_EMA_MOMENTUM_BARS) if int(P_EMA_MOMENTUM_BARS) > 1 else 2
        
        logger.debug(f"📈 动量过滤:")
        logger.debug(f"   REQUIRE_P_EMA_UP: {REQUIRE_P_EMA_UP}")
        
        if REQUIRE_P_EMA_UP and len(self._pema_hist) >= m:
            recent = self._pema_hist[-m:]
            # 简单判断：最后一个大于第一个，或相邻增量之和>0
            momentum_ok = (recent[-1] > recent[0]) and (sum([recent[i] - recent[i-1] for i in range(1, len(recent))]) > 0)
            logger.debug(f"   最近{m}个p_ema: {[f'{x:.4f}' for x in recent]}")
            logger.debug(f"   动量检查: {'✅ 通过' if momentum_ok else '❌ 未通过'}")
        else:
            logger.debug(f"   动量过滤已禁用或数据不足")

        logger.info(f"📊 信号判断: p_ema={p_eval:.4f}, buy_th={buy_th:.4f}, sell_th={sell_th:.4f}, momentum_ok={momentum_ok}")

        # 生成信号
        if p_eval >= buy_th and momentum_ok:
            self._last_signal_bar = current_bar
            signal = {
                "signal": "buy",
                "confidence": p_eval,
                "reason": f"AI 看涨 (p_ema={p_eval:.3f}, th={buy_th:.3f}, mom={momentum_ok})",
                "timestamp": pd.Timestamp.now()
            }
            logger.info(f"🟢 生成买入信号: {signal}")
            return signal
        elif p_eval <= sell_th:
            self._last_signal_bar = current_bar
            signal = {
                "signal": "sell",
                "confidence": p_eval,
                "reason": f"AI 看跌 (p_ema={p_eval:.3f}, th={sell_th:.3f})",
                "timestamp": pd.Timestamp.now()
            }
            logger.info(f"🔴 生成卖出信号: {signal}")
            return signal
        else:
            logger.debug(f"⚪ 持有: 置信度不足 (p_ema={p_eval:.4f} 不在 [{sell_th:.4f}, {buy_th:.4f}] 范围外)")
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
        logger.info(f"🔄 重置AISignalCore状态")
        self._last_signal_bar = -1
        self._prob_ema = None
        self._prob_hist = []
        self._pema_hist = []
        self._signal_count = 0
