import logging
from typing import Dict, Any, Optional

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
            take_profit_pct: float = 6.0,  # 止盈百分比
            stop_loss_pct: float = 3.0,    # 止损百分比
            max_holding_bars: int = 50,    # 最大持仓K线数
    ):
        self.symbol = symbol
        self.timeframe = timeframe
        self.confidence_threshold = confidence_threshold
        self.cooldown_bars = cooldown_bars
        self.trend_filter = trend_filter
        self.take_profit_pct = take_profit_pct
        self.stop_loss_pct = stop_loss_pct
        self.max_holding_bars = max_holding_bars
        
        logger.info(f"🔧 AISignalCore初始化:")
        logger.info(f"   symbol: {symbol}")
        logger.info(f"   timeframe: {timeframe}")
        logger.info(f"   confidence_threshold: {confidence_threshold}")
        logger.info(f"   cooldown_bars: {cooldown_bars}")
        logger.info(f"   trend_filter: {trend_filter}")
        logger.info(f"   take_profit_pct: {take_profit_pct}%")
        logger.info(f"   stop_loss_pct: {stop_loss_pct}%")
        logger.info(f"   max_holding_bars: {max_holding_bars}")

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
        
        # 持仓状态跟踪
        self.has_position = False
        self.entry_price: Optional[float] = None
        self.holding_bars = 0
        self.position_bar = -1  # 开仓时的K线索引

    def update_position_status(self, has_position: bool, entry_price: Optional[float] = None):
        """
        更新持仓状态（由外部调用，如TradingEngine）
        
        Args:
            has_position: 是否有持仓
            entry_price: 开仓价格
        """
        old_status = self.has_position
        self.has_position = has_position
        
        if has_position and not old_status:
            # 新开仓
            self.entry_price = entry_price
            self.holding_bars = 0
            self.position_bar = self._signal_count
            logger.info(f"📍 开仓记录: 价格={entry_price:.5f}, K线#{self._signal_count}")
        elif not has_position and old_status:
            # 平仓
            logger.info(f"📍 平仓记录: 持仓{self.holding_bars}根K线")
            self.entry_price = None
            self.holding_bars = 0
            self.position_bar = -1
        elif has_position:
            # 持仓中
            self.holding_bars += 1

    def calculate_profit_pct(self, current_price: float) -> float:
        """
        计算当前盈亏百分比
        
        Args:
            current_price: 当前价格
            
        Returns:
            盈亏百分比
        """
        if self.entry_price is None or self.entry_price == 0:
            return 0.0
        return ((current_price - self.entry_price) / self.entry_price) * 100.0

    def generate_signal(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        输入完整K线DataFrame，输出标准化信号字典
        """
        self._signal_count += 1
        n_bars = len(df)
        current_price = df['close'].iloc[-1]
        
        logger.debug(f"\n{'='*80}")
        logger.debug(f"🔍 信号生成 #{self._signal_count}")
        logger.debug(f"   K线数量: {n_bars}")
        logger.debug(f"   当前价格: {current_price:.5f}")
        logger.debug(f"   持仓状态: {self.has_position}")
        if self.has_position:
            logger.debug(f"   开仓价格: {self.entry_price:.5f}")
            logger.debug(f"   持仓K线: {self.holding_bars}")
        
        if n_bars < 50:
            logger.debug(f"❌ 数据不足 (需要>=50根K线)")
            return self._hold_signal("数据不足")

        # ========== 核心逻辑：如果有持仓，优先检查卖出条件 ==========
        if self.has_position and self.entry_price is not None:
            current_profit_pct = self.calculate_profit_pct(current_price)
            logger.info(f"💰 持仓盈亏: {current_profit_pct:+.2f}% (持仓{self.holding_bars}根K线)")
            
            # 1. 检查止盈
            if current_profit_pct >= self.take_profit_pct:
                signal = {
                    "signal": "sell",
                    "confidence": 1.0,
                    "reason": f"止盈 ({current_profit_pct:+.2f}% >= {self.take_profit_pct}%)",
                    "timestamp": pd.Timestamp.now()
                }
                logger.info(f"🎯 触发止盈: {signal}")
                return signal
            
            # 2. 检查止损
            if current_profit_pct <= -self.stop_loss_pct:
                signal = {
                    "signal": "sell",
                    "confidence": 1.0,
                    "reason": f"止损 ({current_profit_pct:+.2f}% <= -{self.stop_loss_pct}%)",
                    "timestamp": pd.Timestamp.now()
                }
                logger.info(f"🛑 触发止损: {signal}")
                return signal
            
            # 3. 检查时间止损
            if self.holding_bars >= self.max_holding_bars:
                signal = {
                    "signal": "sell",
                    "confidence": 0.5,
                    "reason": f"时间止损 (持仓{self.holding_bars}根K线 >= {self.max_holding_bars})",
                    "timestamp": pd.Timestamp.now()
                }
                logger.info(f"⏰ 触发时间止损: {signal}")
                return signal
            
            # 4. 检查AI预测（看跌）
            # 构建特征并预测
            try:
                df_feat = add_features(df)
                X = df_feat[self._feature_cols]
                prob = float(self.model.predict(X))
                
                # 概率EMA
                alpha = 2.0 / (float(PROB_EMA_SPAN) + 1.0)
                self._prob_ema = prob if self._prob_ema is None else (alpha * prob + (1 - alpha) * self._prob_ema)
                p_eval = float(self._prob_ema)
                
                sell_th = float(SELL_THRESHOLD)
                logger.info(f"📊 AI预测: p_ema={p_eval:.4f}, sell_th={sell_th:.4f}")
                
                if p_eval <= sell_th:
                    signal = {
                        "signal": "sell",
                        "confidence": p_eval,
                        "reason": f"AI看跌 (p_ema={p_eval:.3f} <= {sell_th:.3f}, 盈亏{current_profit_pct:+.2f}%)",
                        "timestamp": pd.Timestamp.now()
                    }
                    logger.info(f"📉 AI预测看跌: {signal}")
                    return signal
                else:
                    logger.debug(f"✅ 继续持有 (AI预测p_ema={p_eval:.4f} > sell_th={sell_th:.4f})")
            except Exception as e:
                logger.error(f"❌ 特征/预测失败: {e}")
            
            # 默认持有
            return self._hold_signal(f"继续持有 (盈亏{current_profit_pct:+.2f}%, 持仓{self.holding_bars}根K线)", p_eval if 'p_eval' in locals() else 0.0)

        # ========== 如果没有持仓，检查买入条件 ==========
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
            logger.error(f"❌ 特征含缺失值")
            return self._hold_signal("特征含缺失值")

        # 模型预测
        try:
            X = df_feat[self._feature_cols]
            prob = float(self.model.predict(X))
            logger.debug(f"✅ 模型预测: prob={prob:.4f}")
        except Exception as e:
            logger.error(f"❌ 模型预测失败: {e}")
            return self._hold_signal(f"模型预测失败: {e}")

        current_bar = n_bars - 1

        # 冷却期检查
        bars_since_last = current_bar - self._last_signal_bar
        if bars_since_last <= self.cooldown_bars:
            logger.debug(f"⏸️  冷却期中 (距上次信号{bars_since_last}根K线)")
            return self._hold_signal("冷却期中", prob)

        # 概率EMA
        alpha = 2.0 / (float(PROB_EMA_SPAN) + 1.0)
        self._prob_ema = prob if self._prob_ema is None else (alpha * prob + (1 - alpha) * self._prob_ema)
        p_eval = float(self._prob_ema)
        self._pema_hist.append(p_eval)

        # 阈值
        buy_th = float(CONFIDENCE_THRESHOLD)
        sell_th = float(SELL_THRESHOLD)

        # 趋势过滤
        if self.trend_filter:
            close = df["close"].iloc[-1]
            ma50 = df_feat.get("ma50", pd.Series([close])).iloc[-1]
            if pd.isna(ma50):
                ma50 = close
            
            if p_eval > 0.5 and close < ma50:
                logger.debug(f"❌ 趋势过滤阻止买入 (价格 < MA50)")
                return self._hold_signal("趋势过滤（价格 < MA50）", prob)

        # 动量过滤
        momentum_ok = True
        m = int(P_EMA_MOMENTUM_BARS) if int(P_EMA_MOMENTUM_BARS) > 1 else 2
        
        if REQUIRE_P_EMA_UP and len(self._pema_hist) >= m:
            recent = self._pema_hist[-m:]
            momentum_ok = (recent[-1] > recent[0]) and (sum([recent[i] - recent[i-1] for i in range(1, len(recent))]) > 0)

        logger.info(f"📊 买入判断: p_ema={p_eval:.4f}, buy_th={buy_th:.4f}, momentum_ok={momentum_ok}")

        # 生成买入信号
        if p_eval >= buy_th and momentum_ok:
            self._last_signal_bar = current_bar
            signal = {
                "signal": "buy",
                "confidence": p_eval,
                "reason": f"AI看涨 (p_ema={p_eval:.3f} >= {buy_th:.3f})",
                "timestamp": pd.Timestamp.now()
            }
            logger.info(f"🟢 生成买入信号: {signal}")
            return signal
        else:
            logger.debug(f"⚪ 无明确买入信号")
            return self._hold_signal(f"置信度不足 (p_ema={p_eval:.3f} < {buy_th:.3f})", p_eval)

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
        self.has_position = False
        self.entry_price = None
        self.holding_bars = 0
        self.position_bar = -1
