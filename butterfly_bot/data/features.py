# data/features.py
"""
特征工程：基于 OHLCV 数据生成技术指标与目标标签
优化版本：增加更多技术指标，提升模型性能
"""

import pandas as pd
import numpy as np
import logging
from typing import List
from ..config.settings import TARGET_SHIFT, TARGET_THRESHOLD

logger = logging.getLogger(__name__)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    在原始 OHLCV DataFrame 上添加技术指标和目标变量
    
    优化版本：增加了25+个特征，包括：
    - 趋势指标（MA、EMA、MACD、ADX）
    - 动量指标（RSI、Stochastic、ROC、Williams %R）
    - 波动率指标（ATR、Bollinger Bands）
    - 成交量指标（OBV、Volume Profile）
    - 价格形态特征

    参数:
        df (pd.DataFrame): 原始 K 线数据（索引为 datetime）

    返回:
        pd.DataFrame: 添加特征后的完整 DataFrame
    """
    if df.empty:
        logger.warning("输入DataFrame为空")
        return df
    
    if len(df) < 100:
        logger.warning(f"数据量过少（{len(df)}条），建议至少100条以上")
    
    df = df.copy()

    # === 1. 基础价格与收益率 ===
    df["return"] = df["close"].pct_change()
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df["return_5"] = df["close"].pct_change(5)  # 5周期收益率
    df["return_10"] = df["close"].pct_change(10)  # 10周期收益率

    # === 2. 移动平均线（多周期）===
    df["ma5"] = df["close"].rolling(window=5).mean()
    df["ma10"] = df["close"].rolling(window=10).mean()
    df["ma20"] = df["close"].rolling(window=20).mean()
    df["ma50"] = df["close"].rolling(window=50).mean()
    df["ma_diff_short"] = df["ma5"] - df["ma20"]  # 短期均线差
    df["ma_diff_long"] = df["ma20"] - df["ma50"]  # 长期均线差
    
    # 价格与均线的相对位置
    df["price_to_ma20"] = (df["close"] - df["ma20"]) / df["ma20"]
    df["price_to_ma50"] = (df["close"] - df["ma50"]) / df["ma50"]

    # === 3. 指数移动平均线 ===
    df["ema12"] = df["close"].ewm(span=12).mean()
    df["ema26"] = df["close"].ewm(span=26).mean()

    # === 4. RSI (相对强弱指数) ===
    df["rsi"] = compute_rsi(df["close"], window=14)
    df["rsi_6"] = compute_rsi(df["close"], window=6)  # 短期RSI
    df["rsi_24"] = compute_rsi(df["close"], window=24)  # 长期RSI

    # === 5. MACD ===
    df["macd"] = df["ema12"] - df["ema26"]
    df["macd_signal"] = df["macd"].ewm(span=9).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # === 6. 布林带 (Bollinger Bands) ===
    bb_window = 20
    bb_std = 2
    df["bb_middle"] = df["close"].rolling(window=bb_window).mean()
    bb_rolling_std = df["close"].rolling(window=bb_window).std()
    df["bb_upper"] = df["bb_middle"] + (bb_rolling_std * bb_std)
    df["bb_lower"] = df["bb_middle"] - (bb_rolling_std * bb_std)
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]
    df["bb_position"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

    # === 7. ATR (平均真实波幅) ===
    df["atr"] = compute_atr(df, window=14)
    df["atr_ratio"] = df["atr"] / df["close"]  # ATR相对于价格的比例

    # === 8. 波动率 ===
    df["volatility_10"] = df["log_return"].rolling(window=10).std()
    df["volatility_20"] = df["log_return"].rolling(window=20).std()
    df["volatility_50"] = df["log_return"].rolling(window=50).std()

    # === 9. Stochastic Oscillator (随机指标) ===
    stoch_window = 14
    df["stoch_k"] = compute_stochastic(df, window=stoch_window)
    df["stoch_d"] = df["stoch_k"].rolling(window=3).mean()

    # === 10. ROC (变动率指标) ===
    df["roc_10"] = ((df["close"] - df["close"].shift(10)) / df["close"].shift(10)) * 100
    df["roc_20"] = ((df["close"] - df["close"].shift(20)) / df["close"].shift(20)) * 100

    # === 11. Williams %R ===
    df["williams_r"] = compute_williams_r(df, window=14)

    # === 12. 成交量指标 ===
    df["volume_ma5"] = df["volume"].rolling(window=5).mean()
    df["volume_ma20"] = df["volume"].rolling(window=20).mean()
    df["volume_ratio"] = df["volume"] / df["volume_ma20"]
    df["volume_change"] = df["volume"].pct_change()
    
    # OBV (能量潮指标)
    df["obv"] = compute_obv(df)
    df["obv_ma"] = df["obv"].rolling(window=20).mean()

    # === 13. 价格形态特征 ===
    # 最高价和最低价的相对位置
    df["high_low_ratio"] = (df["high"] - df["low"]) / df["close"]
    df["close_position"] = (df["close"] - df["low"]) / (df["high"] - df["low"] + 1e-8)
    
    # 价格动量
    df["momentum_5"] = df["close"] - df["close"].shift(5)
    df["momentum_10"] = df["close"] - df["close"].shift(10)

    # === 14. ADX (平均趋向指数) ===
    df["adx"] = compute_adx(df, window=14)

    # === 15. 目标变量：未来 TARGET_SHIFT 根累计涨幅是否超过阈值 ===
    future_return = (df["close"].shift(-int(TARGET_SHIFT)) / df["close"]) - 1.0
    df["target"] = (future_return >= float(TARGET_THRESHOLD)).astype(int)

    # 删除包含 NaN 的行（因滚动窗口导致）
    rows_before = len(df)
    df.dropna(inplace=True)
    rows_after = len(df)
    
    if rows_after == 0:
        logger.error("特征工程后所有数据都被删除，请检查输入数据")
    elif rows_before - rows_after > rows_before * 0.5:
        logger.warning(f"特征工程删除了{rows_before - rows_after}条数据（{(rows_before - rows_after) / rows_before * 100:.1f}%）")
    else:
        logger.debug(f"特征工程完成，有效数据{rows_after}条，特征数{len(get_feature_columns())}个")

    return df


def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """计算 RSI 指标"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / (loss + 1e-8)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """计算 ATR (平均真实波幅)"""
    high_low = df["high"] - df["low"]
    high_close = np.abs(df["high"] - df["close"].shift())
    low_close = np.abs(df["low"] - df["close"].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=window).mean()
    return atr


def compute_stochastic(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """计算 Stochastic Oscillator %K"""
    low_min = df["low"].rolling(window=window).min()
    high_max = df["high"].rolling(window=window).max()
    stoch_k = 100 * (df["close"] - low_min) / (high_max - low_min + 1e-8)
    return stoch_k


def compute_williams_r(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """计算 Williams %R"""
    high_max = df["high"].rolling(window=window).max()
    low_min = df["low"].rolling(window=window).min()
    williams_r = -100 * (high_max - df["close"]) / (high_max - low_min + 1e-8)
    return williams_r


def compute_obv(df: pd.DataFrame) -> pd.Series:
    """计算 OBV (能量潮指标)"""
    obv = (np.sign(df["close"].diff()) * df["volume"]).fillna(0).cumsum()
    return obv


def compute_adx(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """计算 ADX (平均趋向指数)"""
    # 计算 +DM 和 -DM
    high_diff = df["high"].diff()
    low_diff = -df["low"].diff()
    
    plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
    minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
    
    # 计算 ATR
    atr = compute_atr(df, window)
    
    # 计算 +DI 和 -DI
    plus_di = 100 * (plus_dm.rolling(window=window).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=window).mean() / atr)
    
    # 计算 DX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8)
    
    # 计算 ADX
    adx = dx.rolling(window=window).mean()
    
    return adx


def get_feature_columns() -> List[str]:
    """
    返回用于模型训练和预测的特征列表
    注意：不包含target列，因为预测时不需要
    
    优化版本：包含40+个特征
    """
    return [
        # 基础价格
        "open", "high", "low", "close", "volume",
        
        # 收益率
        "return", "log_return", "return_5", "return_10",
        
        # 移动平均线
        "ma5", "ma10", "ma20", "ma50",
        "ma_diff_short", "ma_diff_long",
        "price_to_ma20", "price_to_ma50",
        
        # 指数移动平均
        "ema12", "ema26",
        
        # RSI
        "rsi", "rsi_6", "rsi_24",
        
        # MACD
        "macd", "macd_signal", "macd_hist",
        
        # 布林带
        "bb_middle", "bb_upper", "bb_lower", "bb_width", "bb_position",
        
        # ATR
        "atr", "atr_ratio",
        
        # 波动率
        "volatility_10", "volatility_20", "volatility_50",
        
        # Stochastic
        "stoch_k", "stoch_d",
        
        # ROC
        "roc_10", "roc_20",
        
        # Williams %R
        "williams_r",
        
        # 成交量
        "volume_ma5", "volume_ma20", "volume_ratio", "volume_change",
        "obv", "obv_ma",
        
        # 价格形态
        "high_low_ratio", "close_position",
        "momentum_5", "momentum_10",
        
        # ADX
        "adx"
    ]
