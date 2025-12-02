# data/features.py
"""
特征工程：基于 OHLCV 数据生成技术指标与目标标签
"""

import pandas as pd
import numpy as np
from typing import List
from ..config.settings import TARGET_SHIFT, TARGET_THRESHOLD


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    在原始 OHLCV DataFrame 上添加技术指标和目标变量

    输入列: open, high, low, close, volume
    输出新增列示例: return, rsi, macd, ma20, ma50, volatility, target

    参数:
        df (pd.DataFrame): 原始 K 线数据（索引为 datetime）

    返回:
        pd.DataFrame: 添加特征后的完整 DataFrame
    """
    df = df.copy()

    # === 基础价格与收益率 ===
    df["return"] = df["close"].pct_change()
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))

    # === 移动平均线 ===
    df["ma20"] = df["close"].rolling(window=20).mean()
    df["ma50"] = df["close"].rolling(window=50).mean()
    df["ma_diff"] = df["ma20"] - df["ma50"]

    # === RSI (相对强弱指数) ===
    df["rsi"] = compute_rsi(df["close"], window=14)

    # === MACD ===
    exp12 = df["close"].ewm(span=12).mean()
    exp26 = df["close"].ewm(span=26).mean()
    df["macd"] = exp12 - exp26
    df["macd_signal"] = df["macd"].ewm(span=9).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # === 波动率（过去20根K线标准差）===
    df["volatility"] = df["log_return"].rolling(window=20).std()

    # === 成交量变化 ===
    df["volume_ma20"] = df["volume"].rolling(window=20).mean()
    df["volume_ratio"] = df["volume"] / df["volume_ma20"]

    # === 目标变量：未来 TARGET_SHIFT 根累计涨幅是否超过阈值 ===
    future_return = (df["close"].shift(-int(TARGET_SHIFT)) / df["close"]) - 1.0
    df["target"] = (future_return >= float(TARGET_THRESHOLD)).astype(int)

    # 删除包含 NaN 的行（因滚动窗口导致）
    df.dropna(inplace=True)

    return df


def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """计算 RSI 指标"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / (loss + 1e-8)
    rsi = 100 - (100 / (1 + rs))
    return rsi


# 可选：导出特征列名（供训练/预测时使用）
def get_feature_columns() -> List[str]:
    return [
        "open", "high", "low", "close", "volume",
        "return", "log_return",
        "ma20", "ma50", "ma_diff",
        "rsi", "macd", "macd_signal", "macd_hist",
        "volatility", "volume_ratio"
    ]