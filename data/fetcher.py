# data/fetcher.py
"""
从交易所获取 OHLCV（K线）数据
使用 ccxt 统一接口，支持多交易所（默认 Binance）
"""

import ccxt
import pandas as pd
from datetime import datetime, timezone
from config.settings import EXCHANGE_NAME, SYMBOL, TIMEFRAME,proxy


def fetch_ohlcv(
    symbol: str = SYMBOL,
    timeframe: str = TIMEFRAME,
    limit: int = 1000,
    exchange_name: str = EXCHANGE_NAME
) -> pd.DataFrame:
    """
    获取指定交易对的 K 线数据

    参数:
        symbol (str): 交易对，如 "BTC/USDT"
        timeframe (str): 时间周期，如 "1h", "15m", "1d"
        limit (int): 获取的 K 线数量（最大 1000）
        exchange_name (str): 交易所名称（如 "binance"）

    返回:
        pd.DataFrame: 包含 timestamp, open, high, low, close, volume 的 DataFrame
                      索引为 datetime（UTC）
    """
    # 初始化交易所（不启用 rate limit，避免回测阻塞）
    exchange = getattr(ccxt, exchange_name)({
        'enableRateLimit': False,
        'timeout': 10000,
        'proxies': {
            'http': proxy,
            'https': proxy,
        }
    })

    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    except Exception as e:
        raise RuntimeError(f"❌ 从 {exchange_name} 获取 {symbol} {timeframe} 数据失败: {e}")

    if not ohlcv:
        raise ValueError(f"未获取到 {symbol} 的 K 线数据，请检查交易对和网络")

    # 转为 DataFrame
    df = pd.DataFrame(
        ohlcv,
        columns=["timestamp", "open", "high", "low", "close", "volume"]
    )

    # 转换时间戳为 UTC datetime 并设为索引
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)

    # 类型转换
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 按时间升序排序（确保最新在最后）
    df.sort_index(inplace=True)

    return df