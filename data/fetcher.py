# data/fetcher.py
"""
从交易所获取 OHLCV（K线）数据
使用 ccxt 统一接口，支持多交易所（默认 Binance）
"""
import time

import ccxt
import pandas as pd

from config.settings import EXCHANGE_NAME, SYMBOL, TIMEFRAME, proxy


def fetch_ohlcv(
        symbol: str = SYMBOL,
        timeframe: str = TIMEFRAME,
        limit: int = 1000,  # 默认获取200根K线
        exchange_name: str = EXCHANGE_NAME,
        since: int = None
) -> pd.DataFrame:
    """
    获取指定交易对的 K 线数据

    参数:
        symbol (str): 交易对，如 "BTC/USDT"
        timeframe (str): 时间周期，如 "1h", "15m", "1d"
        limit (int): 获取的 K 线数量(最大 1000)
        exchange_name (str): 交易所名称(如 "binance")

    返回:
        pd.DataFrame: 包含 timestamp, open, high, low, close, volume 的 DataFrame
                      索引为 datetime(UTC)
    """
    # 初始化交易所（不启用 rate limit，避免回测阻塞）
    try:
        exchange_class = getattr(ccxt, EXCHANGE_NAME)
        exchange = exchange_class({
            'enableRateLimit': False,
            'timeout': 30000,
            'options': {'defaultType': 'spot'},
            'proxies': {
                'http': proxy,
                'https': proxy,
            }
        })
        # exchange.load_markets()
        ohlcv = safe_fetch_ohlcv(exchange, symbol, timeframe=timeframe, limit=limit, since=since)
    except Exception as e:
        raise RuntimeError(f"❌ 从 {exchange_name} 获取 {symbol} {timeframe} 数据失败: ")
    print(f"获取到 {len(ohlcv)} 条数据")
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


def safe_fetch_ohlcv(exchange, symbol, timeframe='1h', limit=300, since=None, retries=3):
    """带重试和错误处理的 K 线获取"""
    for i in range(retries):
        try:
            # 每次重试前重新加载 markets（关键！）
            exchange.load_markets()

            # 检查 symbol 是否存在
            if symbol not in exchange.markets:
                raise ValueError(
                    f"Symbol {symbol} not found in markets. Available: {list(exchange.markets.keys())[:5]}...")
            # 确保严格限制返回数据量
            # ccxt fetch_ohlcv: symbol, timeframe, since=None, limit=None, params={}
            # 只将 since/limit 作为顶层参数传递，params 只传 exchange-specific 参数
            params = {}  # 可扩展为 {'type': 'spot'} 等
            return exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit, params=params)
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            print(f"⚠️ 网络错误 (尝试 {i + 1}/{retries}): {str(e)}")
            time.sleep(2)  # 等待后重试
        except Exception as e:
            print(f"💥 未知错误: {str(e)}")
            raise

    raise RuntimeError("❌ 所有重试失败，请检查网络/API")
