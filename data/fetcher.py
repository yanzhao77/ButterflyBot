# data/fetcher.py
"""
从交易所获取 OHLCV（K线）数据
使用 ccxt 统一接口，支持多交易所（默认 Binance）
"""
import time
import os

import ccxt
import pandas as pd

from config.settings import EXCHANGE_NAME, SYMBOL, TIMEFRAME, proxy, BASE_PATH

# 创建数据缓存目录
CACHE_DIR = BASE_PATH / 'cached_data'
os.makedirs(CACHE_DIR, exist_ok=True)


def get_cache_filename(symbol: str, timeframe: str, exchange_name: str) -> str:
    """生成缓存文件名"""
    filename = f"{exchange_name}_{symbol.replace('/', '_')}_{timeframe}.csv"
    return os.path.join(CACHE_DIR, filename)


def load_cached_data(symbol: str, timeframe: str, exchange_name: str) -> pd.DataFrame:
    """从本地缓存加载数据"""
    cache_file = get_cache_filename(symbol, timeframe, exchange_name)
    if os.path.exists(cache_file):
        try:
            df = pd.read_csv(cache_file)
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
            df.set_index('timestamp', inplace=True)
            return df
        except Exception as e:
            print(f"加载缓存文件失败: {e}")
    return pd.DataFrame()


def save_data_to_cache(df: pd.DataFrame, symbol: str, timeframe: str, exchange_name: str):
    """保存数据到本地缓存"""
    cache_file = get_cache_filename(symbol, timeframe, exchange_name)
    try:
        df.to_csv(cache_file)
        print(f"数据已缓存至: {cache_file}")
    except Exception as e:
        print(f"保存缓存文件失败: {e}")


def fetch_ohlcv(
        symbol: str = SYMBOL,
        timeframe: str = TIMEFRAME,
        limit: int = 1000,  # 交易所单次上限通常为 1000
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
    # 首先尝试从缓存加载数据
    df_cached = load_cached_data(symbol, timeframe, exchange_name)
    if not df_cached.empty:
        print("使用本地缓存数据")
        # 如果需要限制数据量
        if limit:
            return df_cached.tail(limit)
        return df_cached

    # 初始化交易所（不启用 rate limit，避免回测阻塞）
    try:
        exchange_class = getattr(ccxt, EXCHANGE_NAME)
        exchange = exchange_class({
            'enableRateLimit': True,
            'timeout': 30000,
            'options': {'defaultType': 'spot'},
            'proxies': {
                'http': proxy,
                'https': proxy,
            }
        })
        exchange.load_markets()
        # 分页抓取，直到达到 limit 或无更多数据
        all_rows = []
        fetched = 0
        timeframe_ms = int(exchange.parse_timeframe(timeframe) * 1000)
        next_since = since
        page_limit = min(1000, limit) if limit else 1000
        while True:
            remaining = (limit - fetched) if limit else page_limit
            this_limit = min(page_limit, remaining)
            batch = safe_fetch_ohlcv(exchange, symbol, timeframe=timeframe, limit=this_limit, since=next_since)
            if not batch:
                break
            all_rows.extend(batch)
            fetched += len(batch)
            # 终止条件：不足一页/达到上限
            if len(batch) < this_limit or (limit and fetched >= limit):
                break
            # 前进 since，+1 个 timeframe 以避免重叠
            last_ts = batch[-1][0]
            next_since = int(last_ts + timeframe_ms)
            # 避免过快请求
            time.sleep(10)
        ohlcv = all_rows
    except Exception as e:
        raise RuntimeError(f"❌ 从 {exchange_name} 获取 {symbol} {timeframe} 数据失败: {e}")
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

    # 保存到缓存
    save_data_to_cache(df, symbol, timeframe, exchange_name)

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
