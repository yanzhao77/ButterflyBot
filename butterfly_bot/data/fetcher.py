# data/fetcher.py
"""
从交易所获取 OHLCV（K线）数据
使用 ccxt 统一接口，支持多交易所（默认 Binance）
"""
import time
import os
import logging

import ccxt
import pandas as pd

from ..config.settings import EXCHANGE_NAME, SYMBOL, TIMEFRAME, proxy, BASE_PATH

logger = logging.getLogger(__name__)

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
            logger.warning(f"加载缓存文件失败: {e}")
    return pd.DataFrame()


def save_data_to_cache(df: pd.DataFrame, symbol: str, timeframe: str, exchange_name: str):
    """保存数据到本地缓存"""
    cache_file = get_cache_filename(symbol, timeframe, exchange_name)
    try:
        df.to_csv(cache_file)
        logger.info(f"数据已缓存至: {cache_file}")
    except Exception as e:
        logger.error(f"保存缓存文件失败: {e}")


def fetch_ohlcv(
        symbol: str = SYMBOL,
        timeframe: str = TIMEFRAME,
        limit: int = 1000,  # 交易所单次上限通常为 1000
        exchange_name: str = EXCHANGE_NAME,
        since: int = None,
        use_cache: bool = True  # 是否使用缓存
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
    # 首先尝试从缓存加载数据（仅当use_cache=True时）
    if use_cache:
        df_cached = load_cached_data(symbol, timeframe, exchange_name)
        if not df_cached.empty:
            logger.info("使用本地缓存数据")
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
    logger.info(f"获取到 {len(ohlcv)} 条数据")
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
            logger.warning(f"网络错误 (尝试 {i + 1}/{retries}): {str(e)}")
            time.sleep(2)  # 等待后重试
        except Exception as e:
            logger.error(f"未知错误: {str(e)}")
            raise

    raise RuntimeError("❌ 所有重试失败，请检查网络/API")

def fetch_historical_data(symbol, start_date, end_date, timeframe=TIMEFRAME, exchange_name=EXCHANGE_NAME):
    """
    获取指定时间范围的历史数据，支持分批获取
    
    参数:
        symbol (str): 交易对，如 "DOGE/USDT"
        start_date (str): 开始日期，如 "2023-09-01"
        end_date (str): 结束日期，如 "2023-11-30"
        timeframe (str): 时间周期，如 "15m", "1h", "1d"
        exchange_name (str): 交易所名称
    
    返回:
        pd.DataFrame: 包含完整时间范围的K线数据
    """
    logger.info(f"📅 获取历史数据: {symbol} {timeframe} from {start_date} to {end_date}")
    
    # 转换为时间戳（毫秒）
    start_ts = int(pd.Timestamp(start_date).timestamp() * 1000)
    end_ts = int(pd.Timestamp(end_date).timestamp() * 1000)
    
    # 计算时间周期对应的毫秒数
    timeframe_ms_map = {
        '1m': 60 * 1000,
        '5m': 5 * 60 * 1000,
        '15m': 15 * 60 * 1000,
        '30m': 30 * 60 * 1000,
        '1h': 60 * 60 * 1000,
        '4h': 4 * 60 * 60 * 1000,
        '1d': 24 * 60 * 60 * 1000,
    }
    
    tf_ms = timeframe_ms_map.get(timeframe, 15 * 60 * 1000)
    
    # 计算需要的K线数量
    required_bars = int((end_ts - start_ts) / tf_ms) + 1
    logger.info(f"📊 预计需要 {required_bars} 根K线")
    
    # 先尝试从缓存加载
    df_cached = load_cached_data(symbol, timeframe, exchange_name)
    if not df_cached.empty:
        # 过滤时间范围（转换为Timestamp）
        start_ts_filter = pd.Timestamp(start_date, tz='UTC')
        end_ts_filter = pd.Timestamp(end_date, tz='UTC') + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        
        # 检查缓存数据的时间范围是否包含请求的时间范围
        cache_start = df_cached.index[0]
        cache_end = df_cached.index[-1]
        
        if cache_start > start_ts_filter or cache_end < end_ts_filter:
            logger.info(f"⚠️ 缓存数据时间范围不匹配")
            logger.info(f"   缓存: {cache_start} to {cache_end}")
            logger.info(f"   请求: {start_ts_filter} to {end_ts_filter}")
            logger.info(f"🔄 强制重新获取")
        else:
            df_filtered = df_cached[(df_cached.index >= start_ts_filter) & (df_cached.index <= end_ts_filter)]
            
            if len(df_filtered) >= required_bars * 0.95:  # 允许95%的容错
                logger.info(f"✅ 使用缓存数据: {len(df_filtered)} 条")
                return df_filtered
            else:
                logger.info(f"⚠️ 缓存数据不足: {len(df_filtered)}/{required_bars}, 重新获取")
    
    # 分批获取数据（交易所限制每次1000根）
    all_data = []
    current_since = start_ts
    batch_count = 0
    
    logger.info(f"🔄 开始分批获取数据...")
    
    while current_since < end_ts:
        batch_count += 1
        logger.info(f"📥 第 {batch_count} 批: since={pd.Timestamp(current_since, unit='ms')}")
        
        try:
            # 每次获取1000根（禁用缓存以确保获取正确的时间范围）
            df_batch = fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                limit=1000,
                exchange_name=exchange_name,
                since=current_since,
                use_cache=False  # 禁用缓存
            )
            
            if df_batch.empty:
                logger.warning(f"⚠️ 第 {batch_count} 批数据为空，停止获取")
                break
            
            all_data.append(df_batch)
            logger.info(f"✅ 第 {batch_count} 批获取 {len(df_batch)} 条数据")
            
            # 更新 since 为最后一条数据的时间 + 1个时间周期
            last_timestamp = df_batch.index[-1]
            current_since = int(last_timestamp.timestamp() * 1000) + tf_ms
            
            # 如果已经超过结束时间，停止获取
            if current_since >= end_ts:
                logger.info(f"✅ 已达到结束时间，停止获取")
                break
            
            # 避免过快请求（仅当需要多批次时）
            if batch_count > 1:
                time.sleep(1)  # 等待1秒
        
        except Exception as e:
            logger.error(f"❌ 第 {batch_count} 批获取失败: {e}")
            break
    
    if not all_data:
        logger.error("❌ 未获取到任何数据")
        return pd.DataFrame()
    
    # 合并所有批次的数据
    logger.info(f"🔀 合并 {len(all_data)} 批数据...")
    df_all = pd.concat(all_data)
    
    # 去除重复数据（按索引去重）
    df_all = df_all[~df_all.index.duplicated(keep='first')]
    
    # 过滤时间范围（转换为Timestamp进行比较）
    start_ts_filter = pd.Timestamp(start_date, tz='UTC')
    end_ts_filter = pd.Timestamp(end_date, tz='UTC') + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)  # 包含结束日的所有数据
    
    df_filtered = df_all[(df_all.index >= start_ts_filter) & (df_all.index <= end_ts_filter)]
    
    # 按时间排序
    df_filtered = df_filtered.sort_index()
    
    if df_filtered.empty:
        logger.warning(f"⚠️ 时间过滤后数据为空")
        logger.info(f"原始数据时间范围: {df_all.index[0]} to {df_all.index[-1]}")
        logger.info(f"请求时间范围: {start_ts_filter} to {end_ts_filter}")
        return pd.DataFrame()
    
    logger.info(f"✅ 最终获取 {len(df_filtered)} 条数据 (from {df_filtered.index[0]} to {df_filtered.index[-1]})")
    
    # 保存到缓存（更新缓存）
    if not df_all.empty:
        save_data_to_cache(df_all, symbol, timeframe, exchange_name)
    
    return df_filtered
