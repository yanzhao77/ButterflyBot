"""
拉取 DOGE/USDT 15m 近5年历史数据（约 2020-01-01 ~ 2025-12-23）
使用 ccxt 分页抓取，保存到缓存文件
"""
import sys, os, time, logging
sys.path.insert(0, ".")

import ccxt
import pandas as pd
from datetime import datetime, timezone
from butterfly_bot.config.settings import EXCHANGE_NAME, SYMBOL, TIMEFRAME

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CACHE_DIR = "butterfly_bot/cached_data"
os.makedirs(CACHE_DIR, exist_ok=True)

SYMBOL = "DOGE/USDT"
TIMEFRAME = "15m"
# 5年前：2020-04-28
START_DATE = "2020-04-28 00:00:00"
END_DATE   = "2025-12-23 02:00:00"

def fetch_all_ohlcv(symbol, timeframe, start_str, end_str):
    """分页拉取所有历史数据"""
    import ccxt

    # 尝试读取代理配置
    proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY") or ""
    proxies = {"http": proxy, "https": proxy} if proxy else {}

    exchange = ccxt.binance({
        "enableRateLimit": True,
        "timeout": 30000,
        "options": {"defaultType": "spot"},
        **({"proxies": proxies} if proxies else {}),
    })
    exchange.load_markets()

    start_ts = int(datetime.strptime(start_str, "%Y-%m-%d %H:%M:%S")
                   .replace(tzinfo=timezone.utc).timestamp() * 1000)
    end_ts   = int(datetime.strptime(end_str,   "%Y-%m-%d %H:%M:%S")
                   .replace(tzinfo=timezone.utc).timestamp() * 1000)

    timeframe_ms = int(exchange.parse_timeframe(timeframe) * 1000)
    all_rows = []
    since = start_ts
    page = 0

    while since < end_ts:
        page += 1
        try:
            batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=1000)
        except Exception as e:
            logger.warning(f"第{page}页请求失败: {e}，3秒后重试...")
            time.sleep(3)
            continue

        if not batch:
            logger.info("没有更多数据，停止抓取")
            break

        # 过滤超出 end_ts 的数据
        batch = [row for row in batch if row[0] < end_ts]
        all_rows.extend(batch)
        fetched_total = len(all_rows)

        last_ts = batch[-1][0]
        last_dt = datetime.fromtimestamp(last_ts / 1000, tz=timezone.utc)
        logger.info(f"第{page}页: 获取 {len(batch)} 条，累计 {fetched_total} 条，最新时间: {last_dt}")

        if len(batch) < 1000:
            logger.info("最后一页，停止抓取")
            break

        since = last_ts + timeframe_ms
        time.sleep(0.3)  # 避免触发限速

    return all_rows


def main():
    logger.info(f"开始拉取 {SYMBOL} {TIMEFRAME} 5年历史数据")
    logger.info(f"时间范围: {START_DATE} ~ {END_DATE}")

    rows = fetch_all_ohlcv(SYMBOL, TIMEFRAME, START_DATE, END_DATE)

    if not rows:
        logger.error("未获取到任何数据！")
        sys.exit(1)

    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp")
    df = df[~df.index.duplicated(keep="last")]
    df = df.sort_index()

    logger.info(f"数据汇总: {len(df)} 根K线")
    logger.info(f"时间范围: {df.index[0]} ~ {df.index[-1]}")

    # 保存到缓存文件
    cache_file = os.path.join(CACHE_DIR, f"binance_DOGE_USDT_15m.csv")
    df.to_csv(cache_file)
    logger.info(f"✅ 数据已保存到: {cache_file}")

    # 同时保存一份带时间戳的备份
    backup_file = os.path.join(CACHE_DIR, f"binance_DOGE_USDT_15m_5year.csv")
    df.to_csv(backup_file)
    logger.info(f"✅ 备份已保存到: {backup_file}")

    return df


if __name__ == "__main__":
    main()
