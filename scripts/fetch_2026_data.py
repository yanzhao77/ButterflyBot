"""
拉取 DOGE/USDT 15m 2026年最新数据（2026-01-01 至今）
"""
import sys, os, time, logging
sys.path.insert(0, ".")
import ccxt
import pandas as pd
from datetime import datetime, timezone

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CACHE_DIR = "butterfly_bot/cached_data"
os.makedirs(CACHE_DIR, exist_ok=True)

SYMBOL    = "DOGE/USDT"
TIMEFRAME = "15m"
START_DATE = "2026-01-01 00:00:00"
# 拉取到当前时间
END_DATE   = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

def fetch_all_ohlcv(symbol, timeframe, start_str, end_str):
    exchange = ccxt.binance({
        "enableRateLimit": True,
        "timeout": 30000,
        "options": {"defaultType": "spot"},
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
            break
        batch = [r for r in batch if r[0] < end_ts]
        if not batch:
            break
        all_rows.extend(batch)
        since = batch[-1][0] + timeframe_ms
        logger.info(f"第{page}页: {len(batch)}条, 累计{len(all_rows)}条, "
                    f"最新时间: {datetime.utcfromtimestamp(batch[-1][0]/1000).strftime('%Y-%m-%d %H:%M')}")
        time.sleep(0.2)
    return all_rows

def main():
    logger.info(f"开始拉取 {SYMBOL} {TIMEFRAME} 数据: {START_DATE} ~ {END_DATE}")
    rows = fetch_all_ohlcv(SYMBOL, TIMEFRAME, START_DATE, END_DATE)
    if not rows:
        logger.error("未获取到任何数据！")
        return
    df = pd.DataFrame(rows, columns=["timestamp","open","high","low","close","volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp").sort_index().drop_duplicates()
    out_path = os.path.join(CACHE_DIR, "binance_DOGE_USDT_15m_2026.csv")
    df.to_csv(out_path)
    logger.info(f"✅ 2026年数据已保存: {out_path}")
    logger.info(f"   行数: {len(df)}, 时间范围: {df.index[0]} ~ {df.index[-1]}")

if __name__ == "__main__":
    main()
