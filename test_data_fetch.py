import sys
sys.path.insert(0, ".")

from butterfly_bot.data.fetcher import fetch_historical_data
from butterfly_bot.config.settings import SYMBOL, TIMEFRAME
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# 测试3个月数据获取
print("=" * 60)
print("测试3个月数据获取功能")
print("=" * 60)

df = fetch_historical_data(
    symbol=SYMBOL,
    start_date="2023-09-01",
    end_date="2023-11-30",
    timeframe=TIMEFRAME
)

print("\n" + "=" * 60)
print("数据获取结果:")
print("=" * 60)
print(f"总K线数: {len(df)}")
print(f"开始时间: {df.index[0]}")
print(f"结束时间: {df.index[-1]}")
print(f"时间跨度: {(df.index[-1] - df.index[0]).days} 天")
print("\n前5条数据:")
print(df.head())
print("\n后5条数据:")
print(df.tail())
