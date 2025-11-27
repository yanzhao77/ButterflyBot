#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
获取1年历史数据用于扩展回测
"""

import sys
import os
from datetime import datetime, timezone, timedelta
import pandas as pd
import ccxt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import SYMBOL, TIMEFRAME, BASE_PATH

print("=" * 80)
print("获取1年DOGE/USDT历史数据")
print("=" * 80)

# 获取过去1年的数据
since_date = datetime.now(timezone.utc) - timedelta(days=365)
since_ts = int(since_date.timestamp() * 1000)

print(f"\n配置:")
print(f"  交易对: {SYMBOL}")
print(f"  周期: {TIMEFRAME}")
print(f"  起始日期: {since_date.strftime('%Y-%m-%d')}")
print(f"  目标数量: ~35,000条 (1年)")

# 初始化交易所
exchange = ccxt.binance({
    'enableRateLimit': True,
    'options': {'defaultType': 'spot'}
})

print(f"\n开始获取数据...")

all_data = []
current_since = since_ts
limit = 1000  # 每次请求1000条

while True:
    try:
        ohlcv = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, since=current_since, limit=limit)
        
        if not ohlcv:
            break
        
        all_data.extend(ohlcv)
        
        # 更新since为最后一条数据的时间+1
        last_ts = ohlcv[-1][0]
        current_since = last_ts + 1
        
        # 打印进度
        last_date = datetime.fromtimestamp(last_ts / 1000, tz=timezone.utc)
        print(f"  已获取: {len(all_data)} 条, 最新日期: {last_date.strftime('%Y-%m-%d %H:%M')}")
        
        # 如果已经到达当前时间，停止
        if last_ts >= int(datetime.now(timezone.utc).timestamp() * 1000):
            break
        
        # 避免请求过快
        import time
        time.sleep(exchange.rateLimit / 1000)
        
    except Exception as e:
        print(f"⚠️  获取数据出错: {e}")
        break

if not all_data:
    print("❌ 未获取到任何数据")
    sys.exit(1)

# 转换为DataFrame
df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)

# 去重（可能有重复数据）
df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)

print(f"\n✅ 数据获取完成!")
print(f"  总条数: {len(df)}")
print(f"  时间范围: {df['timestamp'].iloc[0]} 至 {df['timestamp'].iloc[-1]}")
print(f"  价格范围: {df['close'].min():.6f} - {df['close'].max():.6f}")

# 保存到缓存
cache_dir = BASE_PATH / 'cached_data'
cache_dir.mkdir(exist_ok=True)
filename = f"binance_{SYMBOL.replace('/', '_')}_{TIMEFRAME}_1year.csv"
cache_path = cache_dir / filename

df.to_csv(cache_path, index=False)
print(f"\n💾 数据已保存: {cache_path}")

# 同时更新主缓存文件
main_cache = cache_dir / f"binance_{SYMBOL.replace('/', '_')}_{TIMEFRAME}.csv"
df.to_csv(main_cache, index=False)
print(f"💾 主缓存已更新: {main_cache}")

print("\n" + "=" * 80)
