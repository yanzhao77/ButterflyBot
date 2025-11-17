#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
获取真实历史数据用于回测
"""

import sys
import os
import ccxt
import pandas as pd
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import SYMBOL, TIMEFRAME, EXCHANGE_NAME, BASE_PATH

print("=" * 80)
print("获取真实历史数据")
print("=" * 80)

def fetch_historical_data(days=180):
    """
    从 Binance 获取真实历史数据
    """
    print(f"\n📊 配置信息:")
    print(f"   交易对: {SYMBOL}")
    print(f"   周期: {TIMEFRAME}")
    print(f"   交易所: {EXCHANGE_NAME}")
    print(f"   获取天数: {days}")
    
    # 创建缓存目录
    cache_dir = BASE_PATH / 'cached_data'
    os.makedirs(cache_dir, exist_ok=True)
    
    # 初始化交易所
    try:
        print(f"\n🔌 连接到 {EXCHANGE_NAME}...")
        exchange = ccxt.binance({
            'enableRateLimit': True,
            'timeout': 30000,
            'options': {'defaultType': 'spot'},
        })
        
        # 计算起始时间
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        since = int(start_time.timestamp() * 1000)
        
        print(f"   时间范围: {start_time.strftime('%Y-%m-%d')} 至 {end_time.strftime('%Y-%m-%d')}")
        
        # 分批获取数据
        all_data = []
        current_since = since
        batch_count = 0
        
        print(f"\n📥 开始获取数据...")
        
        while True:
            try:
                # 每次获取1000条
                ohlcv = exchange.fetch_ohlcv(
                    SYMBOL, 
                    timeframe=TIMEFRAME, 
                    since=current_since, 
                    limit=1000
                )
                
                if not ohlcv:
                    break
                
                all_data.extend(ohlcv)
                batch_count += 1
                
                # 更新进度
                last_timestamp = ohlcv[-1][0]
                last_time = datetime.fromtimestamp(last_timestamp / 1000)
                print(f"   批次 {batch_count}: 获取 {len(ohlcv)} 条，最新时间 {last_time.strftime('%Y-%m-%d %H:%M')}")
                
                # 检查是否已经获取到最新数据
                if len(ohlcv) < 1000:
                    break
                
                # 更新起始时间
                current_since = last_timestamp + 1
                
                # 如果已经超过当前时间，停止
                if last_timestamp >= int(end_time.timestamp() * 1000):
                    break
                
            except Exception as e:
                print(f"   ⚠️  获取数据出错: {e}")
                break
        
        if not all_data:
            print("\n❌ 未获取到任何数据")
            return None
        
        # 转换为 DataFrame
        print(f"\n✅ 总共获取 {len(all_data)} 条数据")
        
        df = pd.DataFrame(
            all_data,
            columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        
        # 转换时间戳
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("timestamp", inplace=True)
        
        # 类型转换
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        
        # 排序并去重
        df.sort_index(inplace=True)
        df = df[~df.index.duplicated(keep='first')]
        
        # 显示数据信息
        print(f"\n📈 数据统计:")
        print(f"   时间范围: {df.index[0]} 至 {df.index[-1]}")
        print(f"   数据条数: {len(df)}")
        print(f"   价格范围: {df['close'].min():.6f} - {df['close'].max():.6f}")
        print(f"   平均成交量: {df['volume'].mean():.0f}")
        
        # 保存到缓存
        filename = f"{EXCHANGE_NAME}_{SYMBOL.replace('/', '_')}_{TIMEFRAME}.csv"
        cache_path = cache_dir / filename
        
        df.to_csv(cache_path)
        print(f"\n💾 数据已保存至: {cache_path}")
        
        # 显示最近几条数据
        print(f"\n📊 最近5条数据:")
        print(df.tail(5).to_string())
        
        return df
        
    except Exception as e:
        print(f"\n❌ 获取数据失败: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    try:
        df = fetch_historical_data(days=180)
        if df is not None:
            print("\n" + "=" * 80)
            print("✅ 数据获取成功！")
            print("=" * 80)
            sys.exit(0)
        else:
            print("\n" + "=" * 80)
            print("❌ 数据获取失败")
            print("=" * 80)
            sys.exit(1)
    except Exception as e:
        print(f"\n💥 程序异常: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
