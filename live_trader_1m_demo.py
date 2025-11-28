#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实时交易系统演示 - 1分钟K线
快速演示WebSocket实时交易功能
"""

import time
from datetime import datetime
from collections import deque
from binance import ThreadedWebsocketManager

class LiveTrader1mDemo:
    """1分钟K线演示版本"""
    
    def __init__(self):
        self.symbol = 'DOGEUSDT'
        self.interval = '1m'
        self.klines = deque(maxlen=50)
        self.twm = None
        self.bar_count = 0
        
        print("="*60)
        print("实时交易系统演示 - 1分钟K线")
        print("="*60)
        print(f"交易对: {self.symbol}")
        print(f"时间框架: {self.interval}")
        print("="*60)
    
    def handle_kline(self, msg):
        """处理K线消息"""
        if msg['e'] != 'kline':
            return
        
        kline = msg['k']
        is_closed = kline['x']
        
        timestamp = datetime.fromtimestamp(kline['t']/1000)
        
        if is_closed:
            # K线完成
            self.bar_count += 1
            kline_data = {
                'timestamp': kline['t'],
                'open': float(kline['o']),
                'high': float(kline['h']),
                'low': float(kline['l']),
                'close': float(kline['c']),
                'volume': float(kline['v']),
            }
            self.klines.append(kline_data)
            
            print(f"\n{'='*60}")
            print(f"[{timestamp}] K线 #{self.bar_count} 完成")
            print(f"  开: {kline_data['open']:.5f}")
            print(f"  高: {kline_data['high']:.5f}")
            print(f"  低: {kline_data['low']:.5f}")
            print(f"  收: {kline_data['close']:.5f}")
            print(f"  量: {kline_data['volume']:.2f}")
            print(f"  缓存K线数: {len(self.klines)}")
            print(f"{'='*60}")
            
            # 模拟交易逻辑
            if self.bar_count % 3 == 0:
                print("📊 模拟: 计算特征和预测...")
                print("📈 模拟: 生成交易信号...")
            
            if self.bar_count >= 10:
                print("\n✅ 演示完成！已接收10根K线")
                print("实际使用时，系统会持续运行并执行交易")
                self.stop()
        else:
            # K线更新中
            current_price = float(kline['c'])
            print(f"\r[{timestamp}] 当前价格: {current_price:.5f} (K线更新中...)", end='', flush=True)
    
    def start(self):
        """启动系统"""
        print("\n🚀 启动WebSocket监听...")
        
        self.twm = ThreadedWebsocketManager()
        self.twm.start()
        
        self.twm.start_kline_socket(
            callback=self.handle_kline,
            symbol=self.symbol,
            interval=self.interval
        )
        
        print("✅ 系统已启动！")
        print("等待K线完成信号...\n")
        
        try:
            while self.bar_count < 10:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n\n🛑 用户中断")
            self.stop()
    
    def stop(self):
        """停止系统"""
        if self.twm:
            self.twm.stop()
        print("\n✅ 系统已停止")

if __name__ == "__main__":
    demo = LiveTrader1mDemo()
    demo.start()
