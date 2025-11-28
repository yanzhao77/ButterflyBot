#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实时交易系统演示 - 1分钟K线
同步WebSocket实现，使用 websocket-client
"""

import json
from datetime import datetime
from collections import deque
import websocket


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
        data = json.loads(msg)
        if data.get('e') != 'kline':
            return

        kline = data['k']
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

        url = f"wss://stream.binance.com:9443/ws/{self.symbol.lower()}@kline_{self.interval}"

        def on_message(ws, message):
            self.handle_kline(message)

        def on_error(ws, error):
            print("\n❌ WebSocket 错误:", error)

        def on_close(ws, close_status_code, close_msg):
            print("\n🛑 WebSocket 已关闭")

        def on_open(ws):
            print("✅ WebSocket 已连接")

        self.twm = websocket.WebSocketApp(
            url,
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close
        )

        try:
            # 使用代理（可选）
            self.twm.run_forever(
                http_proxy_host="127.0.0.1",
                http_proxy_port=7890,
                proxy_type="http"  # socks5: "socks5"
            )
        except KeyboardInterrupt:
            print("\n\n🛑 用户中断")
            self.stop()
    
    def stop(self):
        """停止系统"""
        if self.twm:
            self.twm.stop()
        print("\n✅ 系统已停止")
        exit(0)


if __name__ == "__main__":
    demo = LiveTrader1mDemo()
    demo.start()
