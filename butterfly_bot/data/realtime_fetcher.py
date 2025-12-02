"""
实时数据获取模块
用于获取实时K线数据和当前价格
"""

import logging
import ccxt
import pandas as pd
from typing import Optional
from ..config.settings import EXCHANGE_NAME, proxy

logger = logging.getLogger(__name__)


class RealtimeFetcher:
    """实时数据获取器"""
    
    def __init__(self, exchange_name: str = EXCHANGE_NAME):
        """初始化实时数据获取器
        
        Args:
            exchange_name: 交易所名称
        """
        self.exchange_name = exchange_name
        self.exchange = self._init_exchange()
        logger.info(f"实时数据获取器初始化: 交易所={exchange_name}")
    
    def _init_exchange(self):
        """初始化交易所连接"""
        try:
            exchange_class = getattr(ccxt, self.exchange_name)
            exchange = exchange_class({
                'enableRateLimit': True,
                'timeout': 30000,
                'options': {'defaultType': 'spot'},
                'proxies': {
                    'http': proxy,
                    'https': proxy,
                } if proxy else {}
            })
            exchange.load_markets()
            return exchange
        except Exception as e:
            logger.error(f"交易所初始化失败: {e}")
            raise
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """获取当前价格
        
        Args:
            symbol: 交易对，如 "BTC/USDT"
            
        Returns:
            当前价格，失败返回None
        """
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            price = ticker['last']
            logger.debug(f"获取当前价格: {symbol} = {price}")
            return float(price)
        except Exception as e:
            logger.error(f"获取当前价格失败: {e}")
            return None
    
    def get_recent_klines(
        self,
        symbol: str,
        timeframe: str = '1h',
        limit: int = 200
    ) -> pd.DataFrame:
        """获取最近的K线数据
        
        Args:
            symbol: 交易对
            timeframe: 时间周期
            limit: 获取数量
            
        Returns:
            K线数据DataFrame
        """
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            if not ohlcv:
                logger.warning(f"未获取到K线数据: {symbol}")
                return pd.DataFrame()
            
            df = pd.DataFrame(
                ohlcv,
                columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
            
            # 转换时间戳
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df.set_index("timestamp", inplace=True)
            
            # 类型转换
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            
            logger.info(f"获取K线数据成功: {symbol}, {len(df)}根K线")
            return df
            
        except Exception as e:
            logger.error(f"获取K线数据失败: {e}")
            return pd.DataFrame()
    
    def get_orderbook(self, symbol: str, limit: int = 10) -> dict:
        """获取订单簿
        
        Args:
            symbol: 交易对
            limit: 深度
            
        Returns:
            订单簿数据
        """
        try:
            orderbook = self.exchange.fetch_order_book(symbol, limit)
            return {
                'bids': orderbook['bids'][:limit],  # 买单
                'asks': orderbook['asks'][:limit],  # 卖单
                'timestamp': orderbook['timestamp']
            }
        except Exception as e:
            logger.error(f"获取订单簿失败: {e}")
            return {'bids': [], 'asks': [], 'timestamp': None}
