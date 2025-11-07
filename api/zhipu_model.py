"""
智谱 AI 模型封装
使用智谱 API 进行市场预测
"""

import time

import pandas as pd
import zhipuai

from config.settings import ZHIPU_API_KEY, ZHIPU_MODEL, ZHIPU_API_TIMEOUT


class ZhiPuModel:
    def __init__(self, timeframe):
        """初始化智谱AI模型
        
        Args:
            timeframe (str): K线周期，如 "1h"
        """
        self.timeframe = timeframe
        zhipuai.api_key = ZHIPU_API_KEY
        self._last_call_time = 0
        self._min_interval = 1  # 最小调用间隔（秒）

    def _rate_limit(self):
        """简单的频率限制"""
        now = time.time()
        if now - self._last_call_time < self._min_interval:
            time.sleep(self._min_interval - (now - self._last_call_time))
        self._last_call_time = time.time()

    def _format_market_data(self, df):
        """格式化最近的市场数据，生成提示词
        
        Args:
            df (pd.DataFrame): 市场数据，包含 OHLCV 和技术指标
        
        Returns:
            str: 格式化后的市场描述
        """
        # 确保数据是最新的（取最后一行）
        latest = df.iloc[-1]
        
        # 计算一些简单的变化率
        price_change = (latest['close'] - latest['open']) / latest['open'] * 100
        ma_trend = "上涨" if latest['ma20'] > latest['ma50'] else "下跌"
        rsi_status = "超买" if latest['rsi'] > 70 else "超卖" if latest['rsi'] < 30 else "中性"
        
        prompt = f"""基于以下市场数据分析，预测未来短期价格走势的概率（用0-1之间的数字表示上涨概率）：

1. 最新价格状况：
- 开盘价：{latest['open']:.4f}
- 最高价：{latest['high']:.4f}
- 最低价：{latest['low']:.4f}
- 收盘价：{latest['close']:.4f}
- 成交量：{latest['volume']:.2f}

2. 技术指标：
- RSI：{latest['rsi']:.2f}（{rsi_status}）
- MACD：{latest['macd']:.4f}
- MA20和MA50趋势：{ma_trend}
- 价格变化率：{price_change:.2f}%

3. 波动指标：
- 波动率：{latest['volatility']:.4f}
- 成交量比率：{latest['volume_ratio']:.2f}

请直接返回一个0到1之间的数字，表示看涨概率。数字越大表示看涨概率越高，数字越小表示看跌概率越高。
例如：0.75表示75%的看涨概率。仅返回这个概率值，不要包含其他文字。
"""
        return prompt

    def predict(self, df):
        """使用智谱AI预测市场走势
        
        Args:
            df (pd.DataFrame): 包含市场数据的DataFrame
        
        Returns:
            float: 预测的上涨概率（0-1之间）
        """
        try:
            self._rate_limit()  # 频率限制
            
            # 生成提示词
            prompt = self._format_market_data(df)
            
            # 调用智谱API
            response = zhipuai.model_api.invoke(
                model=ZHIPU_MODEL,
                prompt=prompt,
                top_p=0.7,
                temperature=0.3,
                request_timeout=ZHIPU_API_TIMEOUT
            )
            
            if response["code"] != 200:
                print(f"❌ 智谱API调用失败: {response['msg']}")
                return 0.5  # 出错时返回中性预测
                
            # 解析响应
            try:
                prob = float(response["data"]["content"].strip())
                # 确保概率在0-1之间
                prob = max(0.0, min(1.0, prob))
                return prob
            except (ValueError, TypeError, KeyError) as e:
                print(f"⚠️ 解析智谱响应失败: {e}")
                return 0.5
                
        except Exception as e:
            print(f"❌ 智谱模型预测异常: {e}")
            return 0.5  # 发生异常时返回中性预测