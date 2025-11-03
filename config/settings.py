# config/settings.py
import os
from pathlib import Path

# 实盘 API（仅 USE_REAL_MONEY=True 时需要）
API_KEY = "your_api_key_here"
API_SECRET = "your_api_secret_here"

# 路径配置
MODEL_SAVE_PATH = "./model/saved/lgb_model.pkl"
MODEL_METRICS_PATH = "./backtest/strategy_metrics.json"

# 实盘开关
USE_REAL_MONEY = False  # 设为 True 启用实盘

# 回测参数 初始资金（模拟盘用）
INITIAL_CASH = 10000.0

# config/settings.py
SYMBOL = "DOGE/USDT"
EXCHANGE_NAME = "binance"
TIMEFRAME = "1h"  # 支持: 1m, 5m, 15m, 1h, 4h, 1d

# 风控参数
MAX_RISK_PER_TRADE = 0.02  # 单笔最大风险 2%
STOP_LOSS_PCT = 0.05  # 默认止损 1.5%
TAKE_PROFIT_PCT = 0.03  # 默认止盈 3%
# 风控
MAX_POSITION_RATIO = 0.9  # 最大使用资金比例（90%）

# 策略参数（传递给 AISignalCore）
CONFIDENCE_THRESHOLD = 0.65
TREND_FILTER = True
COOLDOWN_BARS = 2

# 训练参数
TRAIN_TEST_SPLIT_RATIO = 0.8  # 80% 训练，20% 测试（时间顺序）

proxy = 'http://127.0.0.1:7890'  # 替换为你的代理地址（Clash 默认 7890，SS 通常是 1080）

BASE_PATH = Path(__file__).resolve().parent.parent
LOG_PATH = BASE_PATH / 'logs'
REGISTRY_DIR = BASE_PATH / "models/registry"
