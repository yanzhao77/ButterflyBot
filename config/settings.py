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
INITIAL_CASH = 1000.0

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

# 回测/实时策略窗口与特征最小行数
# FEATURE_WINDOW: 策略每次预测时使用的历史 K 线数量（滑动窗口大小），默认 200
# MIN_FEATURE_ROWS: 当特征工程后有效行数少于此值时，跳过预测（避免模型输入为空）
FEATURE_WINDOW = 300  # 增加窗口长度，提升信号覆盖率
MIN_FEATURE_ROWS = 60  # 增加最小有效行数，保证模型输入质量
# 为计算滚动/EMA 等指标预留的额外历史长度（默认 120），
# 回测/实时窗口会在 FEATURE_WINDOW 基础上向前拉取这部分数据用于计算指标，
# 以避免因滚动窗口导致的 dropna 而使有效行数不足。
FEATURE_HISTORY_PADDING = 120

# 自动重训练/更新相关配置
# 是否在回测检测到性能下降时触发重训练
RETRAIN_ON_DEGRADATION = True
# 如果回测 AUC 比训练 AUC 低于此阈值则触发重训练（绝对差值）
RETRAIN_AUC_DIFF = 0.01
# 当回测收益为负且 RETRAIN_ON_DEGRADATION 为 True 时也会触发重训练
# 重训练拉取历史天数（默认 365 天）
RETRAIN_SINCE_DAYS = 365
# 重训练时的最大 K 线条数（fetch limit）
RETRAIN_LIMIT = 10000
# 是否将重训练放到后台线程异步执行（避免阻塞回测流程）
RETRAIN_ASYNC = True
RETRAIN_MAX_ATTEMPTS = 20  # 自动重训练最大尝试次数

proxy = 'http://127.0.0.1:7890'  # 替换为你的代理地址（Clash 默认 7890，SS 通常是 1080）

BASE_PATH = Path(__file__).resolve().parent.parent
LOG_PATH = BASE_PATH / 'logs'
REGISTRY_DIR = BASE_PATH / "models/registry"
