# -*- coding: utf-8 -*-
# config/settings.py
import os
from pathlib import Path

# 智谱 AI API 配置
ZHIPU_API_KEY = "0d62ab35210808d52040993cd53788a5.NrIcZR8TpXjbUwrz"  # 替换为您的智谱API密钥
ZHIPU_MODEL = "glm-4-flash"  # 或其他智谱支持的模型
ZHIPU_API_TIMEOUT = 10  # API 超时时间（秒）

# 实盘 API（仅 USE_REAL_MONEY=True 时需要）
API_KEY = "your_api_key_here"
API_SECRET = "your_api_secret_here"

# 实盘开关
USE_REAL_MONEY = False  # 设为 True 启用实盘

# 回测参数 初始资金（模拟盘用）
INITIAL_CASH = 1000.0  # 增加初始资金以获得更多交易机会

# config/settings.py
SYMBOL = "DOGE/USDT"
EXCHANGE_NAME = "binance"
TIMEFRAME = "15m"  # 支持: 1m, 5m, 15m, 1h, 4h, 1d

# 风控参数
MAX_RISK_PER_TRADE = 0.015  # 单笔最大风险 1.5%
STOP_LOSS_PCT = 0.008  # 默认止损 1.5%（收紧止损）
TAKE_PROFIT_PCT = 0.01  # 默认止盈 3%（降低止盈目标）
# 风控
MAX_POSITION_RATIO = 0.3  # 最大使用资金比例（30%，降低仓位）

# 策略参数（传递给 AISignalCore/策略）
CONFIDENCE_THRESHOLD = 0.47  # 买入置信阈值（降低到0.55，获得交易机会）
SELL_THRESHOLD = 0.45  # 卖出/平仓阈值（降低到0.45）
TREND_FILTER = True
COOLDOWN_BARS = 8  # 平仓/开仓后的冷却条数（增加冷却期）
PROB_EMA_SPAN = 10  # 预测概率EMA平滑窗口（增加平滑度）
TIME_STOP_BARS = 20  # 时间止损：持仓超过N根K线未验证则平仓（缩短至5小时）
USE_QUANTILE_THRESH = True  # 使用分位数自适应阈值
PROB_Q_HIGH = 0.8  # 买入触发的高分位
PROB_Q_LOW = 0.5  # 卖出触发的低分位（回转带）
PROB_WINDOW = 200  # 分位数计算窗口大小
REQUIRE_P_EMA_UP = True  # 仅当 p_ema 动量向上时允许开多
P_EMA_MOMENTUM_BARS = 3  # 动量判断窗口（最近N根 p_ema 需上升）
TRADE_ONLY_ON_CANDLE_CLOSE = True  # 仅在K线闭合时交易；调试可设为 False 支持同K线内交易
TARGET_SHIFT = 2  # 新增：预测未来鏱-2根K线（不远远不走）
TARGET_THRESHOLD = 0.012  # 新增：1.2%涨幅阈值（投资玻记）

# 训练参数
TRAIN_TEST_SPLIT_RATIO = 0.7  # 80% 训练，20% 测试（时间顺序）

# 回测/实时策略窗口与特征最小行数
# FEATURE_WINDOW: 策略每次预测时使用的历史 K 线数量（滑动窗口大小）
FEATURE_WINDOW = 200  # 降低上限（原個 500），加快计算
# MIN_FEATURE_ROWS: 当特征工程后有效行数少于此值时，跳过预测
MIN_FEATURE_ROWS = 30  # 降低上限（原個 60），更容易成一个所了
# 为计算满量/EMA 等指标预留的额外历史
FEATURE_HISTORY_PADDING = 60

# 自动重训练/更新相关配置
# 是否在回测检测到性能下降时触发重训练
RETRAIN_ON_DEGRADATION = True
# 如果回测 AUC 比训练 AUC 低于此阈值则触发重训练（绝对差值）
RETRAIN_AUC_DIFF = 0.01
# 当回测收益为负且 RETRAIN_ON_DEGRADATION 为 True 时也会触发重训练
# 重训练拉取历史天数（默认 365 天，醍渐降低为 180）
RETRAIN_SINCE_DAYS = 1095  # 降低到 180 天，队失会成
# 重训练时的最大 K 线条数（fetch limit）
RETRAIN_LIMIT = 1000000
# 是否将重训练放到后台线程异步执行（避免阻塞回测流程）
RETRAIN_ASYNC = True
RETRAIN_MAX_ATTEMPTS = 2  # 自动重训练最大尝试次数

proxy = 'http://127.0.0.1:7890'  # 替换为你的代理地址（Clash 默认 7890，SS 通常是 1080）

BASE_PATH = Path(__file__).resolve().parent.parent
LOG_PATH = BASE_PATH / 'logs'
REGISTRY_DIR = BASE_PATH / "models/registry"

# 路径配置
MODEL_SAVE_PATH = BASE_PATH / "/model/saved/lgb_model.pkl"
MODEL_METRICS_PATH = BASE_PATH / "/backtest/strategy/strategy_metrics.json"
