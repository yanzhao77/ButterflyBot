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

# 风控参数（优化后）
MAX_RISK_PER_TRADE = 0.025  # 单笔最大风险 2.5%
STOP_LOSS_PCT = 0.02  # 默认止损 2.5%（放宽止损，避免被正常波动扫出）
TAKE_PROFIT_PCT = 0.03  # 默认止盈 5%（提高止盈目标，盈亏比2:1）
# 风控
MAX_POSITION_RATIO = 0.25  # 最大使用资金比例（25%，降低风险敞口）

# 策略参数（传递给 AISignalCore/策略）- 优化后（保守版v2）
CONFIDENCE_THRESHOLD = 0.70  # 买入置信阈值（提高到0.70，大幅降低交易频率）
SELL_THRESHOLD = 0.38  # 卖出/平仓阈值（保持0.38，避免过早离场）
TREND_FILTER = True
COOLDOWN_BARS = 10  # 平仓/开仓后的冷却条数（增加到10，约2.5小时）
PROB_EMA_SPAN = 10  # 预测概率EMA平滑窗口（保持10）
TIME_STOP_BARS = 50  # 时间止损：持仓超过N根K线未验证则平仓（延长至50，约12.5小时）
USE_QUANTILE_THRESH = True  # 使用分位数自适应阈值
PROB_Q_HIGH = 0.90  # 买入触发的高分位（提高到0.90，极度严格）
PROB_Q_LOW = 0.35  # 卖出触发的低分位（降低到0.35，避免过早离场）
PROB_WINDOW = 300  # 分位数计算窗口大小（增加到300，更稳定）
REQUIRE_P_EMA_UP = True  # 仅当 p_ema 动量向上时允许开多
P_EMA_MOMENTUM_BARS = 3  # 动量判断窗口（最近N根 p_ema 需上升）
TRADE_ONLY_ON_CANDLE_CLOSE = True  # 仅在K线闭合时交易；调试可设为 False 支持同K线内交易
TARGET_SHIFT = 4  # 优化v3：预测未来4根K线（1小时，平衡噪音和机会）
TARGET_THRESHOLD = 0.015  # 优化v3：涨幅阈值1.5%（更容易达到，增加正样本）

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
# 重训练拉取历史天数（优化为365天，平衡数据量和训练速度）
RETRAIN_SINCE_DAYS = 30  # 优化到365天，关注近期市场特征
# 重训练时的最大 K 线条数（fetch limit）
RETRAIN_LIMIT = 5000
# 是否将重训练放到后台线程异步执行（避免阻塞回测流程）
RETRAIN_ASYNC = True
RETRAIN_MAX_ATTEMPTS = 2  # 自动重训练最大尝试次数

proxy = None  # 替换为你的代理地址（Clash 默认 7890，SS 通常是 1080）

BASE_PATH = Path(__file__).resolve().parent.parent
LOG_PATH = BASE_PATH / 'logs'
REGISTRY_DIR = BASE_PATH / "models/registry"

# 路径配置
MODEL_SAVE_PATH = BASE_PATH / "/model/saved/lgb_model.pkl"
MODEL_METRICS_PATH = BASE_PATH / "/backtest/strategy/strategy_metrics.json"

# 跟踪止盈配置（新增优化功能）- 优化版
USE_TRAILING_STOP = True  # 启用跟踪止盈
TRAILING_STOP_ACTIVATION = 0.02  # 盈利达到2%后启动跟踪（更早启动）
TRAILING_STOP_DISTANCE = 0.01  # 从最高点回撤1%时止盈（更紧跟踪）

AI_SIGNAL_CONFIG = {
    "confidence_threshold": CONFIDENCE_THRESHOLD,
    "cooldown_bars": COOLDOWN_BARS,
    "trend_filter": TREND_FILTER,
}

RISK_MANAGEMENT_CONFIG = {
    "max_risk_per_trade": MAX_RISK_PER_TRADE,
    "stop_loss_pct": STOP_LOSS_PCT,
    "take_profit_pct": TAKE_PROFIT_PCT,
    "max_position_ratio": MAX_POSITION_RATIO,
}

BACKTEST_CONFIG = {
    "initial_balance": INITIAL_CASH,
    "leverage": 1,
    "contract_type": "spot",
    "start_date": "2023-11-01",
    "end_date": "2023-11-30",
}
