# live/live_runner.py
"""
终极版 AI 量化实盘交易器
- 支持任意币种（默认 DOGE/USDT）
- 基于 AISignalCore 策略核心
- 内置止损、状态恢复、模拟盘开关
- 自动适配交易所精度与最小交易规则
"""

import json
import logging
import os
import time
from datetime import datetime, timezone

import pandas as pd

from config.settings import (
    SYMBOL,
    TIMEFRAME,
    EXCHANGE_NAME,
    INITIAL_CASH,
    USE_REAL_MONEY,
    MAX_POSITION_RATIO,
    STOP_LOSS_PCT,
    API_KEY,
    API_SECRET,
    CONFIDENCE_THRESHOLD,
    TREND_FILTER,
    COOLDOWN_BARS,
    proxy,
    REGISTRY_DIR,
    TRADE_ONLY_ON_CANDLE_CLOSE,
    LOG_PATH
)
from data.fetcher import fetch_ohlcv
from strategies.ai_signal_core import AISignalCore

# ======================
# 日志配置
# ======================
os.makedirs(LOG_PATH, exist_ok=True)
log_file = LOG_PATH / f"live_{SYMBOL.replace('/', '_')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# 在 _init_exchange 中

class LiveRunner:
    def __init__(self):
        self.symbol = SYMBOL
        self.timeframe = TIMEFRAME
        self.use_real_money = USE_REAL_MONEY

        # 初始化交易所
        self._init_exchange()

        # 初始化策略核心（使用你重写的逻辑）
        self.strategy = AISignalCore(
            symbol=self.symbol,
            timeframe=self.timeframe,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            trend_filter=TREND_FILTER,
            cooldown_bars=COOLDOWN_BARS
        )

        # 状态文件
        self.state_file = REGISTRY_DIR / f"live/state_{self.symbol.replace('/', '_')}.json"
        self.last_kline_timestamp = None
        self.last_close = None
        self.position = {"size": 0.0, "entry_price": 0.0}
        self.load_state()

        logger.info(f"🚀 启动 {'实盘' if self.use_real_money else '模拟'} 交易 | {self.symbol} @ {self.timeframe}")
        if self.use_real_money:
            usdt = self.get_usdt_balance()
            logger.info(f"💰 账户余额: {usdt:.2f} USDT")

    def _init_exchange(self):
        """初始化交易所（支持实盘/模拟）"""
        import ccxt
        exchange_class = getattr(ccxt, EXCHANGE_NAME)

        if self.use_real_money:
            self.exchange = ccxt.binance({
                'apiKey': API_KEY,
                'secret': API_SECRET,
                'enableRateLimit': True,
                'timeout': 30000,
                'options': {'defaultType': 'spot'},
                'proxies': {
                    'http': proxy,
                    'https': proxy,
                }
            })
        else:
            self.exchange = exchange_class({
                'enableRateLimit': True,
                'timeout': 30000,
                'options': {'defaultType': 'spot'},
                'proxies': {
                    'http': proxy,
                    'https': proxy,
                }})
        self.exchange.load_markets()
        # 获取市场规则（用于精度和最小量）
        market = self.exchange.market(self.symbol)
        self.price_precision = market['precision']['price']
        self.amount_precision = market['precision']['amount']
        self.min_amount = float(market['limits']['amount']['min'])
        self.min_cost = float(market['limits']['cost']['min'])

    def get_usdt_balance(self) -> float:
        if not self.use_real_money:
            return INITIAL_CASH
        balance = self.exchange.fetch_balance()
        return float(balance.get('USDT', {}).get('free', 0))

    def get_asset_balance(self) -> float:
        base = self.symbol.split('/')[0]
        if not self.use_real_money:
            return self.position["size"]
        balance = self.exchange.fetch_balance()
        return float(balance.get(base, {}).get('free', 0))

    def load_state(self):
        if os.path.exists(self.state_file):
            with open(self.state_file, "r", encoding='utf-8') as f:
                state = json.load(f)
                self.last_kline_timestamp = state.get("last_kline")
                self.last_close = state.get("last_close")
                self.position = state.get("position", {"size": 0.0, "entry_price": 0.0})
                if self.last_kline_timestamp:
                    self.last_kline_timestamp = pd.to_datetime(self.last_kline_timestamp, utc=True)
            logger.info(f"📂 恢复状态: 持仓 {self.position['size']:.6f} | 上次K线 {self.last_kline_timestamp}")

    def save_state(self):
        os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
        state = {
            "last_kline": self.last_kline_timestamp.isoformat() if self.last_kline_timestamp else None,
            "last_close": float(self.last_close) if self.last_close is not None else None,
            "position": self.position,
            "updated_at": datetime.now(timezone.utc).isoformat()
        }
        with open(self.state_file, "w", encoding='utf-8') as f:
            json.dump(state, f, indent=2, ensure_ascii=False)

    def place_order(self, side: str, amount: float) -> bool:
        """下单（自动处理精度、最小量校验）"""
        if amount <= 0:
            logger.warning("⚠️ 下单数量 ≤ 0，跳过")
            return False

        # 精度对齐
        amount = self.exchange.amount_to_precision(self.symbol, amount)

        # 最小量检查
        if float(amount) < self.min_amount:
            logger.warning(f"⚠️ 数量 {amount} < 最小量 {self.min_amount}，跳过")
            return False

        current_price = self.exchange.fetch_ticker(self.symbol)['last']
        if float(amount) * current_price < self.min_cost:
            logger.warning(f"⚠️ 订单价值 < {self.min_cost} USDT，跳过")
            return False

        if not self.use_real_money:
            logger.info(f"🧪 [模拟] {side.upper()} {amount} @ ~{current_price:.6f}")
            if side == "buy":
                self.position["size"] += float(amount)
                self.position["entry_price"] = current_price
            elif side == "sell":
                self.position["size"] = 0.0
                self.position["entry_price"] = 0.0
            return True

        try:
            order = self.exchange.create_market_order(self.symbol, side.upper(), amount)
            avg_price = order.get('average') or order.get('price') or current_price
            logger.info(f"✅ 实盘 {side.upper()} {amount} @ {avg_price:.6f} | ID: {order['id']}")

            if side == "buy":
                self.position["size"] += float(amount)
                self.position["entry_price"] = avg_price
            elif side == "sell":
                self.position["size"] = 0.0
                self.position["entry_price"] = 0.0
            return True
        except Exception as e:
            logger.error(f"❌ 下单失败: {e}")
            return False

    def check_stop_loss(self):
        """动态止损检查"""
        if self.position["size"] <= 0 or self.position["entry_price"] <= 0:
            return

        current_price = self.exchange.fetch_ticker(self.symbol)['last']
        entry = self.position["entry_price"]
        loss_pct = (entry - current_price) / entry

        if loss_pct >= STOP_LOSS_PCT:
            logger.warning(f"⚠️ 触发止损！亏损 {loss_pct:.2%} ≥ {STOP_LOSS_PCT:.2%}")
            self.place_order("sell", self.position["size"])

    def run_once(self):
        """执行一次完整信号判断与交易循环"""
        try:
            current_time = pd.Timestamp.now(tz='UTC')
            logger.info(f"当前UTC时间: {current_time}")

            # 获取最近1000根K线，保证有足够数据计算技术指标
            df = fetch_ohlcv(symbol=self.symbol, timeframe=self.timeframe)
            logger.info(f"获取到K线范围: {df.index[0]} 至 {df.index[-1]}")
            if len(df) < 100:
                logger.warning("⚠️ 数据不足，跳过")
                return

            latest_ts = df.index[-1]
            current_last_close = float(df['close'].iloc[-1])

            if self.last_kline_timestamp:
                # 计算时间差（以秒为单位）
                time_diff = (latest_ts - self.last_kline_timestamp).total_seconds()

                # 根据timeframe判断是否有新K线
                timeframe_seconds = {
                    '1m': 60,
                    '3m': 180,
                    '5m': 300,
                    '15m': 900,
                    '30m': 1800,
                    '1h': 3600,
                    '2h': 7200,
                    '4h': 14400,
                    '6h': 21600,
                    '12h': 43200,
                    '1d': 86400,
                }.get(self.timeframe, 3600)  # 默认1小时

                # 如果时间差小于阈值，说明不是新闭合的K线
                if time_diff < timeframe_seconds * 0.95:  # 添加5%的容差
                    # 可能是在同一根（未闭合）K线上发生价格更新
                    if self.last_close is None or current_last_close != float(self.last_close):
                        if TRADE_ONLY_ON_CANDLE_CLOSE:
                            logger.info(
                                f"同一K线价格更新（未闭合）: 时间={latest_ts} | 旧价={self.last_close} -> 新价={current_last_close}；仅记录价格更新，不进行闭合K线交易")
                            # 更新 last_close 并保存状态
                            self.last_close = current_last_close
                            self.save_state()
                            return
                        else:
                            logger.info(
                                f"同一K线价格更新（未闭合）: 时间={latest_ts} | 旧价={self.last_close} -> 新价={current_last_close}；根据配置允许同K线内交易")
                            # 允许在未闭合K线内进行一次信号评估与可能的交易
                            signal_info = self.strategy.generate_signal(df)
                            signal = signal_info["signal"]
                            confidence = signal_info["confidence"]
                            reason = signal_info["reason"]
                            logger.info(f"🧠(intra) 信号: {signal.upper()} | 置信度: {confidence:.3f} | 原因: {reason}")
                            if signal == "buy" and self.position["size"] == 0:
                                usdt_available = self.get_usdt_balance()
                                max_use = usdt_available * MAX_POSITION_RATIO
                                price = df["close"].iloc[-1]
                                amount = max_use / price
                                if amount > 0:
                                    self.place_order("buy", amount)
                            elif signal == "sell" and self.position["size"] > 0:
                                self.place_order("sell", self.position["size"])
                            # 更新 last_close，但不更新 last_kline_timestamp（仍视为未闭合）
                            self.last_close = current_last_close
                            self.save_state()
                            return
                    else:
                        logger.debug(
                            f"K线未更新: 最新={latest_ts}, 上次={self.last_kline_timestamp}, 时间差={time_diff}秒")
                        return

                logger.info(f"检测到新K线: {latest_ts} (上次: {self.last_kline_timestamp}, 时间差={time_diff}秒)")

            logger.info(f"🕒 新K线闭合: {latest_ts} | 收盘价: {df['close'].iloc[-1]:.6f}")

            # ✅ 使用你重写的策略核心生成信号
            signal_info = self.strategy.generate_signal(df)
            signal = signal_info["signal"]
            confidence = signal_info["confidence"]
            reason = signal_info["reason"]

            logger.info(f"🧠 信号: {signal.upper()} | 置信度: {confidence:.3f} | 原因: {reason}")

            # 执行交易
            if signal == "buy" and self.position["size"] == 0:
                usdt_available = self.get_usdt_balance()
                max_use = usdt_available * MAX_POSITION_RATIO
                price = df["close"].iloc[-1]
                amount = max_use / price
                if amount > 0:
                    self.place_order("buy", amount)

            elif signal == "sell" and self.position["size"] > 0:
                self.place_order("sell", self.position["size"])

            # 更新状态
            # 更新状态（记录闭合K线的时间戳和收盘价）
            self.last_kline_timestamp = latest_ts
            self.last_close = current_last_close
            self.save_state()

        except Exception as e:
            logger.error(f"💥 单次循环异常: {e}", exc_info=True)

    def run(self):
        """主循环"""
        logger.info("🔁 开始轮询...")
        while True:
            try:
                self.check_stop_loss()
                self.run_once()
                time.sleep(5)
            except KeyboardInterrupt:
                logger.info("🛑 用户中断，保存状态...")
                self.save_state()
                break
            except Exception as e:
                logger.error(f"🔥 主循环异常: {e}", exc_info=True)
                time.sleep(10)


# ======================
# 启动入口
# ======================
if __name__ == "__main__":
    if USE_REAL_MONEY and (not API_KEY or not API_SECRET):
        raise EnvironmentError("❌ 实盘模式需在 config/settings.py 中配置 API_KEY 和 API_SECRET")

    runner = LiveRunner()
    runner.run()
