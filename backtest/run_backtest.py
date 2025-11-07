# backtest/run_backtest.py
"""
AI 量化策略回测主程序
"""

import json
import os

import backtrader as bt
import pandas as pd
import traceback
import contextlib
from datetime import datetime, timezone, timedelta

from backtest.metrics import calculate_metrics
from config.settings import TIMEFRAME, INITIAL_CASH, FEATURE_WINDOW, MIN_FEATURE_ROWS, FEATURE_HISTORY_PADDING
from data.features import add_features
from data.fetcher import fetch_ohlcv
from model.ensemble_model import EnsembleModel
from strategies.backtrader_adapters.ai_signal_bt import AISignalStrategy
from model.model_registry import load_latest_model_path, get_model_metadata, update_latest_model, find_best_model_by_auc
from model.train import train_and_evaluate
from config.settings import (
    RETRAIN_ON_DEGRADATION,
    RETRAIN_AUC_DIFF,
    RETRAIN_SINCE_DAYS,
    RETRAIN_LIMIT,
    BASE_PATH,
    MODEL_METRICS_PATH,
    SYMBOL,
    LOG_PATH,
    RETRAIN_MAX_ATTEMPTS,
)
from config.settings import (
    CONFIDENCE_THRESHOLD,
    SELL_THRESHOLD,
    MAX_POSITION_RATIO,
    STOP_LOSS_PCT,
    TAKE_PROFIT_PCT,
    COOLDOWN_BARS,
    PROB_EMA_SPAN,
    TIME_STOP_BARS,
    USE_QUANTILE_THRESH,
    PROB_Q_HIGH,
    PROB_Q_LOW,
    PROB_WINDOW,
)


class AIButterflyStrategy(bt.Strategy):
    params = (
        ("model", None),  # 传入训练好的 EnsembleModel 实例
        ("printlog", False),
    )

    def __init__(self):
        self.data_close = self.datas[0].close
        self.order = None
        # 初始化用到的变量
        self.trade_list = []  # 记录每笔交易
        # 调试打印标志：第一次预测前打印 features 信息，便于排查 add_features 的输出
        self._printed_feature_debug = False
        # 概率EMA与入场信息、冷却
        self._prob_ema = None
        self.entry_price = None
        self.entry_bar = None
        self.cooldown_until = -1

    def next(self):
        if self.order:
            return  # 有未完成订单，跳过

        # 获取当前及最近窗口历史数据（DataFrame 格式）
        # 为了保证 rolling/EMA 等指标能被正确计算，需要额外向前拉取一段历史（FEATURE_HISTORY_PADDING）
        # 构建的历史长度为 FEATURE_WINDOW + FEATURE_HISTORY_PADDING，计算完特征后再取最后 FEATURE_WINDOW 行用于预测
        total_bars = len(self)
        # 如果还没有足够的 bar，直接跳过
        if total_bars == 0:
            return
        window = int(FEATURE_WINDOW)
        padding = int(FEATURE_HISTORY_PADDING)
        total_window = window + padding
        start_idx = max(0, total_bars - total_window)
        idx_range = range(start_idx, total_bars)

        # Backtrader 的 linebuffer 通过相对索引访问：ago = absolute_index - current_index
        current_idx = total_bars - 1
        agos = [i - current_idx for i in idx_range]

        df = pd.DataFrame({
            'timestamp': [bt.num2date(self.datas[0].datetime[ago]) for ago in agos],
            'open': [self.datas[0].open[ago] for ago in agos],
            'high': [self.datas[0].high[ago] for ago in agos],
            'low': [self.datas[0].low[ago] for ago in agos],
            'close': [self.datas[0].close[ago] for ago in agos],
            'volume': [self.datas[0].volume[ago] for ago in agos],
        })
        # 确保 timestamp 为索引并计算特征
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        df = add_features(df)

        # 计算完特征后只取最后 window 行作为模型输入（窗口化输入）
        if len(df) >= window:
            df = df.tail(window)

        # 如果特征工程后数据不足，跳过本柱（避免传入空数据给模型）
        min_rows = int(MIN_FEATURE_ROWS)
        if df is None or df.empty or len(df) < min_rows:
            # 可选：打印日志以方便调试
            if self.p.printlog:
                self.log(
                    f"SKIP: insufficient feature rows ({0 if df is None else len(df)}) for prediction; need >= {min_rows}")
            return

        # 在第一次调用 predict 前打印调试信息（列名、行数、示例）以便排查特征缺失问题
        if not self._printed_feature_debug:
            try:
                print("[DEBUG] feature input columns:", list(df.columns))
                print(f"[DEBUG] feature input len: {len(df)}")
                print("[DEBUG] dtypes:\n", df.dtypes)
                print("[DEBUG] tail():\n", df.tail(3))
            except Exception as e:
                print("[DEBUG] failed to print feature debug info:", e)
            self._printed_feature_debug = True

        # 预测上涨概率
        prob = self.params.model.predict(df)
        # 打印每根 K 线的预测概率和当前持仓状态，便于调试
        if self.p.printlog:
            self.log(f"预测概率: {prob:.4f} | 持仓: {self.position.size if self.position else 0}")
        else:
            print(f"[DEBUG] 预测概率: {prob:.4f} | 持仓: {self.position.size if self.position else 0}")

        # 交易逻辑
        # 1. 计算多层级技术指标
        # 获取更多历史数据用于技术分析
        close_series = pd.Series([self.data_close[ago] for ago in range(-20, 0)])
        volume_series = pd.Series([self.datas[0].volume[ago] for ago in range(-20, 0)])

        # 计算多个时间周期的均线
        ma3 = close_series.rolling(window=3).mean().iloc[-1]
        ma5 = close_series.rolling(window=5).mean().iloc[-1]
        ma10 = close_series.rolling(window=10).mean().iloc[-1]

        # 计算动量指标
        roc = (close_series.iloc[-1] - close_series.iloc[-5]) / close_series.iloc[-5]  # 5周期变化率
        volume_ratio = volume_series.iloc[-1] / volume_series.iloc[-5:].mean()  # 当前成交量/5周期平均

        # 综合技术面评分 (0-100)
        tech_score = 0
        # 均线多头排列
        if ma3 > ma5 > ma10:
            tech_score += 40
        elif ma3 > ma5:
            tech_score += 20
        # 强势上涨
        if roc > 0.02:  # 2%以上涨幅
            tech_score += 30
        elif roc > 0:
            tech_score += 15
        # 放量
        if volume_ratio > 1.5:
            tech_score += 30
        elif volume_ratio > 1:
            tech_score += 15

        # 2. 概率EMA与阈值（使用配置）
        alpha = 2.0 / (float(PROB_EMA_SPAN) + 1.0)
        self._prob_ema = prob if self._prob_ema is None else (alpha * prob + (1 - alpha) * self._prob_ema)
        p_eval = self._prob_ema
        buy_threshold = float(CONFIDENCE_THRESHOLD)
        sell_threshold = float(SELL_THRESHOLD)

        # 保存概率历史用于参考
        if not hasattr(self, '_prob_history'):
            self._prob_history = []
        self._prob_history.append(prob)
        # 使用配置的窗口大小计算分位数
        window_len = int(PROB_WINDOW) if int(PROB_WINDOW) > 10 else 10
        window_hist = self._prob_history[-window_len:] if len(self._prob_history) >= window_len else self._prob_history

        # 分位数自适应阈值（可选）
        if USE_QUANTILE_THRESH and len(window_hist) >= max(30, int(window_len * 0.5)):
            import numpy as np
            qh = float(np.quantile(window_hist, float(PROB_Q_HIGH)))
            ql = float(np.quantile(window_hist, float(PROB_Q_LOW)))
            buy_threshold = qh
            sell_threshold = ql

        # 输出调试信息
        trend_up = (ma3 > ma5 > ma10) or (roc > 0 and volume_ratio >= 1)
        if self.p.printlog:
            self.log(
                f"技术面: {'多头' if trend_up else '空头'} | 买入阈值={buy_threshold:.3f} 卖出阈值={sell_threshold:.3f}")
        else:
            print(
                f"[DEBUG] 技术面: {'多头' if trend_up else '空头'} | 买入阈值={buy_threshold:.3f} 卖出阈值={sell_threshold:.3f}")
        if self.p.printlog:
            self.log(f"阈值(EMA): 买入={buy_threshold:.3f} 卖出={sell_threshold:.3f} | p_ema={p_eval:.3f}")
        else:
            print(f"[DEBUG] 阈值(EMA): 买入={buy_threshold:.3f} 卖出={sell_threshold:.3f} | p_ema={p_eval:.3f}")

        # 使用显式持仓数量判断，避免 Backtrader 中 position 对象在空仓时也被视为真
        current_bar = len(self)
        # 平仓条件：止损/止盈/时间止损 或 概率EMA触及卖出阈值
        if self.position.size > 0:
            price_now = float(self.data_close[0])
            hit_sl = False
            hit_tp = False
            hit_time = False
            if self.entry_price is not None:
                ret = (price_now - self.entry_price) / self.entry_price
                hit_sl = ret <= -float(STOP_LOSS_PCT)
                hit_tp = ret >= float(TAKE_PROFIT_PCT)
            if self.entry_bar is not None and TIME_STOP_BARS and int(TIME_STOP_BARS) > 0:
                hit_time = (current_bar - int(self.entry_bar)) >= int(TIME_STOP_BARS)

            should_sell = (p_eval <= sell_threshold) or hit_sl or hit_tp or hit_time
            if should_sell:
                self.order = self.sell(size=self.position.size)
                if self.p.printlog:
                    self.log(
                        f"SELL CREATE, price={self.data_close[0]:.6f}, size={self.position.size:.6f}, p_ema={p_eval:.3f}, sl={hit_sl}, tp={hit_tp}, tstop={hit_time}")
                else:
                    print(
                        f"[DEBUG] SELL CREATE at {self.data_close[0]:.6f}, size={self.position.size:.6f}, p_ema={p_eval:.3f}, sl={hit_sl}, tp={hit_tp}, tstop={hit_time}")
                # 冷却
                self.cooldown_until = current_bar + int(COOLDOWN_BARS)
                self.entry_price = None
                self.entry_bar = None
            return

        # 空仓：冷却外且满足买入阈值
        if self.position.size == 0 and current_bar >= int(self.cooldown_until):
            if p_eval >= buy_threshold:
                # 计算考虑手续费与安全缓冲后的最大可买数量，并向下取整为整数
                try:
                    commission_rate = float(self.broker.getcommissioninfo(self.data).p.commission)
                except Exception:
                    commission_rate = 0.001
                price = float(self.data_close[0])
                cash = float(self.broker.getcash())
                safety = 0.99
                unit_cost = price * (1.0 + commission_rate)
                budget = cash * float(MAX_POSITION_RATIO)
                size = int((budget * safety) / unit_cost)
                if size <= 0:
                    print(
                        f"[DEBUG] SKIP BUY: computed size<=0 | cash={cash:.2f} price={price:.6f} commission={commission_rate}")
                else:
                    self.order = self.buy(size=size)
                    if self.p.printlog:
                        self.log(
                            f"BUY CREATE, price={price:.6f}, size={size}, p_ema={p_eval:.3f}, cash={cash:.2f}, comm={commission_rate}, budget_ratio={MAX_POSITION_RATIO}")
                    else:
                        print(
                            f"[DEBUG] BUY CREATE at {price:.6f}, size={size}, p_ema={p_eval:.3f}, cash={cash:.2f}, comm={commission_rate}, budget_ratio={MAX_POSITION_RATIO}")
                    # 记录入场信息与冷却
                    self.entry_price = price
                    self.entry_bar = current_bar
                    self.cooldown_until = current_bar + int(COOLDOWN_BARS)

    def notify_order(self, order):
        # 打印所有订单状态，便于诊断为何未成交/被拒
        status_map = {
            order.Submitted: "Submitted",
            order.Accepted: "Accepted",
            order.Partial: "Partial",
            order.Completed: "Completed",
            order.Canceled: "Canceled",
            order.Rejected: "Rejected",
            order.Margin: "Margin",
            order.Expired: "Expired",
        }
        status_str = status_map.get(order.status, str(order.status))
        try:
            created_size = getattr(order.created, 'size', None)
        except Exception:
            created_size = None
        print(
            f"[DEBUG] ORDER STATUS: {status_str} | isbuy={order.isbuy()} | size={created_size if created_size is not None else getattr(order, 'size', 'NA')}")

        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f"BUY EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}")
            elif order.issell():
                self.log(f"SELL EXECUTED, Price: {order.executed.price:.2f}, Value: {order.executed.value:.2f}")
            self.bar_executed = len(self)
        if order.status in [order.Canceled, order.Rejected, order.Margin, order.Expired, order.Completed]:
            self.order = None

    def notify_trade(self, trade):
        if trade.isclosed:
            self.trade_list.append({
                "pnl": trade.pnlcomm,
                "pnl_pct": trade.pnlcomm / (trade.value - trade.pnlcomm),
                "size": trade.size,
                "value": trade.value,
                "entry": trade.price,
                "exit": trade.price + (trade.pnl / trade.size if trade.size != 0 else 0),
                "duration": trade.barlen
            })
            self.log(f"OPERATION PROFIT, GROSS {trade.pnl:.2f}, NET {trade.pnlcomm:.2f}")

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print(f"{dt.isoformat()} {txt}")


def comback_train_and_evaluate(model_path, metrics, df):
    # 7. 判断是否需要重训练并在必要时触发（保守策略）
    try:
        # 当前在线模型元数据
        current_version = os.path.basename(model_path).replace('.pkl', '')
        current_meta = get_model_metadata(current_version)
        current_auc = float(current_meta.get('auc', 0.5))
    except Exception:
        current_auc = 0.5

    backtest_auc = float(metrics.get('auc', 0.5))
    total_return = float(metrics.get('total_return_pct', 0.0))

    # 触发重训练的条件：回测收益为负 或 回测 AUC 明显低于训练时 AUC（阈值 0.01）
    retrain_needed = False
    if total_return < 0:
        retrain_needed = True
    if backtest_auc < (current_auc - 0.01):
        retrain_needed = True

    # 使用配置的 AUC 差值阈值判断
    if backtest_auc < (current_auc - RETRAIN_AUC_DIFF):
        retrain_needed = True

    if retrain_needed and RETRAIN_ON_DEGRADATION:
        print(
            f"🔁 检测到模型性能下降或回测为负，开始重训练循环（最多 {RETRAIN_MAX_ATTEMPTS} 次），将等待训练并验证每次结果...")
        # 准备日志文件
        os.makedirs(LOG_PATH, exist_ok=True)
        ts = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(LOG_PATH, f"retrain_{ts}.log")

        attempt = 0
        accepted = False
        while attempt < RETRAIN_MAX_ATTEMPTS and not accepted:
            attempt += 1
            print(f"🔁 重训练尝试 {attempt}/{RETRAIN_MAX_ATTEMPTS}，日志: {log_file}")
            new_version = None
            new_auc = None
            try:
                with open(log_file, 'a', encoding='utf-8') as f:
                    f.write(f"\n=== Retrain attempt {attempt} started: {datetime.utcnow().isoformat()} UTC ===\n")
                # 在日志中记录并重定向输出
                with open(log_file, 'a', encoding='utf-8') as f:
                    f.write(
                        f"Command: train_and_evaluate(symbol={SYMBOL}, timeframe={TIMEFRAME}, limit={RETRAIN_LIMIT}, since_days={RETRAIN_SINCE_DAYS})\n")
                    f.flush()
                    try:
                        with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
                            new_version, new_auc = train_and_evaluate(symbol=None, timeframe=TIMEFRAME,
                                                                      limit=RETRAIN_LIMIT,
                                                                      since_days=RETRAIN_SINCE_DAYS)
                    except Exception:
                        f.write("\n=== Exception during retrain attempt ===\n")
                        traceback.print_exc(file=f)
                        raise
                    finally:
                        with open(log_file, 'a', encoding='utf-8') as f2:
                            f2.write(
                                f"=== Retrain attempt {attempt} finished: {datetime.utcnow().isoformat()} UTC ===\n")
            except Exception as e:
                print(f"❌ 自动重训练失败（attempt {attempt}）: {e}")

            # 如果产生了新模型，加载并使用它继续回测验证
            if new_version is not None:
                try:
                    print(f"🔧 已训练出模型 {new_version}（AUC={new_auc}），开始用新模型回测验证...")
                    # 加载最新模型路径（训练脚本通常会更新 registry）
                    new_model_path = load_latest_model_path()
                    if not new_model_path:
                        print("⚠️ 无法找到训练出的模型文件，跳过本次验证")
                    else:
                        # 重新运行回测（使用相同数据 df）
                        cerebro = bt.Cerebro()
                        ensemble_model = EnsembleModel(new_model_path, TIMEFRAME)
                        cerebro.addstrategy(AIButterflyStrategy, model=ensemble_model, printlog=False)
                        data = bt.feeds.PandasData(dataname=df, datetime=None, open=0, high=1, low=2, close=3, volume=4,
                                                   openinterest=-1)
                        cerebro.datas = []
                        cerebro.adddata(data)
                        cerebro.broker.setcash(INITIAL_CASH)
                        cerebro.broker.setcommission(commission=0.001)
                        start_value2 = cerebro.broker.getvalue()
                        results2 = cerebro.run()
                        end_value2 = cerebro.broker.getvalue()
                        strategy2 = results2[0]
                        trades2 = strategy2.trade_list
                        # 兼容无 target 列
                        if len(trades2) > 0 and isinstance(df, pd.DataFrame) and ("target" in df.columns):
                            y_true_for_auc2 = df["target"].iloc[-len(trades2):]
                        else:
                            y_true_for_auc2 = None
                        metrics2 = calculate_metrics(trades2, y_true_for_auc2)
                        metrics2.update({
                            "initial_cash": INITIAL_CASH,
                            "final_value": round(end_value2, 2),
                            "total_return_pct": round((end_value2 - start_value2) / start_value2 * 100, 2),
                            "total_trades": len(trades2)
                        })
                        # 把验证结果追加到日志
                        with open(log_file, 'a', encoding='utf-8') as f:
                            f.write(f"Validation metrics for {new_version}: {metrics2}\n")
                        print(
                            f"🔍 验证结果: return={metrics2['total_return_pct']}% | trades={metrics2['total_trades']} | auc={metrics2.get('auc', 'N/A')}")

                        # 接受条件：验证收益非负且 AUC 不低于训练 AUC - 差值阈值
                        try:
                            val_auc = float(metrics2.get('auc', 0.5))
                        except Exception:
                            val_auc = 0.5
                        try:
                            new_auc_f = float(new_auc)
                        except Exception:
                            new_auc_f = 0.5

                        if metrics2['total_return_pct'] >= 0 and val_auc >= (new_auc_f - RETRAIN_AUC_DIFF):
                            accepted = True
                            update_latest_model(new_version)
                            print(f"✅ 新模型 {new_version} 验证通过并已设为最优模型")
                            with open(log_file, 'a', encoding='utf-8') as f:
                                f.write(f"ACCEPTED: {new_version} | val_metrics={metrics2}\n")
                        else:
                            print(f"❌ 新模型 {new_version} 未通过验证（继续重训练）。")
                            with open(log_file, 'a', encoding='utf-8') as f:
                                f.write(f"REJECTED: {new_version} | val_metrics={metrics2}\n")
                except Exception as e:
                    print(f"⚠️ 验证新模型时出错: {e}")
            else:
                print("⚠️ 本次重训练未产出新模型，继续尝试...")

        if not accepted:
            print(f"⚠️ 达到最大重训练次数 ({RETRAIN_MAX_ATTEMPTS})，仍未找到合格模型")
        print(f"📄 重训练日志: {log_file}")


def run_backtest():
    print("🔄 开始回测...")

    # 1. 获取数据（按配置的天数计算 since，并分页抓取 >1000 根）
    since = None
    try:
        dt_since = datetime.now(timezone.utc) - timedelta(days=RETRAIN_SINCE_DAYS)
        since = int(dt_since.timestamp() * 1000)
        print(f"⏳ 回测拉取自 {dt_since.strftime('%Y-%m-%d')} 以来的K线数据")
    except Exception:
        pass
    df = fetch_ohlcv(limit=RETRAIN_LIMIT, since=since)
    if len(df) < 300:
        raise ValueError("回测数据不足，请确保至少有 300 根K线")

    # 2. 加载最新模型（供 AISignalCore 使用，仍做可用性检查）
    model_path = load_latest_model_path()
    if not model_path:
        raise RuntimeError("未找到训练好的模型，请先运行 model/train.py")
    print(f"✅ 已加载模型: {os.path.basename(model_path)}")

    # 3. 初始化 Cerebro 引擎
    cerebro = bt.Cerebro()
    cerebro.addstrategy(AISignalStrategy, save_trades=True, confidence_threshold=CONFIDENCE_THRESHOLD,
                        cooldown_bars=COOLDOWN_BARS, trend_filter=True)

    # 转换为 Backtrader 数据格式
    data = bt.feeds.PandasData(
        dataname=df,
        datetime=None,
        open=0,
        high=1,
        low=2,
        close=3,
        volume=4,
        openinterest=-1
    )
    cerebro.adddata(data)
    cerebro.broker.setcash(INITIAL_CASH)
    cerebro.broker.setcommission(commission=0.001)  # 0.1% 手续费
    # 允许订单在同一根K线的收盘被撮合，便于快速验证成交
    cerebro.broker.set_coc(True)

    # 4. 运行回测
    start_value = cerebro.broker.getvalue()
    results = cerebro.run()
    end_value = cerebro.broker.getvalue()

    # 5. 计算指标
    strategy = results[0]
    trades = getattr(strategy, 'trade_list_bt', []) or strategy.trade_list
    # 兼容无 target 列的情形，AUC 将回退为 0.5
    if len(trades) > 0 and isinstance(df, pd.DataFrame) and ("target" in df.columns):
        y_true_for_auc = df["target"].iloc[-len(trades):]
    else:
        y_true_for_auc = None
    metrics = calculate_metrics(trades, y_true_for_auc)

    # 补充资金曲线指标
    metrics.update({
        "initial_cash": INITIAL_CASH,
        "final_value": round(end_value, 2),
        "total_return_pct": round((end_value - start_value) / start_value * 100, 2),
        "total_trades": len(trades)
    })

    # 6. 保存指标
    metrics_path = MODEL_METRICS_PATH
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)

    # 7. 判断是否需要重训练并在必要时触发（保守策略）
    # comback_train_and_evaluate(model_path, metrics, df)

    # 8. 打印摘要
    print("\n" + "=" * 50)
    print(f"💼 初始资金: {INITIAL_CASH:,.2f} USDT")
    print(f"💰 最终资金: {end_value:,.2f} USDT")
    print(f"📈 收益率: {metrics['total_return_pct']:.2f}%")
    print(f"📊 回测完成 | AUC: {metrics.get('auc', 'N/A')}, 胜率: {metrics.get('win_rate', 0) * 100:.1f}%")
    print("=" * 50)

    return metrics
