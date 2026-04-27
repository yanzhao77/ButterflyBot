"""
ButterflyBot 5年历史数据回测脚本
使用批量预测（高效），模拟真实AISignalCore的逐步EMA预测行为
数据：DOGE/USDT 15m，2020-04-28 ~ 2025-12-23（约198,000根K线）
"""
import sys, os, json, logging
sys.path.insert(0, ".")
import numpy as np
import pandas as pd
from datetime import datetime

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

from butterfly_bot.data.fetcher import fetch_ohlcv
from butterfly_bot.data.features import add_features, get_feature_columns
from butterfly_bot.model.model_registry import load_latest_model_path
import joblib

# ===================== 配置 =====================
INITIAL_BALANCE   = 1000.0   # 初始资金 USDT
LEVERAGE          = 1        # 杠杆
POSITION_PCT      = 0.95     # 每次开仓使用资金比例
CONFIDENCE_TH     = 0.70     # AI信号置信度阈值
STOP_LOSS_PCT     = -0.03    # 固定止损 -3%
TAKE_PROFIT_PCT   = 0.06     # 止盈 +6%
MAX_HOLDING_BARS  = 50       # 最大持仓K线数
COOLDOWN_BARS     = 5        # 冷却期
PROB_EMA_SPAN     = 10       # 概率EMA平滑窗口
TREND_FILTER      = True     # 是否启用趋势过滤（MA50）
FEE_RATE          = 0.001    # 手续费 0.1%

# 移动止损参数
TRAIL_ACTIVATE_PCT = 0.03    # 浮盈达到3%激活移动止损
TRAIL_STOP_PCT     = 0.015   # 移动止损回撤1.5%

os.makedirs("reports/backtest", exist_ok=True)

# ===================== 数据加载 =====================
print("=" * 70)
print("🚀 ButterflyBot 5年历史数据回测")
print("=" * 70)
print("📥 正在加载5年历史数据...")
data = fetch_ohlcv("DOGE/USDT", "15m", limit=None)
print(f"✅ 数据加载成功: {len(data)} 根K线")
print(f"   时间范围: {data.index[0]} ~ {data.index[-1]}")

# ===================== 批量特征工程 + 预测 =====================
print("\n🤖 正在进行批量特征工程和模型预测...")
model_path = load_latest_model_path()
model = joblib.load(model_path)
model_name = os.path.basename(model_path)
print(f"✅ 模型加载成功: {model_name}")

features_df = add_features(data)
feature_cols = get_feature_columns()
available_cols = [c for c in feature_cols if c in features_df.columns]
X = features_df[available_cols].values
raw_probs = model.predict(X)
print(f"✅ 特征工程完成: {len(features_df)} 行特征")
print(f"✅ 批量预测完成: {len(raw_probs)} 个预测值")

# 对齐预测结果到原始数据索引
aligned_probs = pd.Series(np.nan, index=data.index)
aligned_probs.loc[features_df.index] = raw_probs

# 逐步EMA平滑（与AISignalCore一致）
alpha = 2.0 / (PROB_EMA_SPAN + 1.0)
p_ema_val = None
pema_list = []
for prob in aligned_probs.values:
    if not np.isnan(prob):
        p_ema_val = prob if p_ema_val is None else alpha * prob + (1 - alpha) * p_ema_val
    pema_list.append(p_ema_val if p_ema_val is not None else 0.3)

predictions = np.array(pema_list)
print(f"   p_ema 统计: min={predictions.min():.4f}, max={predictions.max():.4f}, mean={predictions.mean():.4f}")
print(f"   85%分位数={np.quantile(predictions, 0.85):.4f}, 90%分位数={np.quantile(predictions, 0.90):.4f}")

# ===================== 回测引擎 =====================
def run_backtest(predictions, data, label, confidence_threshold,
                 use_trailing_stop=False, stop_loss_pct=STOP_LOSS_PCT,
                 take_profit_pct=TAKE_PROFIT_PCT):
    """单次回测引擎"""
    balance = INITIAL_BALANCE
    position = 0.0        # 持仓数量
    entry_price = 0.0
    entry_bar = 0
    cooldown = 0
    peak_price = 0.0      # 持仓期间最高价（用于移动止损）
    trail_activated = False

    trades = []
    equity_curve = [balance]
    equity_dates = [data.index[0]]

    closes = data["close"].values
    ma50 = data["close"].rolling(50).mean().values

    for i in range(len(data)):
        price = closes[i]
        p_ema = predictions[i]

        # 更新权益曲线（每根K线）
        if position > 0:
            unrealized_pnl = (price - entry_price) * position
            equity_curve.append(balance + unrealized_pnl)
        else:
            equity_curve.append(balance)
        equity_dates.append(data.index[i])

        # 冷却期递减
        if cooldown > 0:
            cooldown -= 1

        # ---- 持仓管理 ----
        if position > 0:
            holding_bars = i - entry_bar
            profit_pct = (price - entry_price) / entry_price

            # 更新最高价
            if price > peak_price:
                peak_price = price

            # 移动止损逻辑
            if use_trailing_stop:
                if not trail_activated and profit_pct >= TRAIL_ACTIVATE_PCT:
                    trail_activated = True
                if trail_activated:
                    drawdown_from_peak = (price - peak_price) / peak_price
                    if drawdown_from_peak <= -TRAIL_STOP_PCT:
                        # 移动止损触发
                        pnl = (price - entry_price) * position - abs(entry_price * position * FEE_RATE) - abs(price * position * FEE_RATE)
                        balance += pnl
                        reason = "移动止损-锁利" if profit_pct > 0 else "移动止损-保本"
                        trades.append({"entry": entry_price, "exit": price, "pnl": pnl,
                                       "bars": holding_bars, "reason": reason,
                                       "entry_time": str(data.index[entry_bar]),
                                       "exit_time": str(data.index[i])})
                        position = 0; entry_price = 0; peak_price = 0; trail_activated = False
                        cooldown = COOLDOWN_BARS
                        continue

            # 固定止损
            if profit_pct <= stop_loss_pct:
                pnl = (price - entry_price) * position - abs(entry_price * position * FEE_RATE) - abs(price * position * FEE_RATE)
                balance += pnl
                trades.append({"entry": entry_price, "exit": price, "pnl": pnl,
                               "bars": holding_bars, "reason": "固定止损",
                               "entry_time": str(data.index[entry_bar]),
                               "exit_time": str(data.index[i])})
                position = 0; entry_price = 0; peak_price = 0; trail_activated = False
                cooldown = COOLDOWN_BARS
                continue

            # 止盈
            if profit_pct >= take_profit_pct:
                pnl = (price - entry_price) * position - abs(entry_price * position * FEE_RATE) - abs(price * position * FEE_RATE)
                balance += pnl
                trades.append({"entry": entry_price, "exit": price, "pnl": pnl,
                               "bars": holding_bars, "reason": "止盈",
                               "entry_time": str(data.index[entry_bar]),
                               "exit_time": str(data.index[i])})
                position = 0; entry_price = 0; peak_price = 0; trail_activated = False
                cooldown = COOLDOWN_BARS
                continue

            # AI看跌信号卖出
            sell_th = max(0.3, confidence_threshold - 0.15)
            if p_ema <= sell_th:
                pnl = (price - entry_price) * position - abs(entry_price * position * FEE_RATE) - abs(price * position * FEE_RATE)
                balance += pnl
                trades.append({"entry": entry_price, "exit": price, "pnl": pnl,
                               "bars": holding_bars, "reason": "AI看跌",
                               "entry_time": str(data.index[entry_bar]),
                               "exit_time": str(data.index[i])})
                position = 0; entry_price = 0; peak_price = 0; trail_activated = False
                cooldown = COOLDOWN_BARS
                continue

            # 时间止损
            if holding_bars >= MAX_HOLDING_BARS:
                pnl = (price - entry_price) * position - abs(entry_price * position * FEE_RATE) - abs(price * position * FEE_RATE)
                balance += pnl
                trades.append({"entry": entry_price, "exit": price, "pnl": pnl,
                               "bars": holding_bars, "reason": "时间止损",
                               "entry_time": str(data.index[entry_bar]),
                               "exit_time": str(data.index[i])})
                position = 0; entry_price = 0; peak_price = 0; trail_activated = False
                cooldown = COOLDOWN_BARS
                continue

        # ---- 开仓逻辑 ----
        if position == 0 and cooldown == 0 and p_ema >= confidence_threshold:
            # 趋势过滤：价格在MA50之上
            if TREND_FILTER and not np.isnan(ma50[i]) and price < ma50[i]:
                continue

            invest = balance * POSITION_PCT
            fee = invest * FEE_RATE
            position = (invest - fee) / price
            entry_price = price
            entry_bar = i
            peak_price = price
            trail_activated = False

    # 强制平仓（回测结束时）
    if position > 0:
        price = closes[-1]
        holding_bars = len(data) - 1 - entry_bar
        pnl = (price - entry_price) * position - abs(entry_price * position * FEE_RATE) - abs(price * position * FEE_RATE)
        balance += pnl
        trades.append({"entry": entry_price, "exit": price, "pnl": pnl,
                       "bars": holding_bars, "reason": "回测结束平仓",
                       "entry_time": str(data.index[entry_bar]),
                       "exit_time": str(data.index[-1])})

    # 统计
    total_trades = len(trades)
    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    win_rate = len(wins) / total_trades * 100 if total_trades > 0 else 0
    total_profit = sum(t["pnl"] for t in wins)
    total_loss = abs(sum(t["pnl"] for t in losses))
    profit_factor = total_profit / total_loss if total_loss > 0 else float("inf")
    net_return = (balance - INITIAL_BALANCE) / INITIAL_BALANCE * 100

    # 最大回撤
    eq = np.array(equity_curve)
    peak_eq = np.maximum.accumulate(eq)
    drawdown = (eq - peak_eq) / peak_eq * 100
    max_drawdown = drawdown.min()

    # 年化收益率（按实际天数）
    total_days = (data.index[-1] - data.index[0]).days
    annual_return = ((balance / INITIAL_BALANCE) ** (365 / total_days) - 1) * 100 if total_days > 0 else 0

    # 卖出原因统计
    reason_counts = {}
    for t in trades:
        r = t["reason"]
        reason_counts[r] = reason_counts.get(r, 0) + 1

    result = {
        "label": label,
        "initial_balance": INITIAL_BALANCE,
        "final_balance": round(balance, 2),
        "net_return_pct": round(net_return, 2),
        "annual_return_pct": round(annual_return, 2),
        "total_trades": total_trades,
        "win_rate": round(win_rate, 2),
        "profit_factor": round(profit_factor, 3) if profit_factor != float("inf") else "inf",
        "total_profit": round(total_profit, 2),
        "total_loss": round(total_loss, 2),
        "avg_profit": round(total_profit / len(wins), 2) if wins else 0,
        "avg_loss": round(-total_loss / len(losses), 2) if losses else 0,
        "max_drawdown_pct": round(max_drawdown, 2),
        "reason_counts": reason_counts,
        "equity_curve": [round(e, 2) for e in equity_curve[::96]],  # 每天4根（每96根15min=1天）采样
        "equity_dates": [str(d) for d in equity_dates[::96]],
        "trades": trades,
    }
    return result


# ===================== 运行三组对比回测 =====================
print("\n" + "=" * 70)
print("📊 运行三组对比回测...")
print("=" * 70)

print("📊 运行基准回测 (固定阈值0.70 + 固定止损)...")
r_base = run_backtest(predictions, data, "基准(固定阈值+固定止损)",
                      confidence_threshold=CONFIDENCE_TH, use_trailing_stop=False)

print("📊 运行优化1 (固定阈值0.70 + 移动止损)...")
r_trail = run_backtest(predictions, data, "优化1(固定阈值+移动止损)",
                       confidence_threshold=CONFIDENCE_TH, use_trailing_stop=True)

dyn_th = float(np.quantile(predictions, 0.85))
print(f"📊 运行优化2 (动态阈值{dyn_th:.4f} + 移动止损)...")
r_dyn = run_backtest(predictions, data, f"优化2(动态阈值{dyn_th:.2f}+移动止损)",
                     confidence_threshold=dyn_th, use_trailing_stop=True)

results = [r_base, r_trail, r_dyn]

# ===================== 打印结果 =====================
def print_result(r):
    print(f"\n{'─' * 55}")
    print(f"  📌 {r['label']}")
    print(f"{'─' * 55}")
    print(f"  初始资金:     {r['initial_balance']:.2f} USDT")
    print(f"  最终资金:     {r['final_balance']:.2f} USDT")
    print(f"  净收益率:     {r['net_return_pct']:+.2f}%")
    print(f"  年化收益率:   {r['annual_return_pct']:+.2f}%")
    print(f"  最大回撤:     {r['max_drawdown_pct']:.2f}%")
    print(f"  总交易次数:   {r['total_trades']}")
    print(f"  胜率:         {r['win_rate']:.2f}%")
    print(f"  盈利因子:     {r['profit_factor']}")
    print(f"  总盈利:       +{r['total_profit']:.2f} USDT")
    print(f"  总亏损:       -{r['total_loss']:.2f} USDT")
    print(f"  平均盈利:     +{r['avg_profit']:.2f} USDT/笔")
    print(f"  平均亏损:     {r['avg_loss']:.2f} USDT/笔")
    print(f"  卖出原因:")
    for reason, cnt in sorted(r['reason_counts'].items(), key=lambda x: -x[1]):
        pct = cnt / r['total_trades'] * 100 if r['total_trades'] > 0 else 0
        print(f"    {reason}: {cnt}次 ({pct:.1f}%)")

for r in results:
    print_result(r)

# ===================== 汇总对比表 =====================
print("\n" + "=" * 70)
print("📈 5年回测汇总对比")
print("=" * 70)
headers = ["指标", "基准", "优化1(移动止损)", "优化2(动态阈值)"]
rows = [
    ["净收益率",     f"{r_base['net_return_pct']:+.2f}%",  f"{r_trail['net_return_pct']:+.2f}%",  f"{r_dyn['net_return_pct']:+.2f}%"],
    ["年化收益率",   f"{r_base['annual_return_pct']:+.2f}%",f"{r_trail['annual_return_pct']:+.2f}%",f"{r_dyn['annual_return_pct']:+.2f}%"],
    ["最大回撤",     f"{r_base['max_drawdown_pct']:.2f}%", f"{r_trail['max_drawdown_pct']:.2f}%", f"{r_dyn['max_drawdown_pct']:.2f}%"],
    ["最终资金",     f"{r_base['final_balance']:.2f}",     f"{r_trail['final_balance']:.2f}",     f"{r_dyn['final_balance']:.2f}"],
    ["总交易次数",   str(r_base['total_trades']),           str(r_trail['total_trades']),           str(r_dyn['total_trades'])],
    ["胜率",         f"{r_base['win_rate']:.2f}%",          f"{r_trail['win_rate']:.2f}%",          f"{r_dyn['win_rate']:.2f}%"],
    ["盈利因子",     str(r_base['profit_factor']),          str(r_trail['profit_factor']),          str(r_dyn['profit_factor'])],
    ["平均盈利",     f"+{r_base['avg_profit']:.2f}",        f"+{r_trail['avg_profit']:.2f}",        f"+{r_dyn['avg_profit']:.2f}"],
    ["平均亏损",     f"{r_base['avg_loss']:.2f}",           f"{r_trail['avg_loss']:.2f}",           f"{r_dyn['avg_loss']:.2f}"],
]
col_w = 20
print(f"{'指标':<16}" + "".join(f"{h:<{col_w}}" for h in headers[1:]))
print("─" * (16 + col_w * 3))
for row in rows:
    print(f"{row[0]:<16}" + "".join(f"{v:<{col_w}}" for v in row[1:]))

# ===================== 保存结果 =====================
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
report_path = f"reports/backtest/backtest_5year_{timestamp}.json"
report = {
    "timestamp": timestamp,
    "symbol": "DOGE/USDT",
    "timeframe": "15m",
    "data_range": f"{data.index[0]} ~ {data.index[-1]}",
    "total_bars": len(data),
    "model": model_name,
    "config": {
        "initial_balance": INITIAL_BALANCE,
        "confidence_threshold": CONFIDENCE_TH,
        "stop_loss_pct": STOP_LOSS_PCT,
        "take_profit_pct": TAKE_PROFIT_PCT,
        "max_holding_bars": MAX_HOLDING_BARS,
        "cooldown_bars": COOLDOWN_BARS,
        "fee_rate": FEE_RATE,
        "trend_filter": TREND_FILTER,
        "trail_activate_pct": TRAIL_ACTIVATE_PCT,
        "trail_stop_pct": TRAIL_STOP_PCT,
    },
    "results": {
        "base": {k: v for k, v in r_base.items() if k != "trades"},
        "trailing": {k: v for k, v in r_trail.items() if k != "trades"},
        "dynamic": {k: v for k, v in r_dyn.items() if k != "trades"},
    },
    "trades": {
        "base": r_base["trades"],
        "trailing": r_trail["trades"],
        "dynamic": r_dyn["trades"],
    }
}
with open(report_path, "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
print(f"\n✅ 回测报告已保存: {report_path}")
print("=" * 70)
