"""
ButterflyBot 新旧模型5年对比回测脚本
对比：
  A. 旧模型（v20251222，52天训练，阈值0.70）
  B. 新模型（v20260427，5年训练，阈值0.35）
  C. 新模型（v20260427，5年训练，阈值0.40）
  D. 新模型 + 账户硬止损（总回撤-15%暂停）

回测参数：
  - 初始资金: 1000 USDT
  - 仓位: 10%（新风控参数）
  - 固定止损: -2%（优化后）
  - 止盈: +4%（优化后）
  - 手续费: 0.1%
  - 趋势过滤: MA50
"""
import sys, os, json, logging, time
sys.path.insert(0, ".")
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter
from datetime import datetime

logging.basicConfig(level=logging.WARNING)

from butterfly_bot.data.fetcher import fetch_ohlcv
from butterfly_bot.model.model_registry import REGISTRY_DIR

os.makedirs("reports/backtest", exist_ok=True)

# ===================== 回测参数 =====================
INITIAL_BALANCE   = 1000.0
POSITION_PCT      = 0.10    # 降低至10%（新风控）
FEE_RATE          = 0.001
STOP_LOSS_PCT     = -0.02   # 优化：-2%
TAKE_PROFIT_PCT   = 0.04    # 优化：+4%
MAX_HOLDING_BARS  = 50
COOLDOWN_BARS     = 5
PROB_EMA_SPAN     = 10
TREND_FILTER      = True
HARD_STOP_PCT     = 0.15    # 账户级硬止损 -15%

print("=" * 70)
print("🚀 ButterflyBot 新旧模型5年对比回测")
print("=" * 70)

# ===================== 加载数据 =====================
print("\n📥 加载5年历史数据...")
data = fetch_ohlcv("DOGE/USDT", "15m", limit=None)
print(f"✅ 数据加载完成: {len(data)} 根K线")
print(f"   时间范围: {data.index[0]} ~ {data.index[-1]}")

# ===================== 加载特征（新目标变量）=====================
print("\n🔧 构建特征...")
import butterfly_bot.config.settings as settings
settings.TARGET_SHIFT = 2
settings.TARGET_THRESHOLD = 0.003
import importlib
import butterfly_bot.data.features as feat_module
importlib.reload(feat_module)
features_df = feat_module.add_features(data)
feature_cols = feat_module.get_feature_columns()
available_cols = [c for c in feature_cols if c in features_df.columns]
X_all = features_df[available_cols].values
print(f"✅ 特征工程完成: {len(features_df)} 行")

# ===================== 生成预测概率 =====================
def get_predictions(model_path, calibrator_path=None, label=""):
    """加载模型并生成预测概率（带EMA平滑）"""
    print(f"\n🤖 加载模型: {os.path.basename(model_path)}")
    model = joblib.load(model_path)
    
    raw_probs = model.predict(X_all)
    
    # 如果有校准器，应用校准
    if calibrator_path and os.path.exists(calibrator_path):
        calibrator = joblib.load(calibrator_path)
        raw_probs = calibrator.predict_proba(raw_probs.reshape(-1, 1))[:, 1]
        print(f"   ✅ 已应用 Platt 校准")
    
    # 对齐到原始数据索引
    aligned = pd.Series(np.nan, index=data.index)
    aligned.loc[features_df.index] = raw_probs
    
    # 逐步EMA平滑
    alpha = 2.0 / (PROB_EMA_SPAN + 1.0)
    p_ema_val = None
    pema_list = []
    for prob in aligned.values:
        if not np.isnan(prob):
            p_ema_val = prob if p_ema_val is None else alpha * prob + (1 - alpha) * p_ema_val
        pema_list.append(p_ema_val if p_ema_val is not None else 0.25)
    
    predictions = np.array(pema_list)
    print(f"   p_ema 统计: min={predictions.min():.4f}, max={predictions.max():.4f}, mean={predictions.mean():.4f}")
    print(f"   85%分位数={np.quantile(predictions, 0.85):.4f}, 90%分位数={np.quantile(predictions, 0.90):.4f}")
    return predictions

# 旧模型（v20251222）
old_model_path = os.path.join(REGISTRY_DIR, "v20251222_031926.pkl")
# 新模型（v20260427_145956）
new_model_path = os.path.join(REGISTRY_DIR, "v20260427_145956.pkl")
cal_path = os.path.join(REGISTRY_DIR, "platt_calibrator.pkl")

preds_old = get_predictions(old_model_path, label="旧模型")
preds_new = get_predictions(new_model_path, calibrator_path=cal_path, label="新模型（校准）")

# ===================== 回测引擎 =====================
def run_backtest(predictions, label, confidence_threshold,
                 stop_loss_pct=STOP_LOSS_PCT, take_profit_pct=TAKE_PROFIT_PCT,
                 position_pct=POSITION_PCT, use_hard_stop=False,
                 use_trailing_stop=False):
    """单次回测引擎"""
    balance = INITIAL_BALANCE
    peak_balance = INITIAL_BALANCE
    position = 0.0
    entry_price = 0.0
    entry_bar = 0
    cooldown = 0
    peak_price = 0.0
    trail_activated = False
    hard_stopped = False

    trades = []
    equity_curve = [balance]
    equity_dates = [data.index[0]]

    closes = data["close"].values
    ma50 = data["close"].rolling(50).mean().values

    for i in range(len(data)):
        price = closes[i]
        p_ema = predictions[i]

        # 更新权益
        if position > 0:
            equity_curve.append(balance + (price - entry_price) * position)
        else:
            equity_curve.append(balance)
        equity_dates.append(data.index[i])

        # 账户级硬止损检查
        if use_hard_stop:
            current_equity = equity_curve[-1]
            if current_equity > peak_balance:
                peak_balance = current_equity
            drawdown = (peak_balance - current_equity) / peak_balance
            if drawdown >= HARD_STOP_PCT and not hard_stopped:
                hard_stopped = True
                # 强制平仓
                if position > 0:
                    pnl = (price - entry_price) * position - abs(entry_price * position * FEE_RATE) - abs(price * position * FEE_RATE)
                    balance += pnl
                    trades.append({"entry": entry_price, "exit": price, "pnl": pnl,
                                   "bars": i - entry_bar, "reason": "硬止损平仓",
                                   "entry_time": str(data.index[entry_bar]),
                                   "exit_time": str(data.index[i])})
                    position = 0; entry_price = 0
                # 停止所有交易
                continue

        if hard_stopped:
            continue

        if cooldown > 0:
            cooldown -= 1

        # ---- 持仓管理 ----
        if position > 0:
            holding_bars = i - entry_bar
            profit_pct = (price - entry_price) / entry_price

            if price > peak_price:
                peak_price = price

            # 移动止损
            if use_trailing_stop:
                if not trail_activated and profit_pct >= 0.015:
                    trail_activated = True
                if trail_activated:
                    if (price - peak_price) / peak_price <= -0.010:
                        pnl = (price - entry_price) * position - abs(entry_price * position * FEE_RATE) - abs(price * position * FEE_RATE)
                        balance += pnl
                        trades.append({"entry": entry_price, "exit": price, "pnl": pnl,
                                       "bars": holding_bars, "reason": "移动止损",
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

            # AI看跌
            sell_th = max(0.2, confidence_threshold - 0.10)
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

        # ---- 开仓 ----
        if position == 0 and cooldown == 0 and p_ema >= confidence_threshold:
            if TREND_FILTER and not np.isnan(ma50[i]) and price < ma50[i]:
                continue
            invest = balance * position_pct
            fee = invest * FEE_RATE
            position = (invest - fee) / price
            entry_price = price
            entry_bar = i
            peak_price = price
            trail_activated = False

    # 强制平仓
    if position > 0:
        price = closes[-1]
        pnl = (price - entry_price) * position - abs(entry_price * position * FEE_RATE) - abs(price * position * FEE_RATE)
        balance += pnl
        trades.append({"entry": entry_price, "exit": price, "pnl": pnl,
                       "bars": len(data) - 1 - entry_bar, "reason": "回测结束",
                       "entry_time": str(data.index[entry_bar]),
                       "exit_time": str(data.index[-1])})

    # 统计
    total_trades = len(trades)
    wins  = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    win_rate = len(wins) / total_trades * 100 if total_trades > 0 else 0
    total_profit = sum(t["pnl"] for t in wins)
    total_loss   = abs(sum(t["pnl"] for t in losses))
    profit_factor = total_profit / total_loss if total_loss > 0 else float("inf")
    net_return = (balance - INITIAL_BALANCE) / INITIAL_BALANCE * 100

    eq = np.array(equity_curve)
    peak_eq = np.maximum.accumulate(eq)
    drawdown = (eq - peak_eq) / peak_eq * 100
    max_drawdown = drawdown.min()

    total_days = (data.index[-1] - data.index[0]).days
    annual_return = ((balance / INITIAL_BALANCE) ** (365 / total_days) - 1) * 100 if total_days > 0 else 0

    reason_counts = {}
    for t in trades:
        r = t["reason"]
        reason_counts[r] = reason_counts.get(r, 0) + 1

    return {
        "label": label,
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
        "hard_stopped": hard_stopped,
        "equity_curve": [round(e, 2) for e in equity_curve[::96]],
        "equity_dates": [str(d) for d in equity_dates[::96]],
        "trades": trades,
    }

# ===================== 运行5组对比回测 =====================
print("\n" + "=" * 70)
print("📊 运行5组对比回测...")
print("=" * 70)

print("\n[A] 旧模型（v20251222，阈值0.70，仓位25%，止损-3%）")
rA = run_backtest(preds_old, "A.旧模型(阈值0.70,仓位25%)", 0.70,
                  stop_loss_pct=-0.03, take_profit_pct=0.06, position_pct=0.25)

print("\n[B] 新模型（5年训练，阈值0.35，仓位10%，止损-2%）")
rB = run_backtest(preds_new, "B.新模型(阈值0.35,仓位10%)", 0.35,
                  stop_loss_pct=-0.02, take_profit_pct=0.04, position_pct=0.10)

print("\n[C] 新模型（5年训练，阈值0.40，仓位10%，止损-2%）")
rC = run_backtest(preds_new, "C.新模型(阈值0.40,仓位10%)", 0.40,
                  stop_loss_pct=-0.02, take_profit_pct=0.04, position_pct=0.10)

print("\n[D] 新模型（阈值0.40，仓位10%，止损-2%，账户硬止损-15%）")
rD = run_backtest(preds_new, "D.新模型+硬止损(阈值0.40)", 0.40,
                  stop_loss_pct=-0.02, take_profit_pct=0.04, position_pct=0.10,
                  use_hard_stop=True)

print("\n[E] 新模型（阈值0.40，仓位10%，止损-2%，移动止损+硬止损）")
rE = run_backtest(preds_new, "E.新模型+移动止损+硬止损", 0.40,
                  stop_loss_pct=-0.02, take_profit_pct=0.04, position_pct=0.10,
                  use_hard_stop=True, use_trailing_stop=True)

results = [rA, rB, rC, rD, rE]

# ===================== 打印结果 =====================
def print_result(r):
    print(f"\n{'─'*60}")
    print(f"  📌 {r['label']}")
    print(f"{'─'*60}")
    print(f"  净收益率:   {r['net_return_pct']:+.2f}%  |  年化: {r['annual_return_pct']:+.2f}%")
    print(f"  最大回撤:   {r['max_drawdown_pct']:.2f}%  |  最终资金: {r['final_balance']:.2f} USDT")
    print(f"  交易次数:   {r['total_trades']}  |  胜率: {r['win_rate']:.2f}%  |  盈利因子: {r['profit_factor']}")
    print(f"  平均盈利:   +{r['avg_profit']:.2f} USDT  |  平均亏损: {r['avg_loss']:.2f} USDT")
    if r.get("hard_stopped"):
        print(f"  ⚠️  账户硬止损已触发（总回撤超过-15%）")
    print(f"  卖出原因: {r['reason_counts']}")

for r in results:
    print_result(r)

# ===================== 汇总表 =====================
print("\n" + "=" * 70)
print("📈 5年回测汇总对比")
print("=" * 70)
col_w = 18
header_row = f"{'指标':<14}" + "".join(f"{r['label'][:16]:<{col_w}}" for r in results)
print(header_row)
print("─" * (14 + col_w * len(results)))
rows_data = [
    ("净收益率",   [f"{r['net_return_pct']:+.2f}%" for r in results]),
    ("年化收益率", [f"{r['annual_return_pct']:+.2f}%" for r in results]),
    ("最大回撤",   [f"{r['max_drawdown_pct']:.2f}%" for r in results]),
    ("最终资金",   [f"{r['final_balance']:.2f}" for r in results]),
    ("交易次数",   [str(r['total_trades']) for r in results]),
    ("胜率",       [f"{r['win_rate']:.2f}%" for r in results]),
    ("盈利因子",   [str(r['profit_factor']) for r in results]),
]
for name, vals in rows_data:
    print(f"{name:<14}" + "".join(f"{v:<{col_w}}" for v in vals))

# ===================== 可视化 =====================
print("\n📊 生成可视化图表...")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Micro Hei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

C_BG   = "#161b22"
C_GRID = "#30363d"
C_TEXT = "#c9d1d9"
COLORS = ["#58a6ff", "#3fb950", "#f78166", "#e3b341", "#bc8cff"]

fig = plt.figure(figsize=(18, 12))
fig.patch.set_facecolor("#0d1117")
gs = GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.30)

def style_ax(ax):
    ax.set_facecolor(C_BG)
    ax.tick_params(colors=C_TEXT, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(C_GRID)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color=C_GRID, linewidth=0.5, alpha=0.7)
    ax.grid(axis="x", color=C_GRID, linewidth=0.3, alpha=0.4)
    ax.xaxis.label.set_color(C_TEXT)
    ax.yaxis.label.set_color(C_TEXT)
    ax.title.set_color(C_TEXT)

# 子图1：权益曲线
ax1 = fig.add_subplot(gs[0, :])
ax1_r = ax1.twinx()
data_daily = data["close"].resample("1D").last().dropna()
ax1_r.fill_between(data_daily.index, data_daily.values, alpha=0.06, color="#e3b341")
ax1_r.plot(data_daily.index, data_daily.values, color="#e3b341", linewidth=0.7, alpha=0.4)
ax1_r.set_ylabel("DOGE价格", color="#e3b341", fontsize=9)
ax1_r.tick_params(colors="#e3b341", labelsize=8)
ax1_r.spines["right"].set_color("#e3b341")
ax1_r.spines["top"].set_visible(False)
ax1_r.spines["left"].set_visible(False)
ax1_r.spines["bottom"].set_visible(False)
ax1_r.set_facecolor(C_BG)

labels_short = ["A.旧模型", "B.新模型0.35", "C.新模型0.40", "D.+硬止损", "E.+移动止损"]
for i, r in enumerate(results):
    eq_dates = pd.to_datetime(r["equity_dates"])
    ax1.plot(eq_dates, r["equity_curve"], color=COLORS[i], linewidth=1.5,
             label=f"{labels_short[i]} ({r['net_return_pct']:+.1f}%)")
ax1.axhline(1000, color="#555", linewidth=0.8, linestyle="--", alpha=0.6)
ax1.set_title("5年回测权益曲线对比（新旧模型 + 风控参数）", fontsize=12)
ax1.set_ylabel("账户权益 (USDT)", fontsize=9)
style_ax(ax1)
ax1.legend(loc="upper left", fontsize=8, facecolor=C_BG, edgecolor=C_GRID, labelcolor=C_TEXT)

# 子图2：年度盈亏
ax2 = fig.add_subplot(gs[1, 0])
def calc_annual_pnl(trades):
    if not trades: return {}
    df = pd.DataFrame(trades)
    df["exit_time"] = pd.to_datetime(df["exit_time"])
    df["year"] = df["exit_time"].dt.year
    return df.groupby("year")["pnl"].sum().to_dict()

all_years = set()
for r in results[:3]:
    all_years.update(calc_annual_pnl(r["trades"]).keys())
years = sorted(all_years)
x = np.arange(len(years))
w = 0.25
for i, r in enumerate(results[:3]):
    annual = calc_annual_pnl(r["trades"])
    vals = [annual.get(y, 0) for y in years]
    colors_bar = [COLORS[i] if v >= 0 else "#ff6b6b" for v in vals]
    ax2.bar(x + (i-1)*w, vals, w, color=colors_bar, alpha=0.85, label=labels_short[i])
ax2.axhline(0, color=C_GRID, linewidth=0.8)
ax2.set_xticks(x)
ax2.set_xticklabels([str(y) for y in years])
ax2.set_title("年度盈亏对比（前3组）", fontsize=10)
ax2.set_ylabel("盈亏 (USDT)", fontsize=9)
style_ax(ax2)
ax2.legend(fontsize=7, facecolor=C_BG, edgecolor=C_GRID, labelcolor=C_TEXT)

# 子图3：关键指标柱状对比
ax3 = fig.add_subplot(gs[1, 1])
metrics = ["净收益率+50", "年化收益+50", "胜率(%)", "盈利因子×10"]
def safe_pf(r):
    pf = r["profit_factor"]
    return float(pf) if pf != "inf" else 10.0
vals_all = []
for r in results:
    v = [r["net_return_pct"]+50, r["annual_return_pct"]+50, r["win_rate"], safe_pf(r)*10]
    vals_all.append(v)
x3 = np.arange(len(metrics))
w3 = 0.15
for i, (r, v) in enumerate(zip(results, vals_all)):
    ax3.bar(x3 + (i-2)*w3, v, w3, color=COLORS[i], alpha=0.85, label=labels_short[i])
ax3.set_xticks(x3)
ax3.set_xticklabels(metrics, fontsize=8)
ax3.set_title("关键指标对比（归一化）", fontsize=10)
style_ax(ax3)
ax3.legend(fontsize=7, facecolor=C_BG, edgecolor=C_GRID, labelcolor=C_TEXT)

fig.suptitle(f"ButterflyBot 新旧模型5年回测对比\nDOGE/USDT 15m | {data.index[0].date()} ~ {data.index[-1].date()} | 初始资金: 1000 USDT",
             fontsize=13, color=C_TEXT, y=0.98)

chart_path = "reports/backtest/backtest_5year_compare_chart.png"
plt.savefig(chart_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"✅ 图表已保存: {chart_path}")
plt.close()

# ===================== 保存JSON报告 =====================
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
report_path = f"reports/backtest/backtest_5year_compare_{timestamp}.json"
report = {
    "timestamp": timestamp,
    "symbol": "DOGE/USDT",
    "timeframe": "15m",
    "data_range": f"{data.index[0]} ~ {data.index[-1]}",
    "total_bars": len(data),
    "config": {
        "initial_balance": INITIAL_BALANCE,
        "fee_rate": FEE_RATE,
        "hard_stop_pct": HARD_STOP_PCT,
    },
    "results": {r["label"]: {k: v for k, v in r.items() if k != "trades"} for r in results},
}
with open(report_path, "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
print(f"✅ JSON报告已保存: {report_path}")
print("=" * 70)
