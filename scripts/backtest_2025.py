"""
ButterflyBot 2025年专项回测脚本
使用新模型（v20260427_145956，5年训练，Platt校准）
数据范围：2025-01-01 ~ 2025-12-23（34,184 根 K 线）

回测参数（当前生产配置）：
  - 初始资金: 1000 USDT
  - 仓位: 10%
  - 固定止损: -2%
  - 止盈: +4%
  - 手续费: 0.1%
  - 趋势过滤: MA50
  - 账户硬止损: -15%
  - 置信度阈值: 0.40
"""
import sys, os, json, logging, time
sys.path.insert(0, ".")
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
from datetime import datetime

logging.basicConfig(level=logging.WARNING)

from butterfly_bot.data.fetcher import fetch_ohlcv
from butterfly_bot.model.model_registry import REGISTRY_DIR

os.makedirs("reports/backtest", exist_ok=True)

# ===================== 回测参数 =====================
INITIAL_BALANCE   = 1000.0
POSITION_PCT      = 0.10
FEE_RATE          = 0.001
STOP_LOSS_PCT     = -0.02
TAKE_PROFIT_PCT   = 0.04
MAX_HOLDING_BARS  = 50
COOLDOWN_BARS     = 5
PROB_EMA_SPAN     = 10
TREND_FILTER      = True
HARD_STOP_PCT     = 0.15
CONFIDENCE_TH     = 0.40

print("=" * 70)
print("🚀 ButterflyBot 2025年专项回测")
print("=" * 70)

# ===================== 1. 加载全量数据（用于特征工程）=====================
print("\n📥 加载历史数据（含预热期）...")
df_all = fetch_ohlcv("DOGE/USDT", "15m", limit=None)
print(f"✅ 全量数据: {len(df_all)} 根K线")

# ===================== 2. 特征工程 =====================
print("\n🔧 构建特征...")
import butterfly_bot.config.settings as settings
settings.TARGET_SHIFT = 2
settings.TARGET_THRESHOLD = 0.003
import importlib
import butterfly_bot.data.features as feat_module
importlib.reload(feat_module)

df_feat_all = feat_module.add_features(df_all)
feature_cols = feat_module.get_feature_columns()
available_cols = [c for c in feature_cols if c in df_feat_all.columns]
X_all = df_feat_all[available_cols].values
print(f"✅ 特征工程完成: {len(df_feat_all)} 行，{len(available_cols)} 个特征")

# ===================== 3. 加载模型并生成全量预测 =====================
print("\n🤖 加载新模型（v20260427_145956）...")
model_path = os.path.join(REGISTRY_DIR, "v20260427_145956.pkl")
cal_path   = os.path.join(REGISTRY_DIR, "platt_calibrator.pkl")

model = joblib.load(model_path)
calibrator = joblib.load(cal_path)

raw_probs = model.predict(X_all)
cal_probs = calibrator.predict_proba(raw_probs.reshape(-1, 1))[:, 1]

# 对齐到原始索引
aligned = pd.Series(np.nan, index=df_all.index)
aligned.loc[df_feat_all.index] = cal_probs

# 逐步EMA平滑
alpha = 2.0 / (PROB_EMA_SPAN + 1.0)
p_ema_val = None
pema_list = []
for prob in aligned.values:
    if not np.isnan(prob):
        p_ema_val = prob if p_ema_val is None else alpha * prob + (1 - alpha) * p_ema_val
    pema_list.append(p_ema_val if p_ema_val is not None else 0.25)
pema_all = np.array(pema_list)

# ===================== 4. 截取2025年数据 =====================
mask_2025 = df_all.index.year == 2025
df_2025 = df_all[mask_2025].copy()
pema_2025 = pema_all[mask_2025]

print(f"\n📅 2025年数据: {len(df_2025)} 根K线")
print(f"   时间范围: {df_2025.index[0]} ~ {df_2025.index[-1]}")
print(f"   p_ema 统计: min={pema_2025.min():.4f}, max={pema_2025.max():.4f}, mean={pema_2025.mean():.4f}")
print(f"   超阈值0.40次数: {(pema_2025 >= CONFIDENCE_TH).sum()}")
print(f"   超阈值0.45次数: {(pema_2025 >= 0.45).sum()}")

# ===================== 5. 回测引擎 =====================
def run_backtest_2025(df, pema, label, conf_th, sl_pct, tp_pct, pos_pct, use_hard_stop=True):
    balance = INITIAL_BALANCE
    peak_balance = INITIAL_BALANCE
    position = 0.0
    entry_price = 0.0
    entry_bar = 0
    cooldown = 0
    peak_price = 0.0
    hard_stopped = False

    trades = []
    equity_curve = []
    equity_dates = []

    closes = df["close"].values
    ma50 = df["close"].rolling(50).mean().values

    for i in range(len(df)):
        price = closes[i]
        p = pema[i]

        cur_equity = balance + (price - entry_price) * position if position > 0 else balance
        equity_curve.append(cur_equity)
        equity_dates.append(df.index[i])

        # 账户硬止损
        if use_hard_stop:
            if cur_equity > peak_balance:
                peak_balance = cur_equity
            dd = (peak_balance - cur_equity) / peak_balance
            if dd >= HARD_STOP_PCT and not hard_stopped:
                hard_stopped = True
                if position > 0:
                    pnl = (price - entry_price) * position - (entry_price * position + price * position) * FEE_RATE
                    balance += pnl
                    trades.append({"entry": entry_price, "exit": price, "pnl": pnl,
                                   "bars": i - entry_bar, "reason": "账户硬止损",
                                   "entry_time": str(df.index[entry_bar]),
                                   "exit_time": str(df.index[i]),
                                   "entry_idx": entry_bar, "exit_idx": i})
                    position = 0

        if hard_stopped:
            continue

        if cooldown > 0:
            cooldown -= 1

        # 持仓管理
        if position > 0:
            holding_bars = i - entry_bar
            profit_pct = (price - entry_price) / entry_price
            if price > peak_price:
                peak_price = price

            if profit_pct <= sl_pct:
                pnl = (price - entry_price) * position - (entry_price * position + price * position) * FEE_RATE
                balance += pnl
                trades.append({"entry": entry_price, "exit": price, "pnl": pnl,
                               "bars": holding_bars, "reason": "固定止损",
                               "entry_time": str(df.index[entry_bar]),
                               "exit_time": str(df.index[i]),
                               "entry_idx": entry_bar, "exit_idx": i})
                position = 0; entry_price = 0; peak_price = 0
                cooldown = COOLDOWN_BARS
                continue

            if profit_pct >= tp_pct:
                pnl = (price - entry_price) * position - (entry_price * position + price * position) * FEE_RATE
                balance += pnl
                trades.append({"entry": entry_price, "exit": price, "pnl": pnl,
                               "bars": holding_bars, "reason": "止盈",
                               "entry_time": str(df.index[entry_bar]),
                               "exit_time": str(df.index[i]),
                               "entry_idx": entry_bar, "exit_idx": i})
                position = 0; entry_price = 0; peak_price = 0
                cooldown = COOLDOWN_BARS
                continue

            sell_th = max(0.20, conf_th - 0.10)
            if p <= sell_th:
                pnl = (price - entry_price) * position - (entry_price * position + price * position) * FEE_RATE
                balance += pnl
                trades.append({"entry": entry_price, "exit": price, "pnl": pnl,
                               "bars": holding_bars, "reason": "AI看跌",
                               "entry_time": str(df.index[entry_bar]),
                               "exit_time": str(df.index[i]),
                               "entry_idx": entry_bar, "exit_idx": i})
                position = 0; entry_price = 0; peak_price = 0
                cooldown = COOLDOWN_BARS
                continue

            if holding_bars >= MAX_HOLDING_BARS:
                pnl = (price - entry_price) * position - (entry_price * position + price * position) * FEE_RATE
                balance += pnl
                trades.append({"entry": entry_price, "exit": price, "pnl": pnl,
                               "bars": holding_bars, "reason": "时间止损",
                               "entry_time": str(df.index[entry_bar]),
                               "exit_time": str(df.index[i]),
                               "entry_idx": entry_bar, "exit_idx": i})
                position = 0; entry_price = 0; peak_price = 0
                cooldown = COOLDOWN_BARS
                continue

        # 开仓
        if position == 0 and cooldown == 0 and p >= conf_th:
            if TREND_FILTER and not np.isnan(ma50[i]) and price < ma50[i]:
                continue
            invest = balance * pos_pct
            fee = invest * FEE_RATE
            position = (invest - fee) / price
            entry_price = price
            entry_bar = i
            peak_price = price

    # 强制平仓
    if position > 0:
        price = closes[-1]
        pnl = (price - entry_price) * position - (entry_price * position + price * position) * FEE_RATE
        balance += pnl
        trades.append({"entry": entry_price, "exit": price, "pnl": pnl,
                       "bars": len(df) - 1 - entry_bar, "reason": "回测结束",
                       "entry_time": str(df.index[entry_bar]),
                       "exit_time": str(df.index[-1]),
                       "entry_idx": entry_bar, "exit_idx": len(df)-1})

    total_trades = len(trades)
    wins   = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    win_rate = len(wins) / total_trades * 100 if total_trades > 0 else 0
    total_profit = sum(t["pnl"] for t in wins)
    total_loss   = abs(sum(t["pnl"] for t in losses))
    profit_factor = total_profit / total_loss if total_loss > 0 else float("inf")
    net_return = (balance - INITIAL_BALANCE) / INITIAL_BALANCE * 100

    eq = np.array(equity_curve)
    peak_eq = np.maximum.accumulate(eq)
    dd_arr  = (eq - peak_eq) / peak_eq * 100
    max_dd  = dd_arr.min()

    reason_counts = {}
    for t in trades:
        reason_counts[t["reason"]] = reason_counts.get(t["reason"], 0) + 1

    return {
        "label": label,
        "final_balance": round(balance, 2),
        "net_return_pct": round(net_return, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "total_trades": total_trades,
        "win_rate": round(win_rate, 2),
        "profit_factor": round(profit_factor, 3) if profit_factor != float("inf") else "inf",
        "total_profit": round(total_profit, 2),
        "total_loss": round(total_loss, 2),
        "avg_profit": round(total_profit / len(wins), 2) if wins else 0,
        "avg_loss": round(-total_loss / len(losses), 2) if losses else 0,
        "reason_counts": reason_counts,
        "hard_stopped": hard_stopped,
        "equity_curve": equity_curve,
        "equity_dates": equity_dates,
        "drawdown_curve": dd_arr.tolist(),
        "trades": trades,
    }

# ===================== 6. 运行3组回测 =====================
print("\n" + "=" * 70)
print("📊 运行2025年回测（3组参数对比）...")
print("=" * 70)

print("\n[A] 当前生产配置（阈值0.40，止损-2%，止盈+4%，硬止损-15%）")
rA = run_backtest_2025(df_2025, pema_2025, "A.当前配置(0.40,-2%,+4%)", 0.40, -0.02, 0.04, 0.10)

print("\n[B] 宽松阈值（阈值0.35，止损-2%，止盈+4%）")
rB = run_backtest_2025(df_2025, pema_2025, "B.宽松阈值(0.35,-2%,+4%)", 0.35, -0.02, 0.04, 0.10)

print("\n[C] 收紧止损（阈值0.40，止损-1.5%，止盈+3%）")
rC = run_backtest_2025(df_2025, pema_2025, "C.收紧止损(0.40,-1.5%,+3%)", 0.40, -0.015, 0.03, 0.10)

results = [rA, rB, rC]

# ===================== 7. 打印结果 =====================
for r in results:
    print(f"\n{'─'*60}")
    print(f"  📌 {r['label']}")
    print(f"{'─'*60}")
    print(f"  净收益率:   {r['net_return_pct']:+.2f}%  |  最终资金: {r['final_balance']:.2f} USDT")
    print(f"  最大回撤:   {r['max_drawdown_pct']:.2f}%")
    print(f"  交易次数:   {r['total_trades']}  |  胜率: {r['win_rate']:.2f}%  |  盈利因子: {r['profit_factor']}")
    print(f"  平均盈利:   +{r['avg_profit']:.2f} USDT  |  平均亏损: {r['avg_loss']:.2f} USDT")
    print(f"  卖出原因:   {r['reason_counts']}")
    if r.get("hard_stopped"):
        print(f"  ⚠️  账户硬止损已触发")

# ===================== 8. 月度盈亏分析 =====================
print("\n📅 月度盈亏分析（A组）:")
if rA["trades"]:
    df_trades = pd.DataFrame(rA["trades"])
    df_trades["exit_time"] = pd.to_datetime(df_trades["exit_time"])
    df_trades["month"] = df_trades["exit_time"].dt.to_period("M")
    monthly = df_trades.groupby("month").agg(
        trades=("pnl", "count"),
        pnl=("pnl", "sum"),
        wins=("pnl", lambda x: (x > 0).sum())
    )
    monthly["win_rate"] = monthly["wins"] / monthly["trades"] * 100
    for idx, row in monthly.iterrows():
        sign = "✅" if row["pnl"] >= 0 else "❌"
        print(f"  {sign} {idx}: {int(row['trades']):3d}笔, 盈亏={row['pnl']:+.2f} USDT, 胜率={row['win_rate']:.0f}%")

# ===================== 9. 可视化 =====================
print("\n📊 生成可视化图表...")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Micro Hei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

C_BG   = "#161b22"
C_GRID = "#30363d"
C_TEXT = "#c9d1d9"
COLORS = ["#58a6ff", "#3fb950", "#f78166"]

fig = plt.figure(figsize=(18, 14))
fig.patch.set_facecolor("#0d1117")
gs = GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.30,
              height_ratios=[2.5, 1.5, 1.5])

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

dates_2025 = pd.to_datetime(rA["equity_dates"])

# ── 子图1：权益曲线 ──
ax1 = fig.add_subplot(gs[0, :])
ax1_r = ax1.twinx()
ax1_r.fill_between(df_2025.index, df_2025["close"].values, alpha=0.07, color="#e3b341")
ax1_r.plot(df_2025.index, df_2025["close"].values, color="#e3b341", linewidth=0.8, alpha=0.5)
ax1_r.set_ylabel("DOGE 价格 (USDT)", color="#e3b341", fontsize=9)
ax1_r.tick_params(colors="#e3b341", labelsize=8)
ax1_r.spines["right"].set_color("#e3b341")
for sp in ["top","left","bottom"]:
    ax1_r.spines[sp].set_visible(False)
ax1_r.set_facecolor(C_BG)

for i, r in enumerate(results):
    ax1.plot(pd.to_datetime(r["equity_dates"]), r["equity_curve"],
             color=COLORS[i], linewidth=1.8,
             label=f"{r['label']} ({r['net_return_pct']:+.2f}%)")
ax1.axhline(1000, color="#555", linewidth=0.8, linestyle="--", alpha=0.6, label="初始资金")

# 标注A组的交易点
if rA["trades"]:
    for t in rA["trades"]:
        ei = t["entry_idx"]
        xi = t["exit_idx"]
        if ei < len(df_2025) and xi < len(df_2025):
            ax1.axvline(df_2025.index[ei], color="#58a6ff", linewidth=0.4, alpha=0.3)

ax1.set_title("2025年回测权益曲线（3组参数对比）", fontsize=12)
ax1.set_ylabel("账户权益 (USDT)", fontsize=9)
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax1.xaxis.set_major_locator(mdates.MonthLocator())
style_ax(ax1)
ax1.legend(loc="upper left", fontsize=9, facecolor=C_BG, edgecolor=C_GRID, labelcolor=C_TEXT)

# ── 子图2：回撤曲线 ──
ax2 = fig.add_subplot(gs[1, :])
for i, r in enumerate(results):
    ax2.fill_between(pd.to_datetime(r["equity_dates"]), r["drawdown_curve"],
                     alpha=0.25, color=COLORS[i])
    ax2.plot(pd.to_datetime(r["equity_dates"]), r["drawdown_curve"],
             color=COLORS[i], linewidth=1.2, label=r["label"].split("(")[0])
ax2.axhline(-15, color="#ff6b6b", linewidth=1.2, linestyle="--", alpha=0.8, label="硬止损线 -15%")
ax2.axhline(0, color="#555", linewidth=0.5)
ax2.set_title("回撤曲线", fontsize=10)
ax2.set_ylabel("回撤 (%)", fontsize=9)
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax2.xaxis.set_major_locator(mdates.MonthLocator())
style_ax(ax2)
ax2.legend(fontsize=8, facecolor=C_BG, edgecolor=C_GRID, labelcolor=C_TEXT)

# ── 子图3：月度盈亏（A组）──
ax3 = fig.add_subplot(gs[2, 0])
if rA["trades"]:
    df_t = pd.DataFrame(rA["trades"])
    df_t["exit_time"] = pd.to_datetime(df_t["exit_time"])
    df_t["month"] = df_t["exit_time"].dt.to_period("M").astype(str)
    monthly_pnl = df_t.groupby("month")["pnl"].sum()
    bar_colors = [COLORS[1] if v >= 0 else COLORS[2] for v in monthly_pnl.values]
    ax3.bar(range(len(monthly_pnl)), monthly_pnl.values, color=bar_colors, alpha=0.85)
    ax3.set_xticks(range(len(monthly_pnl)))
    ax3.set_xticklabels([m[5:] for m in monthly_pnl.index], fontsize=8, rotation=45)
    ax3.axhline(0, color=C_GRID, linewidth=0.8)
    ax3.set_title("月度盈亏 (A组)", fontsize=10)
    ax3.set_ylabel("盈亏 (USDT)", fontsize=9)
style_ax(ax3)

# ── 子图4：卖出原因饼图（A组）──
ax4 = fig.add_subplot(gs[2, 1])
if rA["reason_counts"]:
    rc = rA["reason_counts"]
    pie_colors = {"固定止损": "#f78166", "止盈": "#3fb950", "AI看跌": "#58a6ff",
                  "时间止损": "#e3b341", "账户硬止损": "#ff0000", "回测结束": "#888"}
    labels = list(rc.keys())
    sizes  = list(rc.values())
    colors_pie = [pie_colors.get(l, "#aaa") for l in labels]
    wedges, texts, autotexts = ax4.pie(
        sizes, labels=labels, colors=colors_pie,
        autopct="%1.0f%%", startangle=90,
        textprops={"color": C_TEXT, "fontsize": 9},
        wedgeprops={"edgecolor": "#0d1117", "linewidth": 1.5}
    )
    for at in autotexts:
        at.set_color(C_TEXT)
        at.set_fontsize(8)
    ax4.set_title("卖出原因分布 (A组)", fontsize=10, color=C_TEXT)
    ax4.set_facecolor(C_BG)

# 汇总信息文字
summary_text = (
    f"2025年回测汇总\n"
    f"{'─'*28}\n"
    f"数据: 2025-01-01 ~ 2025-12-23\n"
    f"K线: {len(df_2025):,} 根 (15m)\n"
    f"{'─'*28}\n"
    f"A组净收益: {rA['net_return_pct']:+.2f}%\n"
    f"A组最大回撤: {rA['max_drawdown_pct']:.2f}%\n"
    f"A组交易次数: {rA['total_trades']}\n"
    f"A组胜率: {rA['win_rate']:.2f}%\n"
    f"A组盈利因子: {rA['profit_factor']}"
)
fig.text(0.01, 0.01, summary_text, fontsize=8, color=C_TEXT,
         va="bottom", ha="left",
         bbox=dict(boxstyle="round", facecolor=C_BG, edgecolor=C_GRID, alpha=0.8))

fig.suptitle(
    f"ButterflyBot 2025年专项回测\nDOGE/USDT 15m | 新模型 v20260427 | 初始资金: 1,000 USDT",
    fontsize=13, color=C_TEXT, y=0.99
)

chart_path = "reports/backtest/backtest_2025_chart.png"
plt.savefig(chart_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"✅ 图表已保存: {chart_path}")
plt.close()

# ===================== 10. 保存详细JSON =====================
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
report_data = {
    "timestamp": timestamp,
    "symbol": "DOGE/USDT",
    "timeframe": "15m",
    "period": "2025-01-01 ~ 2025-12-23",
    "total_bars": len(df_2025),
    "model": "v20260427_145956",
    "config": {
        "initial_balance": INITIAL_BALANCE,
        "position_pct": POSITION_PCT,
        "fee_rate": FEE_RATE,
        "hard_stop_pct": HARD_STOP_PCT,
    },
    "results": {
        r["label"]: {k: v for k, v in r.items() if k not in ("equity_curve", "equity_dates", "drawdown_curve", "trades")}
        for r in results
    },
}
json_path = f"reports/backtest/backtest_2025_{timestamp}.json"
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(report_data, f, ensure_ascii=False, indent=2)
print(f"✅ JSON报告已保存: {json_path}")

# ===================== 11. p_ema 分布分析 =====================
print("\n📊 2025年 p_ema 分布分析:")
print(f"   均值:       {pema_2025.mean():.4f}")
print(f"   中位数:     {np.median(pema_2025):.4f}")
print(f"   75%分位:    {np.quantile(pema_2025, 0.75):.4f}")
print(f"   85%分位:    {np.quantile(pema_2025, 0.85):.4f}")
print(f"   90%分位:    {np.quantile(pema_2025, 0.90):.4f}")
print(f"   95%分位:    {np.quantile(pema_2025, 0.95):.4f}")
print(f"   最大值:     {pema_2025.max():.4f}")
print(f"   超0.35次数: {(pema_2025 >= 0.35).sum()}")
print(f"   超0.40次数: {(pema_2025 >= 0.40).sum()}")
print(f"   超0.45次数: {(pema_2025 >= 0.45).sum()}")

print("\n" + "=" * 70)
print("✅ 2025年回测完成！")
print(f"   推荐配置(A组): 净收益={rA['net_return_pct']:+.2f}%, 最大回撤={rA['max_drawdown_pct']:.2f}%, 交易{rA['total_trades']}笔")
print("=" * 70)
