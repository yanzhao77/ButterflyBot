"""
ButterflyBot 5年回测可视化脚本
生成：权益曲线对比、年度收益、月度热力图、交易分布等图表
"""
import sys, os, json, glob
sys.path.insert(0, ".")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter

# 设置中文字体
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Micro Hei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

os.makedirs("reports/backtest", exist_ok=True)

# ===================== 加载最新回测结果 =====================
report_files = sorted(glob.glob("reports/backtest/backtest_5year_*.json"))
if not report_files:
    print("❌ 未找到5年回测报告文件！")
    sys.exit(1)

report_path = report_files[-1]
print(f"📂 加载回测报告: {report_path}")
with open(report_path, "r", encoding="utf-8") as f:
    report = json.load(f)

r_base    = report["results"]["base"]
r_trail   = report["results"]["trailing"]
r_dyn     = report["results"]["dynamic"]
trades_b  = report["trades"]["base"]
trades_t  = report["trades"]["trailing"]
trades_d  = report["trades"]["dynamic"]

# 权益曲线（采样后）
eq_b = r_base["equity_curve"]
eq_t = r_trail["equity_curve"]
eq_d = r_dyn["equity_curve"]
eq_dates = pd.to_datetime(r_base["equity_dates"])

# 原始数据（用于DOGE价格曲线）
from butterfly_bot.data.fetcher import fetch_ohlcv
data = fetch_ohlcv("DOGE/USDT", "15m", limit=None)
# 按天采样
data_daily = data["close"].resample("1D").last().dropna()

# ===================== 图1：主面板 - 权益曲线 + DOGE价格 =====================
print("📊 绘制权益曲线对比图...")
fig = plt.figure(figsize=(18, 14))
fig.patch.set_facecolor("#0d1117")
gs = GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

# 颜色方案
C_BASE  = "#58a6ff"
C_TRAIL = "#3fb950"
C_DYN   = "#f78166"
C_PRICE = "#e3b341"
C_BG    = "#161b22"
C_GRID  = "#30363d"
C_TEXT  = "#c9d1d9"

def style_ax(ax):
    ax.set_facecolor(C_BG)
    ax.tick_params(colors=C_TEXT, labelsize=9)
    ax.spines["bottom"].set_color(C_GRID)
    ax.spines["left"].set_color(C_GRID)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color=C_GRID, linewidth=0.5, alpha=0.7)
    ax.grid(axis="x", color=C_GRID, linewidth=0.3, alpha=0.4)
    ax.xaxis.label.set_color(C_TEXT)
    ax.yaxis.label.set_color(C_TEXT)
    ax.title.set_color(C_TEXT)

# ---- 子图1：权益曲线对比 ----
ax1 = fig.add_subplot(gs[0, :])
ax1_r = ax1.twinx()

# DOGE价格（右轴）
ax1_r.fill_between(data_daily.index, data_daily.values, alpha=0.08, color=C_PRICE)
ax1_r.plot(data_daily.index, data_daily.values, color=C_PRICE, linewidth=0.8, alpha=0.5, label="DOGE价格")
ax1_r.set_ylabel("DOGE/USDT 价格", color=C_PRICE, fontsize=9)
ax1_r.tick_params(colors=C_PRICE, labelsize=8)
ax1_r.spines["right"].set_color(C_PRICE)
ax1_r.spines["top"].set_visible(False)
ax1_r.spines["left"].set_visible(False)
ax1_r.spines["bottom"].set_visible(False)
ax1_r.set_facecolor(C_BG)

# 权益曲线（左轴）
ax1.plot(eq_dates, eq_b, color=C_BASE,  linewidth=1.5, label=f"基准  ({r_base['net_return_pct']:+.1f}%)")
ax1.plot(eq_dates, eq_t, color=C_TRAIL, linewidth=1.5, label=f"移动止损 ({r_trail['net_return_pct']:+.1f}%)")
ax1.plot(eq_dates, eq_d, color=C_DYN,   linewidth=1.5, label=f"动态阈值 ({r_dyn['net_return_pct']:+.1f}%)")
ax1.axhline(1000, color="#555", linewidth=0.8, linestyle="--", alpha=0.6)
ax1.set_title("ButterflyBot 5年回测 - 权益曲线对比 (DOGE/USDT 15m, 2020-2025)", fontsize=13, pad=10)
ax1.set_ylabel("账户权益 (USDT)", fontsize=9)
ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0f}"))
style_ax(ax1)
ax1.legend(loc="upper left", fontsize=9, facecolor=C_BG, edgecolor=C_GRID, labelcolor=C_TEXT)

# ---- 子图2：年度收益柱状图（基准策略）----
ax2 = fig.add_subplot(gs[1, 0])
# 从trades计算每年收益
def calc_annual_pnl(trades):
    if not trades:
        return {}
    df = pd.DataFrame(trades)
    df["exit_time"] = pd.to_datetime(df["exit_time"])
    df["year"] = df["exit_time"].dt.year
    return df.groupby("year")["pnl"].sum().to_dict()

annual_b = calc_annual_pnl(trades_b)
annual_t = calc_annual_pnl(trades_t)
years = sorted(set(list(annual_b.keys()) + list(annual_t.keys())))
x = np.arange(len(years))
w = 0.35
bars_b = [annual_b.get(y, 0) for y in years]
bars_t = [annual_t.get(y, 0) for y in years]
colors_b = [C_BASE if v >= 0 else C_DYN for v in bars_b]
colors_t = [C_TRAIL if v >= 0 else "#ff6b6b" for v in bars_t]
ax2.bar(x - w/2, bars_b, w, color=colors_b, alpha=0.85, label="基准")
ax2.bar(x + w/2, bars_t, w, color=colors_t, alpha=0.85, label="移动止损")
ax2.axhline(0, color=C_GRID, linewidth=0.8)
ax2.set_xticks(x)
ax2.set_xticklabels([str(y) for y in years])
ax2.set_title("年度盈亏对比 (USDT)", fontsize=10)
ax2.set_ylabel("盈亏 (USDT)", fontsize=9)
style_ax(ax2)
ax2.legend(fontsize=8, facecolor=C_BG, edgecolor=C_GRID, labelcolor=C_TEXT)

# ---- 子图3：月度盈亏热力图（基准策略）----
ax3 = fig.add_subplot(gs[1, 1])
if trades_b:
    df_b = pd.DataFrame(trades_b)
    df_b["exit_time"] = pd.to_datetime(df_b["exit_time"])
    df_b["year"]  = df_b["exit_time"].dt.year
    df_b["month"] = df_b["exit_time"].dt.month
    monthly = df_b.groupby(["year", "month"])["pnl"].sum().unstack(fill_value=0)
    # 确保12个月都有
    for m in range(1, 13):
        if m not in monthly.columns:
            monthly[m] = 0
    monthly = monthly[[m for m in range(1, 13)]]
    
    vmax = max(abs(monthly.values.max()), abs(monthly.values.min()), 1)
    im = ax3.imshow(monthly.values, aspect="auto", cmap="RdYlGn",
                    vmin=-vmax, vmax=vmax, interpolation="nearest")
    ax3.set_xticks(range(12))
    ax3.set_xticklabels(["1月","2月","3月","4月","5月","6月","7月","8月","9月","10月","11月","12月"], fontsize=7)
    ax3.set_yticks(range(len(monthly.index)))
    ax3.set_yticklabels([str(y) for y in monthly.index], fontsize=8)
    ax3.set_title("月度盈亏热力图（基准策略）", fontsize=10)
    ax3.tick_params(colors=C_TEXT)
    ax3.set_facecolor(C_BG)
    ax3.spines[:].set_color(C_GRID)
    # 在格子中写数值
    for i in range(len(monthly.index)):
        for j in range(12):
            val = monthly.values[i, j]
            if abs(val) > 0.1:
                ax3.text(j, i, f"{val:.0f}", ha="center", va="center",
                         fontsize=6, color="white" if abs(val) > vmax * 0.5 else "black")
    plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04).ax.tick_params(colors=C_TEXT, labelsize=7)

# ---- 子图4：交易盈亏分布（基准策略）----
ax4 = fig.add_subplot(gs[2, 0])
if trades_b:
    pnls = [t["pnl"] for t in trades_b]
    wins  = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    bins = np.linspace(min(pnls), max(pnls), 40)
    ax4.hist(losses, bins=bins, color=C_DYN,  alpha=0.8, label=f"亏损 ({len(losses)}笔)")
    ax4.hist(wins,   bins=bins, color=C_BASE, alpha=0.8, label=f"盈利 ({len(wins)}笔)")
    ax4.axvline(0, color="white", linewidth=0.8, linestyle="--")
    ax4.set_title("交易盈亏分布（基准策略）", fontsize=10)
    ax4.set_xlabel("单笔盈亏 (USDT)", fontsize=9)
    ax4.set_ylabel("频次", fontsize=9)
    style_ax(ax4)
    ax4.legend(fontsize=8, facecolor=C_BG, edgecolor=C_GRID, labelcolor=C_TEXT)

# ---- 子图5：关键指标雷达/柱状对比 ----
ax5 = fig.add_subplot(gs[2, 1])
metrics = ["胜率(%)", "盈利因子×10", "年化收益%+50", "最大回撤%+100"]
pf_b = r_base["profit_factor"] if r_base["profit_factor"] != "inf" else 10
pf_t = r_trail["profit_factor"] if r_trail["profit_factor"] != "inf" else 10
pf_d = r_dyn["profit_factor"] if r_dyn["profit_factor"] != "inf" else 10
vals_b = [r_base["win_rate"],  pf_b * 10, r_base["annual_return_pct"] + 50, r_base["max_drawdown_pct"] + 100]
vals_t = [r_trail["win_rate"], pf_t * 10, r_trail["annual_return_pct"] + 50, r_trail["max_drawdown_pct"] + 100]
vals_d = [r_dyn["win_rate"],   pf_d * 10, r_dyn["annual_return_pct"] + 50, r_dyn["max_drawdown_pct"] + 100]
x5 = np.arange(len(metrics))
w5 = 0.25
ax5.bar(x5 - w5, vals_b, w5, color=C_BASE,  alpha=0.85, label="基准")
ax5.bar(x5,      vals_t, w5, color=C_TRAIL, alpha=0.85, label="移动止损")
ax5.bar(x5 + w5, vals_d, w5, color=C_DYN,   alpha=0.85, label="动态阈值")
ax5.set_xticks(x5)
ax5.set_xticklabels(metrics, fontsize=8)
ax5.set_title("关键指标对比（归一化）", fontsize=10)
style_ax(ax5)
ax5.legend(fontsize=8, facecolor=C_BG, edgecolor=C_GRID, labelcolor=C_TEXT)

# 总标题
fig.suptitle(f"ButterflyBot 量化策略 5年回测分析报告\nDOGE/USDT 15m | {report['data_range']} | 初始资金: 1000 USDT",
             fontsize=14, color=C_TEXT, y=0.98)

out_path = "reports/backtest/backtest_5year_chart.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"✅ 图表已保存: {out_path}")
plt.close()

# ===================== 图2：p_ema 分布与信号质量 =====================
print("📊 绘制p_ema分布与信号质量图...")
from butterfly_bot.data.features import add_features, get_feature_columns
import joblib
from butterfly_bot.model.model_registry import load_latest_model_path

model_path = load_latest_model_path()
model = joblib.load(model_path)
features_df = add_features(data)
feature_cols = get_feature_columns()
available_cols = [c for c in feature_cols if c in features_df.columns]
X = features_df[available_cols].values
raw_probs = model.predict(X)

aligned_probs = pd.Series(np.nan, index=data.index)
aligned_probs.loc[features_df.index] = raw_probs
alpha = 2.0 / (10 + 1.0)
p_ema_val = None
pema_list = []
for prob in aligned_probs.values:
    if not np.isnan(prob):
        p_ema_val = prob if p_ema_val is None else alpha * prob + (1 - alpha) * p_ema_val
    pema_list.append(p_ema_val if p_ema_val is not None else 0.3)
predictions = np.array(pema_list)

# 按年分组统计p_ema分布
pema_series = pd.Series(predictions, index=data.index)

fig2, axes = plt.subplots(2, 3, figsize=(15, 8))
fig2.patch.set_facecolor("#0d1117")
fig2.suptitle("p_ema 年度分布分析（模型预测概率）", fontsize=13, color=C_TEXT)

years_all = sorted(pema_series.index.year.unique())
# 选取最近6年
years_plot = years_all[-6:]

for idx, yr in enumerate(years_plot):
    ax = axes[idx // 3][idx % 3]
    ax.set_facecolor(C_BG)
    yr_data = pema_series[pema_series.index.year == yr]
    ax.hist(yr_data.values, bins=50, color=C_BASE, alpha=0.8, edgecolor="none")
    ax.axvline(0.70, color=C_DYN, linewidth=1.5, linestyle="--", label="阈值0.70")
    q85 = np.quantile(yr_data.values, 0.85)
    ax.axvline(q85, color=C_TRAIL, linewidth=1.2, linestyle=":", label=f"85%分位={q85:.2f}")
    above_th = (yr_data >= 0.70).sum()
    ax.set_title(f"{yr}年 (超阈值{above_th}次/{len(yr_data)})", fontsize=9, color=C_TEXT)
    ax.tick_params(colors=C_TEXT, labelsize=8)
    ax.spines[:].set_color(C_GRID)
    ax.grid(axis="y", color=C_GRID, linewidth=0.4, alpha=0.5)
    ax.legend(fontsize=7, facecolor=C_BG, edgecolor=C_GRID, labelcolor=C_TEXT)

plt.tight_layout()
out_path2 = "reports/backtest/backtest_5year_pema_dist.png"
plt.savefig(out_path2, dpi=150, bbox_inches="tight", facecolor=fig2.get_facecolor())
print(f"✅ p_ema分布图已保存: {out_path2}")
plt.close()

print("\n✅ 所有图表生成完成！")
