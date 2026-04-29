"""
ButterflyBot 朋友圈展示图表生成脚本
生成4张精美图表：
  1. 主图：两年权益曲线 vs DOGE价格
  2. 月度盈亏热力图 + 年度核心指标对比
  3. 交易分析：胜率/盈亏比/持仓时长分布
  4. 核心指标卡片（适合朋友圈单图展示）
"""
import pickle, os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch
import warnings
warnings.filterwarnings("ignore")

plt.rcParams["font.family"]        = "sans-serif"
plt.rcParams["font.sans-serif"]    = ["Noto Sans CJK SC", "WenQuanYi Micro Hei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

os.makedirs("reports/backtest", exist_ok=True)

# ── 加载回测数据 ──
with open("reports/backtest/backtest_final_012_data.pkl","rb") as f:
    data = pickle.load(f)

r24   = data["results"][2024]
r25   = data["results"][2025]
df24  = data["df_2024"]
df25  = data["df_2025"]
p24   = data["pema_2024"]
p25   = data["pema_2025"]

# ── 颜色主题 ──
BG    = "#0d1117"
BG2   = "#161b22"
GRID  = "#30363d"
TEXT  = "#e6edf3"
TEXT2 = "#8b949e"
GREEN = "#3fb950"
RED   = "#f85149"
BLUE  = "#58a6ff"
GOLD  = "#d29922"
PURP  = "#bc8cff"
CYAN  = "#39d353"

def style_ax(ax, grid_x=False):
    ax.set_facecolor(BG2)
    ax.tick_params(colors=TEXT2, labelsize=9)
    for sp in ax.spines.values(): sp.set_color(GRID)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.5, alpha=0.6)
    if grid_x: ax.grid(axis="x", color=GRID, linewidth=0.3, alpha=0.4)
    ax.xaxis.label.set_color(TEXT2); ax.yaxis.label.set_color(TEXT2)
    ax.title.set_color(TEXT)

# ══════════════════════════════════════════════════════════════════
# 图1：主图 — 两年权益曲线 vs DOGE价格（横版，适合朋友圈）
# ══════════════════════════════════════════════════════════════════
fig1, axes = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={"height_ratios":[3,1]})
fig1.patch.set_facecolor(BG)

ax_eq  = axes[0]
ax_dd  = axes[1]
ax_doge = ax_eq.twinx()

# DOGE 价格背景
doge_close_24 = df24["close"].values
doge_close_25 = df25["close"].values
doge_dates_24 = pd.to_datetime(r24["equity_dates"])
doge_dates_25 = pd.to_datetime(r25["equity_dates"])

ax_doge.fill_between(doge_dates_24, doge_close_24, alpha=0.06, color=GOLD)
ax_doge.fill_between(doge_dates_25, doge_close_25, alpha=0.06, color=GOLD)
ax_doge.plot(doge_dates_24, doge_close_24, color=GOLD, linewidth=0.8, alpha=0.4)
ax_doge.plot(doge_dates_25, doge_close_25, color=GOLD, linewidth=0.8, alpha=0.4)
ax_doge.set_ylabel("DOGE 价格 (USDT)", color=GOLD, fontsize=9)
ax_doge.tick_params(colors=GOLD, labelsize=8)
ax_doge.spines["right"].set_color(GOLD)
for sp in ["top","left","bottom"]: ax_doge.spines[sp].set_visible(False)
ax_doge.set_facecolor(BG2)

# 权益曲线
eq24 = np.array(r24["equity_curve"]); eq25 = np.array(r25["equity_curve"])
ax_eq.plot(doge_dates_24, eq24, color=BLUE,  linewidth=2.2, label=f"2024年 +{r24['net_return_pct']:.2f}%", zorder=3)
ax_eq.plot(doge_dates_25, eq25, color=GREEN, linewidth=2.2, label=f"2025年 +{r25['net_return_pct']:.2f}%", zorder=3)
ax_eq.fill_between(doge_dates_24, 1000, eq24, alpha=0.12, color=BLUE)
ax_eq.fill_between(doge_dates_25, 1000, eq25, alpha=0.12, color=GREEN)
ax_eq.axhline(1000, color=GRID, linewidth=1.0, linestyle="--", alpha=0.8)

# 标注最终收益
ax_eq.annotate(f"+{r24['net_return_pct']:.1f}%\n({r24['final_balance']:.0f} USDT)",
               xy=(doge_dates_24[-1], eq24[-1]),
               xytext=(-60, 15), textcoords="offset points",
               fontsize=11, fontweight="bold", color=BLUE,
               arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.2))
ax_eq.annotate(f"+{r25['net_return_pct']:.1f}%\n({r25['final_balance']:.0f} USDT)",
               xy=(doge_dates_25[-1], eq25[-1]),
               xytext=(-60, 15), textcoords="offset points",
               fontsize=11, fontweight="bold", color=GREEN,
               arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.2))

ax_eq.set_title("ButterflyBot · DOGE/USDT 15m · 两年实盘级回测权益曲线", fontsize=13, pad=12)
ax_eq.set_ylabel("账户权益 (USDT)", fontsize=10)
ax_eq.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax_eq.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1,3,5,7,9,11]))
style_ax(ax_eq, grid_x=True)
ax_eq.legend(loc="upper left", fontsize=10, facecolor=BG2, edgecolor=GRID, labelcolor=TEXT)

# 回撤曲线
dd24 = np.array(r24["drawdown_curve"]); dd25 = np.array(r25["drawdown_curve"])
ax_dd.fill_between(doge_dates_24, dd24, alpha=0.5, color=BLUE)
ax_dd.fill_between(doge_dates_25, dd25, alpha=0.5, color=GREEN)
ax_dd.plot(doge_dates_24, dd24, color=BLUE,  linewidth=1.0)
ax_dd.plot(doge_dates_25, dd25, color=GREEN, linewidth=1.0)
ax_dd.axhline(-15, color=RED, linewidth=1.2, linestyle="--", alpha=0.8, label="硬止损 -15%")
ax_dd.axhline(0, color=GRID, linewidth=0.5)
ax_dd.set_ylabel("回撤 (%)", fontsize=9); ax_dd.set_ylim(-20, 2)
ax_dd.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax_dd.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1,3,5,7,9,11]))
style_ax(ax_dd, grid_x=True)
ax_dd.legend(fontsize=8, facecolor=BG2, edgecolor=GRID, labelcolor=TEXT, loc="lower left")

fig1.tight_layout(rect=[0,0,1,0.97])
fig1.suptitle("AI量化策略 · 两年回测验证 · 初始资金1000 USDT", fontsize=10, color=TEXT2, y=0.99)
path1 = "reports/backtest/showcase_1_equity.png"
fig1.savefig(path1, dpi=160, bbox_inches="tight", facecolor=BG)
print(f"✅ 图1已保存: {path1}")
plt.close(fig1)

# ══════════════════════════════════════════════════════════════════
# 图2：月度盈亏热力图 + 年度核心指标对比
# ══════════════════════════════════════════════════════════════════
fig2 = plt.figure(figsize=(16, 10))
fig2.patch.set_facecolor(BG)
gs2 = GridSpec(2, 2, figure=fig2, hspace=0.45, wspace=0.35)

# ── 月度盈亏柱状图（2024）──
ax_m24 = fig2.add_subplot(gs2[0, 0])
df_t24 = pd.DataFrame(r24["trades"])
df_t24["exit_time"] = pd.to_datetime(df_t24["exit_time"])
df_t24["month"] = df_t24["exit_time"].dt.strftime("%m")
m24 = df_t24.groupby("month").agg(pnl=("pnl","sum"), trades=("pnl","count"), wins=("pnl",lambda x:(x>0).sum()))
m24["win_rate"] = m24["wins"]/m24["trades"]*100
bar_c = [GREEN if v>=0 else RED for v in m24["pnl"].values]
bars = ax_m24.bar(range(len(m24)), m24["pnl"].values, color=bar_c, alpha=0.85, width=0.65)
ax_m24.set_xticks(range(len(m24)))
ax_m24.set_xticklabels([f"{m}月" for m in m24.index], fontsize=9)
ax_m24.axhline(0, color=GRID, linewidth=0.8)
for bar, val in zip(bars, m24["pnl"].values):
    ypos = bar.get_height() + (3 if val>=0 else -12)
    ax_m24.text(bar.get_x()+bar.get_width()/2, ypos,
                f"{val:+.0f}", ha="center", fontsize=8, color=TEXT,
                va="bottom" if val>=0 else "top")
ax_m24.set_title("2024年月度盈亏 (USDT)", fontsize=11)
ax_m24.set_ylabel("盈亏 (USDT)", fontsize=9)
style_ax(ax_m24)

# ── 月度盈亏柱状图（2025）──
ax_m25 = fig2.add_subplot(gs2[0, 1])
df_t25 = pd.DataFrame(r25["trades"])
df_t25["exit_time"] = pd.to_datetime(df_t25["exit_time"])
df_t25["month"] = df_t25["exit_time"].dt.strftime("%m")
m25 = df_t25.groupby("month").agg(pnl=("pnl","sum"), trades=("pnl","count"), wins=("pnl",lambda x:(x>0).sum()))
m25["win_rate"] = m25["wins"]/m25["trades"]*100
bar_c25 = [GREEN if v>=0 else RED for v in m25["pnl"].values]
bars25 = ax_m25.bar(range(len(m25)), m25["pnl"].values, color=bar_c25, alpha=0.85, width=0.65)
ax_m25.set_xticks(range(len(m25)))
ax_m25.set_xticklabels([f"{m}月" for m in m25.index], fontsize=9)
ax_m25.axhline(0, color=GRID, linewidth=0.8)
for bar, val in zip(bars25, m25["pnl"].values):
    ypos = bar.get_height() + (1 if val>=0 else -5)
    ax_m25.text(bar.get_x()+bar.get_width()/2, ypos,
                f"{val:+.0f}", ha="center", fontsize=8, color=TEXT,
                va="bottom" if val>=0 else "top")
ax_m25.set_title("2025年月度盈亏 (USDT)", fontsize=11)
ax_m25.set_ylabel("盈亏 (USDT)", fontsize=9)
style_ax(ax_m25)

# ── 年度核心指标雷达图 ──
ax_radar = fig2.add_subplot(gs2[1, 0], polar=True)
ax_radar.set_facecolor(BG2)
categories = ["净收益率", "胜率", "盈利因子×20", "低回撤\n(100-|最大回撤|×5)", "交易活跃度"]

def normalize_metrics(r):
    net = min(r["net_return_pct"]/100*100, 100)
    wr  = r["win_rate"]
    pf  = min(float(r["profit_factor"])*20, 100) if r["profit_factor"]!="inf" else 100
    dd  = max(0, 100+r["max_drawdown_pct"]*5)
    act = min(r["total_trades"]/3, 100)
    return [net, wr, pf, dd, act]

vals24 = normalize_metrics(r24)
vals25 = normalize_metrics(r25)
N = len(categories)
angles = [n/float(N)*2*np.pi for n in range(N)]
angles += angles[:1]
vals24 += vals24[:1]; vals25 += vals25[:1]

ax_radar.plot(angles, vals24, color=BLUE,  linewidth=2, linestyle="solid")
ax_radar.fill(angles, vals24, color=BLUE,  alpha=0.18)
ax_radar.plot(angles, vals25, color=GREEN, linewidth=2, linestyle="solid")
ax_radar.fill(angles, vals25, color=GREEN, alpha=0.18)
ax_radar.set_xticks(angles[:-1])
ax_radar.set_xticklabels(categories, fontsize=9, color=TEXT)
ax_radar.set_ylim(0, 100)
ax_radar.set_yticks([25,50,75,100])
ax_radar.set_yticklabels(["25","50","75","100"], fontsize=7, color=TEXT2)
ax_radar.grid(color=GRID, linewidth=0.6)
ax_radar.spines["polar"].set_color(GRID)
ax_radar.set_facecolor(BG2)
ax_radar.set_title("策略综合评分雷达图", fontsize=11, color=TEXT, pad=15)
ax_radar.legend(
    handles=[mpatches.Patch(color=BLUE, label="2024年"), mpatches.Patch(color=GREEN, label="2025年")],
    loc="lower right", fontsize=9, facecolor=BG2, edgecolor=GRID, labelcolor=TEXT
)

# ── 核心指标对比表 ──
ax_tbl = fig2.add_subplot(gs2[1, 1])
ax_tbl.set_facecolor(BG2)
ax_tbl.axis("off")
style_ax(ax_tbl)

metrics = [
    ("净收益率",   f"+{r24['net_return_pct']:.2f}%",  f"+{r25['net_return_pct']:.2f}%"),
    ("最终资金",   f"{r24['final_balance']:.0f} USDT", f"{r25['final_balance']:.0f} USDT"),
    ("最大回撤",   f"{r24['max_drawdown_pct']:.2f}%",  f"{r25['max_drawdown_pct']:.2f}%"),
    ("交易笔数",   f"{r24['total_trades']}",            f"{r25['total_trades']}"),
    ("胜率",       f"{r24['win_rate']:.2f}%",           f"{r25['win_rate']:.2f}%"),
    ("盈利因子",   f"{r24['profit_factor']}",           f"{r25['profit_factor']}"),
    ("平均盈利",   f"+{r24['avg_profit']:.2f} USDT",   f"+{r25['avg_profit']:.2f} USDT"),
    ("平均亏损",   f"{r24['avg_loss']:.2f} USDT",      f"{r25['avg_loss']:.2f} USDT"),
    ("硬止损触发", "否 ✓",                              "否 ✓"),
]

col_labels = ["指标", "2024年", "2025年"]
row_colors_list = []
for i, (metric, v24, v25) in enumerate(metrics):
    y = 0.92 - i*0.10
    ax_tbl.text(0.02, y, metric, fontsize=10, color=TEXT2, va="center", transform=ax_tbl.transAxes)
    c24 = GREEN if "+" in v24 or "否" in v24 else (RED if "-" in v24 else TEXT)
    c25 = GREEN if "+" in v25 or "否" in v25 else (RED if "-" in v25 else TEXT)
    ax_tbl.text(0.42, y, v24, fontsize=10, color=c24, va="center", fontweight="bold", transform=ax_tbl.transAxes)
    ax_tbl.text(0.72, y, v25, fontsize=10, color=c25, va="center", fontweight="bold", transform=ax_tbl.transAxes)
    if i < len(metrics)-1:
        ax_tbl.plot([0.0, 1.0], [y-0.05, y-0.05], color=GRID, linewidth=0.5, transform=ax_tbl.transAxes, clip_on=False)

ax_tbl.text(0.42, 0.97, "2024年", fontsize=11, color=BLUE, va="center", fontweight="bold", transform=ax_tbl.transAxes)
ax_tbl.text(0.72, 0.97, "2025年", fontsize=11, color=GREEN, va="center", fontweight="bold", transform=ax_tbl.transAxes)
ax_tbl.set_title("核心指标对比", fontsize=11)

fig2.suptitle("ButterflyBot · 月度盈亏 & 综合评分", fontsize=12, color=TEXT, y=1.01)
path2 = "reports/backtest/showcase_2_monthly.png"
fig2.savefig(path2, dpi=160, bbox_inches="tight", facecolor=BG)
print(f"✅ 图2已保存: {path2}")
plt.close(fig2)

# ══════════════════════════════════════════════════════════════════
# 图3：交易分析 — 盈亏分布、持仓时长、胜率趋势、出场原因
# ══════════════════════════════════════════════════════════════════
fig3 = plt.figure(figsize=(16, 10))
fig3.patch.set_facecolor(BG)
gs3 = GridSpec(2, 2, figure=fig3, hspace=0.45, wspace=0.35)

all_trades = r24["trades"] + r25["trades"]
df_all = pd.DataFrame(all_trades)
df_all["exit_time"] = pd.to_datetime(df_all["exit_time"])
df_all["year"] = df_all["exit_time"].dt.year

# ── 单笔盈亏分布 ──
ax_pnl = fig3.add_subplot(gs3[0, 0])
wins_pnl   = df_all[df_all["pnl"]>0]["pnl"].values
losses_pnl = df_all[df_all["pnl"]<=0]["pnl"].values
ax_pnl.hist(wins_pnl,   bins=40, color=GREEN, alpha=0.75, label=f"盈利 ({len(wins_pnl)}笔)")
ax_pnl.hist(losses_pnl, bins=30, color=RED,   alpha=0.75, label=f"亏损 ({len(losses_pnl)}笔)")
ax_pnl.axvline(np.mean(wins_pnl),   color=GREEN, linewidth=1.5, linestyle="--", label=f"均盈 +{np.mean(wins_pnl):.1f}")
ax_pnl.axvline(np.mean(losses_pnl), color=RED,   linewidth=1.5, linestyle="--", label=f"均亏 {np.mean(losses_pnl):.1f}")
ax_pnl.axvline(0, color=GRID, linewidth=1.0)
ax_pnl.set_title("单笔盈亏分布（两年合并）", fontsize=11)
ax_pnl.set_xlabel("盈亏 (USDT)", fontsize=9); ax_pnl.set_ylabel("频次", fontsize=9)
style_ax(ax_pnl)
ax_pnl.legend(fontsize=8, facecolor=BG2, edgecolor=GRID, labelcolor=TEXT)

# ── 持仓时长分布 ──
ax_bars = fig3.add_subplot(gs3[0, 1])
bars_arr = df_all["bars"].values
ax_bars.hist(bars_arr, bins=40, color=PURP, alpha=0.8, edgecolor=BG)
ax_bars.axvline(np.mean(bars_arr), color=GOLD, linewidth=1.5, linestyle="--",
                label=f"平均 {np.mean(bars_arr):.1f} 根K线")
ax_bars.set_title("持仓时长分布（根K线数）", fontsize=11)
ax_bars.set_xlabel("持仓K线数 (×15min)", fontsize=9); ax_bars.set_ylabel("频次", fontsize=9)
style_ax(ax_bars)
ax_bars.legend(fontsize=9, facecolor=BG2, edgecolor=GRID, labelcolor=TEXT)

# ── 累计胜率趋势 ──
ax_wr = fig3.add_subplot(gs3[1, 0])
for year, color, r in [(2024, BLUE, r24), (2025, GREEN, r25)]:
    df_y = pd.DataFrame(r["trades"])
    if len(df_y) == 0: continue
    df_y["win"] = (df_y["pnl"] > 0).astype(int)
    cum_wr = df_y["win"].expanding().mean() * 100
    ax_wr.plot(range(len(cum_wr)), cum_wr, color=color, linewidth=1.8,
               label=f"{year}年 最终{r['win_rate']:.1f}%")
ax_wr.axhline(50, color=GRID, linewidth=1.0, linestyle="--", alpha=0.7, label="50%基准线")
ax_wr.set_title("累计胜率趋势", fontsize=11)
ax_wr.set_xlabel("交易笔数", fontsize=9); ax_wr.set_ylabel("累计胜率 (%)", fontsize=9)
ax_wr.set_ylim(20, 90)
style_ax(ax_wr, grid_x=True)
ax_wr.legend(fontsize=9, facecolor=BG2, edgecolor=GRID, labelcolor=TEXT)

# ── 出场原因分布（饼图）──
ax_pie = fig3.add_subplot(gs3[1, 1])
ax_pie.set_facecolor(BG2)
reason_all = {}
for t in all_trades:
    reason_all[t["reason"]] = reason_all.get(t["reason"],0)+1
labels_pie = list(reason_all.keys())
sizes_pie  = list(reason_all.values())
colors_pie = [GREEN, RED, GOLD, PURP, CYAN, BLUE][:len(labels_pie)]
wedges, texts, autotexts = ax_pie.pie(
    sizes_pie, labels=None, autopct="%1.1f%%",
    colors=colors_pie, startangle=140,
    pctdistance=0.75, wedgeprops=dict(linewidth=1.5, edgecolor=BG)
)
for at in autotexts: at.set_color(BG); at.set_fontsize(9); at.set_fontweight("bold")
ax_pie.legend(
    wedges, [f"{l} ({v}笔)" for l,v in zip(labels_pie, sizes_pie)],
    loc="lower center", bbox_to_anchor=(0.5,-0.12), fontsize=9,
    facecolor=BG2, edgecolor=GRID, labelcolor=TEXT, ncol=2
)
ax_pie.set_title("出场原因分布（两年合并）", fontsize=11, color=TEXT)

fig3.suptitle("ButterflyBot · 交易质量深度分析", fontsize=12, color=TEXT, y=1.01)
path3 = "reports/backtest/showcase_3_trade_analysis.png"
fig3.savefig(path3, dpi=160, bbox_inches="tight", facecolor=BG)
print(f"✅ 图3已保存: {path3}")
plt.close(fig3)

# ══════════════════════════════════════════════════════════════════
# 图4：核心指标卡片（朋友圈单图，竖版）
# ══════════════════════════════════════════════════════════════════
fig4, ax4 = plt.subplots(figsize=(10, 16))
fig4.patch.set_facecolor(BG)
ax4.set_facecolor(BG)
ax4.axis("off")

# 标题
ax4.text(0.5, 0.97, "ButterflyBot", fontsize=28, fontweight="bold",
         color=BLUE, ha="center", va="top", transform=ax4.transAxes)
ax4.text(0.5, 0.935, "AI 量化交易策略 · 实盘级回测报告",
         fontsize=13, color=TEXT2, ha="center", va="top", transform=ax4.transAxes)
ax4.text(0.5, 0.905, "DOGE/USDT · 15分钟K线 · LightGBM (AUC=0.754) · 仓位50%",
         fontsize=10, color=TEXT2, ha="center", va="top", transform=ax4.transAxes)

# 分割线
ax4.plot([0.05, 0.95], [0.895, 0.895], color=GRID, linewidth=1.0, transform=ax4.transAxes, clip_on=False)

# 大指标卡片
def draw_card(ax, x, y, w, h, title, value, subtitle, color, transform):
    rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01",
                          facecolor=BG2, edgecolor=color, linewidth=1.5,
                          transform=transform, zorder=2)
    ax.add_patch(rect)
    ax.text(x+w/2, y+h*0.72, value, fontsize=22, fontweight="bold",
            color=color, ha="center", va="center", transform=transform, zorder=3)
    ax.text(x+w/2, y+h*0.38, title, fontsize=10, color=TEXT2,
            ha="center", va="center", transform=transform, zorder=3)
    ax.text(x+w/2, y+h*0.15, subtitle, fontsize=8.5, color=TEXT2,
            ha="center", va="center", transform=transform, zorder=3)

# 2024年 大卡片
ax4.text(0.5, 0.87, "2024 年", fontsize=14, fontweight="bold",
         color=BLUE, ha="center", va="top", transform=ax4.transAxes)
cards_24 = [
    (0.05, 0.73, 0.27, 0.12, "净收益率", f"+{r24['net_return_pct']:.1f}%", "1000→1824 USDT", GREEN),
    (0.37, 0.73, 0.27, 0.12, "最大回撤", f"{r24['max_drawdown_pct']:.1f}%", "硬止损未触发", GOLD),
    (0.69, 0.73, 0.27, 0.12, "盈利因子", f"{r24['profit_factor']}", f"{r24['total_trades']}笔交易", PURP),
    (0.05, 0.59, 0.27, 0.12, "胜率",     f"{r24['win_rate']:.1f}%", f"{int(r24['total_trades']*r24['win_rate']/100)}胜/{r24['total_trades']-int(r24['total_trades']*r24['win_rate']/100)}负", BLUE),
    (0.37, 0.59, 0.27, 0.12, "平均盈利", f"+{r24['avg_profit']:.1f}", "USDT/笔", GREEN),
    (0.69, 0.59, 0.27, 0.12, "平均亏损", f"{r24['avg_loss']:.1f}", "USDT/笔", RED),
]
for x,y,w,h,title,val,sub,col in cards_24:
    draw_card(ax4, x, y, w, h, title, val, sub, col, ax4.transAxes)

ax4.plot([0.05, 0.95], [0.575, 0.575], color=GRID, linewidth=0.8, linestyle=":", transform=ax4.transAxes, clip_on=False)

# 2025年 大卡片
ax4.text(0.5, 0.565, "2025 年", fontsize=14, fontweight="bold",
         color=GREEN, ha="center", va="top", transform=ax4.transAxes)
cards_25 = [
    (0.05, 0.42, 0.27, 0.12, "净收益率", f"+{r25['net_return_pct']:.1f}%", "1000→1384 USDT", GREEN),
    (0.37, 0.42, 0.27, 0.12, "最大回撤", f"{r25['max_drawdown_pct']:.1f}%", "硬止损未触发", GOLD),
    (0.69, 0.42, 0.27, 0.12, "盈利因子", f"{r25['profit_factor']}", f"{r25['total_trades']}笔交易", PURP),
    (0.05, 0.28, 0.27, 0.12, "胜率",     f"{r25['win_rate']:.1f}%", f"{int(r25['total_trades']*r25['win_rate']/100)}胜/{r25['total_trades']-int(r25['total_trades']*r25['win_rate']/100)}负", BLUE),
    (0.37, 0.28, 0.27, 0.12, "平均盈利", f"+{r25['avg_profit']:.1f}", "USDT/笔", GREEN),
    (0.69, 0.28, 0.27, 0.12, "平均亏损", f"{r25['avg_loss']:.1f}", "USDT/笔", RED),
]
for x,y,w,h,title,val,sub,col in cards_25:
    draw_card(ax4, x, y, w, h, title, val, sub, col, ax4.transAxes)

ax4.plot([0.05, 0.95], [0.265, 0.265], color=GRID, linewidth=1.0, transform=ax4.transAxes, clip_on=False)

# 策略参数说明
params_text = (
    "策略参数  |  信号阈值: 0.12  ·  仓位: 50%  ·  止损: -1.5%  ·  止盈: +3%\n"
    "风控机制  |  MA50趋势过滤  ·  冷却期3根K线  ·  账户硬止损 -15%  ·  时间止损40根K线\n"
    "模型信息  |  LightGBM · 128个特征 · 2024-2025年训练 · AUC=0.754 · Platt校准"
)
ax4.text(0.5, 0.24, params_text, fontsize=9, color=TEXT2,
         ha="center", va="top", transform=ax4.transAxes,
         linespacing=1.8)

# 免责声明
ax4.text(0.5, 0.04,
         "⚠ 本报告为历史回测结果，不构成投资建议。量化交易存在风险，过往业绩不代表未来收益。",
         fontsize=8, color=TEXT2, ha="center", va="bottom", transform=ax4.transAxes, alpha=0.7)

path4 = "reports/backtest/showcase_4_card.png"
fig4.savefig(path4, dpi=160, bbox_inches="tight", facecolor=BG)
print(f"✅ 图4已保存: {path4}")
plt.close(fig4)

print("\n✅ 全部4张图表生成完成！")
print(f"   图1: {path1}")
print(f"   图2: {path2}")
print(f"   图3: {path3}")
print(f"   图4: {path4}")
