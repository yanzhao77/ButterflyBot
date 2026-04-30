"""
ButterflyBot 2026年最新数据专项回测
使用最优配置: 阈值0.12, 仓位50%, 止损-1.5%, 止盈+3%, 硬止损-15%
模型: v20260429_112717 (AUC=0.754, 128特征)
数据: 2026-01-01 ~ 2026-04-30
"""
import sys, os, json, logging, pickle
sys.path.insert(0, ".")
import numpy as np
import pandas as pd
import joblib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings; warnings.filterwarnings("ignore")
from datetime import datetime

plt.rcParams["font.family"]        = "sans-serif"
plt.rcParams["font.sans-serif"]    = ["Noto Sans CJK SC", "WenQuanYi Micro Hei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

logging.basicConfig(level=logging.WARNING)
from butterfly_bot.model.model_registry import REGISTRY_DIR
os.makedirs("reports/backtest", exist_ok=True)

VERSION       = "v20260429_112717"
INITIAL_BAL   = 1000.0
POSITION_PCT  = 0.50
FEE_RATE      = 0.001
SL_LONG       = -0.015
TP_LONG       = 0.030
LONG_TH       = 0.12
MAX_HOLD_BARS = 40
COOLDOWN_BARS = 3
HARD_STOP_PCT = 0.15

print("=" * 70)
print(f"🚀 ButterflyBot 2026年最新数据回测")
print(f"   模型: {VERSION} | 仓位: {POSITION_PCT*100:.0f}% | 阈值: {LONG_TH}")
print(f"   止损: {SL_LONG*100:.1f}% | 止盈: {TP_LONG*100:.1f}% | 硬止损: -{HARD_STOP_PCT*100:.0f}%")
print("=" * 70)

# ===================== 1. 加载2026年数据（含预热数据） =====================
print("\n📥 加载数据...")
# 加载2025年末数据作为预热（计算MA50/特征需要历史数据）
df_hist = pd.read_csv(
    "butterfly_bot/cached_data/binance_DOGE_USDT_15m_5year.csv",
    index_col=0, parse_dates=True
)
df_hist.index = pd.to_datetime(df_hist.index, utc=True)
df_warmup = df_hist.tail(500).copy()  # 取500根K线作为预热

# 加载2026年数据
df_2026 = pd.read_csv(
    "butterfly_bot/cached_data/binance_DOGE_USDT_15m_2026.csv",
    index_col=0, parse_dates=True
)
df_2026.index = pd.to_datetime(df_2026.index, utc=True)

# 合并
df_raw = pd.concat([df_warmup, df_2026]).drop_duplicates().sort_index()
print(f"✅ 合并数据: {len(df_raw)} 根K线 ({df_raw.index[0]} ~ {df_raw.index[-1]})")
print(f"   2026年数据: {len(df_2026)} 根K线 ({df_2026.index[0]} ~ {df_2026.index[-1]})")

# ===================== 2. 增强版特征工程（与训练时完全一致） =====================
def compute_rsi(series, window=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
    return 100 - (100 / (1 + gain / (loss + 1e-8)))

def compute_atr(df, window=14):
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift()).abs()
    lc = (df["low"] - df["close"].shift()).abs()
    return pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(window).mean()

def compute_stoch(df, window=14):
    lo = df["low"].rolling(window).min()
    hi = df["high"].rolling(window).max()
    return 100 * (df["close"] - lo) / (hi - lo + 1e-8)

def compute_adx(df, window=14):
    hi_diff = df["high"].diff(); lo_diff = -df["low"].diff()
    plus_dm  = hi_diff.where((hi_diff > lo_diff) & (hi_diff > 0), 0)
    minus_dm = lo_diff.where((lo_diff > hi_diff) & (lo_diff > 0), 0)
    atr = compute_atr(df, window)
    plus_di  = 100 * plus_dm.rolling(window).mean() / (atr + 1e-8)
    minus_di = 100 * minus_dm.rolling(window).mean() / (atr + 1e-8)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-8)
    return dx.rolling(window).mean(), plus_di, minus_di

def compute_obv(df):
    return (np.sign(df["close"].diff()) * df["volume"]).fillna(0).cumsum()

def add_enhanced_features(df):
    df = df.copy()
    c = df["close"]; o = df["open"]; h = df["high"]; l = df["low"]; v = df["volume"]
    # 基础收益率
    df["return"]     = c.pct_change()
    df["log_return"] = np.log(c / c.shift(1))
    df["return_3"]   = c.pct_change(3)
    df["return_5"]   = c.pct_change(5)
    df["return_10"]  = c.pct_change(10)
    df["return_20"]  = c.pct_change(20)
    df["return_30"]  = c.pct_change(30)
    # 多周期移动平均
    for w in [5, 10, 20, 50, 100, 200]:
        df[f"ma{w}"] = c.rolling(w).mean()
    for span in [9, 12, 21, 26, 55]:
        df[f"ema{span}"] = c.ewm(span=span).mean()
    df["ma_diff_short"]  = df["ma5"] - df["ma20"]
    df["ma_diff_long"]   = df["ma20"] - df["ma50"]
    df["ma_diff_200"]    = df["ma50"] - df["ma200"]
    df["price_to_ma20"]  = (c - df["ma20"]) / (df["ma20"] + 1e-8)
    df["price_to_ma50"]  = (c - df["ma50"]) / (df["ma50"] + 1e-8)
    df["price_to_ma100"] = (c - df["ma100"]) / (df["ma100"] + 1e-8)
    df["price_to_ma200"] = (c - df["ma200"]) / (df["ma200"] + 1e-8)
    df["ema_cross"]      = df["ema12"] - df["ema26"]
    df["ema_cross_9_21"] = df["ema9"] - df["ema21"]
    # 多周期RSI
    for w in [6, 9, 14, 21, 28]:
        df[f"rsi_{w}"] = compute_rsi(c, w)
    df["rsi_diff"]  = df["rsi_14"] - df["rsi_14"].shift(3)
    df["rsi_cross"] = df["rsi_6"] - df["rsi_14"]
    # MACD
    df["macd"]         = df["ema12"] - df["ema26"]
    df["macd_signal"]  = df["macd"].ewm(span=9).mean()
    df["macd_hist"]    = df["macd"] - df["macd_signal"]
    df["macd_hist_change"] = df["macd_hist"] - df["macd_hist"].shift(1)
    fast_macd = c.ewm(span=5).mean() - c.ewm(span=13).mean()
    df["fast_macd_hist"] = fast_macd - fast_macd.ewm(span=4).mean()
    # 布林带（多周期）
    for w, std_mult in [(10, 2), (20, 2), (20, 1.5), (50, 2)]:
        mid = c.rolling(w).mean(); s = c.rolling(w).std()
        up  = mid + std_mult * s; lo_ = mid - std_mult * s
        tag = f"bb{w}_{str(std_mult).replace('.','')}"
        df[f"{tag}_width"]    = (up - lo_) / (mid + 1e-8)
        df[f"{tag}_position"] = (c - lo_) / (up - lo_ + 1e-8)
    df["bb_upper"]  = df["ma20"] + 2 * c.rolling(20).std()
    df["bb_lower"]  = df["ma20"] - 2 * c.rolling(20).std()
    df["bb_middle"] = df["ma20"]
    # ATR
    df["atr"]       = compute_atr(df, 14)
    df["atr_7"]     = compute_atr(df, 7)
    df["atr_ratio"] = df["atr"] / (c + 1e-8)
    df["atr_change"]= df["atr"] / (df["atr"].shift(5) + 1e-8)
    # 波动率
    for w in [5, 10, 20, 50]:
        df[f"volatility_{w}"] = df["log_return"].rolling(w).std()
    df["vol_ratio"]  = df["volatility_10"] / (df["volatility_50"] + 1e-8)
    df["vol_change"] = df["volatility_10"] / (df["volatility_10"].shift(10) + 1e-8)
    # Stochastic
    df["stoch_k"]    = compute_stoch(df, 14); df["stoch_d"] = df["stoch_k"].rolling(3).mean()
    df["stoch_k9"]   = compute_stoch(df, 9);  df["stoch_cross"] = df["stoch_k"] - df["stoch_d"]
    # Williams %R
    for w in [14, 28]:
        hi_max = h.rolling(w).max(); lo_min = l.rolling(w).min()
        df[f"williams_r_{w}"] = -100 * (hi_max - c) / (hi_max - lo_min + 1e-8)
    # ROC
    for w in [5, 10, 20, 30]: df[f"roc_{w}"] = ((c - c.shift(w)) / (c.shift(w) + 1e-8)) * 100
    # ADX
    df["adx"], df["plus_di"], df["minus_di"] = compute_adx(df, 14)
    df["di_diff"] = df["plus_di"] - df["minus_di"]
    # 成交量特征
    for w in [5, 10, 20, 50]: df[f"volume_ma{w}"] = v.rolling(w).mean()
    df["volume_ratio"]    = v / (df["volume_ma20"] + 1e-8)
    df["volume_ratio_5"]  = v / (df["volume_ma5"]  + 1e-8)
    df["volume_change"]   = v.pct_change()
    df["volume_change_5"] = v.pct_change(5)
    df["obv"]      = compute_obv(df)
    df["obv_ma"]   = df["obv"].rolling(20).mean()
    df["obv_ratio"]= df["obv"] / (df["obv_ma"].abs() + 1e-8)
    df["vwap_ratio"]= (c*v).rolling(20).sum() / (v.rolling(20).sum() + 1e-8) / (c + 1e-8)
    # 蜡烛图形态
    body = (c - o).abs()
    upper_wick = h - pd.concat([c,o], axis=1).max(axis=1)
    lower_wick = pd.concat([c,o], axis=1).min(axis=1) - l
    total_range = h - l + 1e-8
    df["body_ratio"]       = body / total_range
    df["upper_wick_ratio"] = upper_wick / total_range
    df["lower_wick_ratio"] = lower_wick / total_range
    df["close_position"]   = (c - l) / total_range
    df["high_low_ratio"]   = total_range / (c + 1e-8)
    df["is_bullish"]       = (c > o).astype(int)
    df["hammer_score"]     = (lower_wick / total_range) * (1 - body / total_range)
    df["shooting_star"]    = (upper_wick / total_range) * (1 - body / total_range)
    df["consecutive_bull"] = (c > o).astype(int).rolling(3).sum()
    df["consecutive_bear"] = (c < o).astype(int).rolling(3).sum()
    # 动量
    for w in [3, 5, 10, 20]: df[f"momentum_{w}"] = c - c.shift(w)
    df["accel_5"]  = df["momentum_5"]  - df["momentum_5"].shift(5)
    df["accel_10"] = df["momentum_10"] - df["momentum_10"].shift(10)
    # 时间特征
    df["hour"]        = df.index.hour
    df["day_of_week"] = df.index.dayofweek
    df["hour_sin"]    = np.sin(2*np.pi*df.index.hour/24)
    df["hour_cos"]    = np.cos(2*np.pi*df.index.hour/24)
    df["dow_sin"]     = np.sin(2*np.pi*df.index.dayofweek/7)
    df["dow_cos"]     = np.cos(2*np.pi*df.index.dayofweek/7)
    # 高低点距离
    for w in [10, 20, 50, 100]:
        df[f"high_{w}"]      = h.rolling(w).max()
        df[f"low_{w}"]       = l.rolling(w).min()
        df[f"dist_high_{w}"] = (df[f"high_{w}"] - c) / (c + 1e-8)
        df[f"dist_low_{w}"]  = (c - df[f"low_{w}"]) / (c + 1e-8)
    # 交叉特征
    df["rsi_vol_cross"]  = df["rsi_14"] * df["volume_ratio"]
    df["macd_atr_cross"] = df["macd_hist"] / (df["atr"] + 1e-8)
    df["bb_rsi_cross"]   = df["bb10_2_position"] * df["rsi_14"] / 100
    df["vol_momentum"]   = df["volume_ratio"] * df["return_5"]
    df.dropna(inplace=True)
    return df

print("\n🔧 构建增强版特征...")
df_feat_all = add_enhanced_features(df_raw)
# 只保留2026年数据（去掉预热部分）
start_2026 = pd.Timestamp("2026-01-01", tz="UTC")
df_feat_2026 = df_feat_all[df_feat_all.index >= start_2026].copy()

with open(os.path.join(REGISTRY_DIR, f"{VERSION}_features.json")) as f:
    feature_cols = json.load(f)
available_cols = [c for c in feature_cols if c in df_feat_2026.columns]
missing_cols   = [c for c in feature_cols if c not in df_feat_2026.columns]
print(f"✅ 特征: {len(available_cols)} 个（缺失: {len(missing_cols)}），2026年样本: {len(df_feat_2026)} 行")
if missing_cols:
    print(f"   缺失特征: {missing_cols[:5]}...")

# ===================== 3. 模型预测 =====================
print("\n🤖 生成预测概率...")
model_long = joblib.load(os.path.join(REGISTRY_DIR, f"{VERSION}_long.pkl"))
cal_long   = joblib.load(os.path.join(REGISTRY_DIR, f"{VERSION}_cal_long.pkl"))
X_2026     = df_feat_2026[available_cols].fillna(0).values
raw_long   = model_long.predict(X_2026)
cal_probs  = cal_long.predict_proba(raw_long.reshape(-1,1))[:,1]

# 对齐到原始2026年索引
p_long_series = pd.Series(np.nan, index=df_2026.index)
p_long_series.loc[df_feat_2026.index] = cal_probs

# EMA平滑
alpha = 2.0 / (10 + 1.0)
def ema_smooth(vals):
    ema_val = None; result = []
    for v in vals:
        if not np.isnan(v):
            ema_val = v if ema_val is None else alpha*v + (1-alpha)*ema_val
        result.append(ema_val if ema_val is not None else 0.03)
    return np.array(result)

pema_2026 = ema_smooth(p_long_series.values)
print(f"✅ p_ema 统计: min={pema_2026.min():.4f}, max={pema_2026.max():.4f}, "
      f"mean={pema_2026.mean():.4f}, ≥0.12: {(pema_2026>=0.12).sum()}")

# ===================== 4. 回测引擎 =====================
def run_backtest(df, pema, label, long_th=LONG_TH, sl=SL_LONG, tp=TP_LONG, pos_pct=POSITION_PCT):
    balance = INITIAL_BAL; peak_balance = INITIAL_BAL
    position = 0.0; entry_price = 0.0; entry_bar = 0
    cooldown = 0; hard_stopped = False
    trades = []; equity_curve = []; equity_dates = []
    closes = df["close"].values
    ma50 = df["close"].rolling(50).mean().values
    for i in range(len(df)):
        price = closes[i]; pl = pema[i]
        cur_equity = balance + (price - entry_price)*position if position > 0 else balance
        equity_curve.append(cur_equity); equity_dates.append(df.index[i])
        if not hard_stopped:
            if cur_equity > peak_balance: peak_balance = cur_equity
            if (peak_balance - cur_equity) / peak_balance >= HARD_STOP_PCT:
                hard_stopped = True
                if position > 0:
                    pnl = (price-entry_price)*position - (entry_price*position+price*position)*FEE_RATE
                    balance += pnl
                    trades.append({"entry":entry_price,"exit":price,"pnl":pnl,
                                   "bars":i-entry_bar,"reason":"硬止损",
                                   "entry_time":str(df.index[entry_bar]),"exit_time":str(df.index[i]),
                                   "direction":"long"})
                    position = 0
                continue
        if hard_stopped: continue
        if cooldown > 0: cooldown -= 1
        if position > 0:
            profit_pct = (price - entry_price) / entry_price
            holding_bars = i - entry_bar; reason = None
            if profit_pct <= sl:                           reason = "固定止损"
            elif profit_pct >= tp:                         reason = "止盈"
            elif holding_bars >= MAX_HOLD_BARS:            reason = "时间止损"
            elif pl <= max(0.02, long_th - 0.04):          reason = "AI看跌"
            if reason:
                pnl = (price-entry_price)*position - (entry_price*position+price*position)*FEE_RATE
                balance += pnl
                trades.append({"entry":entry_price,"exit":price,"pnl":pnl,
                               "bars":holding_bars,"reason":reason,
                               "entry_time":str(df.index[entry_bar]),"exit_time":str(df.index[i]),
                               "direction":"long"})
                position = 0; cooldown = COOLDOWN_BARS; continue
        if position == 0 and cooldown == 0:
            ma50_val = ma50[i]
            if pl >= long_th and not np.isnan(ma50_val) and price >= ma50_val:
                invest = balance * pos_pct; fee = invest * FEE_RATE
                position = (invest - fee) / price; entry_price = price; entry_bar = i
    if position > 0:
        price = closes[-1]
        pnl = (price-entry_price)*position - (entry_price*position+price*position)*FEE_RATE
        balance += pnl
        trades.append({"entry":entry_price,"exit":price,"pnl":pnl,
                       "bars":len(df)-1-entry_bar,"reason":"回测结束",
                       "entry_time":str(df.index[entry_bar]),"exit_time":str(df.index[-1]),
                       "direction":"long"})
    wins   = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    total  = len(trades)
    win_rate   = len(wins)/total*100 if total > 0 else 0
    tot_profit = sum(t["pnl"] for t in wins)
    tot_loss   = abs(sum(t["pnl"] for t in losses))
    pf = tot_profit/tot_loss if tot_loss > 0 else float("inf")
    net_return = (balance - INITIAL_BAL) / INITIAL_BAL * 100
    eq = np.array(equity_curve); peak_eq = np.maximum.accumulate(eq)
    dd_arr = (eq - peak_eq) / peak_eq * 100
    reason_counts = {}
    for t in trades: reason_counts[t["reason"]] = reason_counts.get(t["reason"],0)+1
    return {
        "label": label, "final_balance": round(balance,2),
        "net_return_pct": round(net_return,2), "max_drawdown_pct": round(dd_arr.min(),2),
        "total_trades": total, "win_rate": round(win_rate,2),
        "profit_factor": round(pf,3) if pf != float("inf") else "inf",
        "avg_profit": round(tot_profit/len(wins),2) if wins else 0,
        "avg_loss":   round(-tot_loss/len(losses),2) if losses else 0,
        "reason_counts": reason_counts, "hard_stopped": hard_stopped,
        "equity_curve": equity_curve, "equity_dates": equity_dates,
        "drawdown_curve": dd_arr.tolist(), "trades": trades,
    }

# ===================== 5. 运行多组配置对比 =====================
print("\n📊 运行回测...")
configs = [
    {"label": "最优配置 阈值0.12 止损-1.5% 止盈+3%", "long_th": 0.12, "sl": -0.015, "tp": 0.030},
    {"label": "宽松阈值 阈值0.10 止损-1.5% 止盈+3%", "long_th": 0.10, "sl": -0.015, "tp": 0.030},
    {"label": "收紧止损 阈值0.12 止损-1.0% 止盈+2%", "long_th": 0.12, "sl": -0.010, "tp": 0.020},
]
results_2026 = []
for cfg in configs:
    r = run_backtest(df_2026, pema_2026, cfg["label"],
                     long_th=cfg["long_th"], sl=cfg["sl"], tp=cfg["tp"])
    results_2026.append(r)
    hs = "⚠️ 硬止损" if r["hard_stopped"] else "✅"
    print(f"  {hs} [{r['label'][:20]}] 净收益={r['net_return_pct']:+.2f}%, "
          f"回撤={r['max_drawdown_pct']:.2f}%, 交易={r['total_trades']}笔, "
          f"胜率={r['win_rate']:.1f}%, PF={r['profit_factor']}")

# ===================== 6. 月度分析 =====================
r_main = results_2026[0]
if r_main["trades"]:
    df_trades = pd.DataFrame(r_main["trades"])
    df_trades["exit_time"] = pd.to_datetime(df_trades["exit_time"])
    df_trades["month"] = df_trades["exit_time"].dt.strftime("%Y-%m")
    monthly = df_trades.groupby("month").agg(
        pnl=("pnl","sum"), trades=("pnl","count"),
        wins=("pnl", lambda x:(x>0).sum())
    )
    monthly["win_rate"] = monthly["wins"]/monthly["trades"]*100
    print("\n月度盈亏（最优配置）:")
    for m, row in monthly.iterrows():
        print(f"  {m}: {row['pnl']:+.2f} USDT, {int(row['trades'])}笔, 胜率{row['win_rate']:.0f}%")

# ===================== 7. 生成图表 =====================
BG="#0d1117"; BG2="#161b22"; GRID="#30363d"; TEXT="#e6edf3"; TEXT2="#8b949e"
GREEN="#3fb950"; RED="#f85149"; BLUE="#58a6ff"; GOLD="#d29922"; PURP="#bc8cff"
COLORS=[BLUE, GREEN, PURP]

def style_ax(ax, grid_x=False):
    ax.set_facecolor(BG2)
    ax.tick_params(colors=TEXT2, labelsize=9)
    for sp in ax.spines.values(): sp.set_color(GRID)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.5, alpha=0.6)
    if grid_x: ax.grid(axis="x", color=GRID, linewidth=0.3, alpha=0.4)
    ax.xaxis.label.set_color(TEXT2); ax.yaxis.label.set_color(TEXT2)
    ax.title.set_color(TEXT)

fig, axes = plt.subplots(3, 1, figsize=(16, 14),
                          gridspec_kw={"height_ratios":[3,1,1]})
fig.patch.set_facecolor(BG)
ax_eq  = axes[0]; ax_dd = axes[1]; ax_sig = axes[2]
ax_doge = ax_eq.twinx()

# DOGE价格背景
close_arr = df_2026["close"].values
dates_arr  = df_2026.index
ax_doge.fill_between(dates_arr, close_arr, alpha=0.06, color=GOLD)
ax_doge.plot(dates_arr, close_arr, color=GOLD, linewidth=0.8, alpha=0.4)
ax_doge.set_ylabel("DOGE 价格 (USDT)", color=GOLD, fontsize=9)
ax_doge.tick_params(colors=GOLD, labelsize=8)
ax_doge.spines["right"].set_color(GOLD)
for sp in ["top","left","bottom"]: ax_doge.spines[sp].set_visible(False)
ax_doge.set_facecolor(BG2)

# 权益曲线
for i, r in enumerate(results_2026):
    eq = np.array(r["equity_curve"])
    dates_eq = pd.to_datetime(r["equity_dates"])
    n = min(len(eq), len(dates_eq))
    eq, dates_eq = eq[:n], dates_eq[:n]
    lbl = f"{r['label'][:18]}  {r['net_return_pct']:+.1f}%"
    ax_eq.plot(dates_eq, eq, color=COLORS[i], linewidth=2.0, label=lbl, zorder=3)
    if i == 0:
        ax_eq.fill_between(dates_eq, INITIAL_BAL, eq, alpha=0.10, color=COLORS[i])
        # 标注最终收益
        ax_eq.annotate(f"{r['net_return_pct']:+.1f}%\n({r['final_balance']:.0f} USDT)",
                       xy=(dates_eq[-1], eq[-1]),
                       xytext=(-80, 10), textcoords="offset points",
                       fontsize=10, fontweight="bold", color=COLORS[i],
                       arrowprops=dict(arrowstyle="->", color=COLORS[i], lw=1.2))

ax_eq.axhline(INITIAL_BAL, color=GRID, linewidth=1.0, linestyle="--", alpha=0.8)
ax_eq.set_title("ButterflyBot · 2026年最新数据回测 · 权益曲线对比", fontsize=13, pad=12)
ax_eq.set_ylabel("账户权益 (USDT)", fontsize=10)
ax_eq.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax_eq.xaxis.set_major_locator(mdates.MonthLocator())
style_ax(ax_eq, grid_x=True)
ax_eq.legend(loc="upper left", fontsize=9, facecolor=BG2, edgecolor=GRID, labelcolor=TEXT)

# 回撤曲线
for i, r in enumerate(results_2026):
    dd = np.array(r["drawdown_curve"])
    dates_eq = pd.to_datetime(r["equity_dates"])
    n = min(len(dd), len(dates_eq))
    ax_dd.fill_between(dates_eq[:n], dd[:n], alpha=0.35, color=COLORS[i])
    ax_dd.plot(dates_eq[:n], dd[:n], color=COLORS[i], linewidth=1.0)
ax_dd.axhline(-15, color=RED, linewidth=1.2, linestyle="--", alpha=0.8, label="硬止损 -15%")
ax_dd.axhline(0, color=GRID, linewidth=0.5)
ax_dd.set_ylabel("回撤 (%)", fontsize=9); ax_dd.set_ylim(-20, 2)
ax_dd.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax_dd.xaxis.set_major_locator(mdates.MonthLocator())
style_ax(ax_dd, grid_x=True)
ax_dd.legend(fontsize=8, facecolor=BG2, edgecolor=GRID, labelcolor=TEXT, loc="lower left")

# p_ema信号
ax_sig.plot(dates_arr, pema_2026, color=PURP, linewidth=0.8, alpha=0.7, label="p_ema (多头)")
ax_sig.axhline(0.12, color=GREEN, linewidth=1.2, linestyle="--", alpha=0.8, label="阈值 0.12")
ax_sig.axhline(0.10, color=GOLD,  linewidth=0.8, linestyle=":",  alpha=0.6, label="阈值 0.10")
ax_sig.set_ylabel("信号强度", fontsize=9)
ax_sig.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax_sig.xaxis.set_major_locator(mdates.MonthLocator())
style_ax(ax_sig, grid_x=True)
ax_sig.legend(fontsize=8, facecolor=BG2, edgecolor=GRID, labelcolor=TEXT, loc="upper right")

# 标注交易点（最优配置）
for t in r_main["trades"]:
    try:
        etime = pd.Timestamp(t["exit_time"])
        color = GREEN if t["pnl"] > 0 else RED
        eq_arr = np.array(r_main["equity_curve"])
        dates_eq = pd.to_datetime(r_main["equity_dates"])
        idx = np.searchsorted(dates_eq, etime)
        if 0 <= idx < len(eq_arr):
            ax_eq.scatter(etime, eq_arr[min(idx, len(eq_arr)-1)],
                         color=color, s=25, zorder=5, alpha=0.8)
    except: pass

fig.tight_layout(rect=[0,0,1,0.97])
fig.suptitle(f"ButterflyBot · 2026年回测 · 2026-01-01 ~ 2026-04-30 · {len(df_2026)}根K线",
             fontsize=10, color=TEXT2, y=0.99)
path_chart = "reports/backtest/backtest_2026_chart.png"
fig.savefig(path_chart, dpi=150, bbox_inches="tight", facecolor=BG)
print(f"\n✅ 主图已保存: {path_chart}")
plt.close(fig)

# ── 月度盈亏 + 出场原因图 ──
if r_main["trades"]:
    fig2, axes2 = plt.subplots(1, 2, figsize=(16, 6))
    fig2.patch.set_facecolor(BG)

    ax_m = axes2[0]
    df_trades = pd.DataFrame(r_main["trades"])
    df_trades["exit_time"] = pd.to_datetime(df_trades["exit_time"])
    df_trades["month"] = df_trades["exit_time"].dt.strftime("%m月")
    monthly = df_trades.groupby("month").agg(
        pnl=("pnl","sum"), trades=("pnl","count"),
        wins=("pnl", lambda x:(x>0).sum())
    )
    monthly["win_rate"] = monthly["wins"]/monthly["trades"]*100
    bar_c = [GREEN if v>=0 else RED for v in monthly["pnl"].values]
    bars = ax_m.bar(range(len(monthly)), monthly["pnl"].values, color=bar_c, alpha=0.85, width=0.6)
    ax_m.set_xticks(range(len(monthly)))
    ax_m.set_xticklabels(monthly.index, fontsize=10)
    ax_m.axhline(0, color=GRID, linewidth=0.8)
    for bar, val, wr in zip(bars, monthly["pnl"].values, monthly["win_rate"].values):
        ypos = bar.get_height() + (1 if val>=0 else -3)
        ax_m.text(bar.get_x()+bar.get_width()/2, ypos,
                  f"{val:+.1f}\n({wr:.0f}%)", ha="center", fontsize=8.5, color=TEXT,
                  va="bottom" if val>=0 else "top")
    ax_m.set_title("2026年月度盈亏 (USDT) — 最优配置", fontsize=11)
    ax_m.set_ylabel("盈亏 (USDT)", fontsize=9)
    style_ax(ax_m)

    ax_pie = axes2[1]
    ax_pie.set_facecolor(BG2)
    reason_cnt = r_main["reason_counts"]
    colors_pie = [GREEN, RED, GOLD, PURP, BLUE][:len(reason_cnt)]
    wedges, texts, autotexts = ax_pie.pie(
        list(reason_cnt.values()), labels=None, autopct="%1.1f%%",
        colors=colors_pie, startangle=140,
        pctdistance=0.75, wedgeprops=dict(linewidth=1.5, edgecolor=BG)
    )
    for at in autotexts: at.set_color(BG); at.set_fontsize(10); at.set_fontweight("bold")
    ax_pie.legend(
        wedges, [f"{l} ({v}笔)" for l,v in reason_cnt.items()],
        loc="lower center", bbox_to_anchor=(0.5,-0.1), fontsize=9,
        facecolor=BG2, edgecolor=GRID, labelcolor=TEXT, ncol=2
    )
    ax_pie.set_title("出场原因分布", fontsize=11, color=TEXT)

    fig2.suptitle("ButterflyBot · 2026年交易分析", fontsize=12, color=TEXT)
    path_monthly = "reports/backtest/backtest_2026_monthly.png"
    fig2.savefig(path_monthly, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"✅ 月度图已保存: {path_monthly}")
    plt.close(fig2)

# ── 三年汇总对比图 ──
fig3, ax3 = plt.subplots(figsize=(12, 6))
fig3.patch.set_facecolor(BG); ax3.set_facecolor(BG2)
years = ["2024", "2025", "2026"]
returns = [82.37, 38.38, r_main["net_return_pct"]]
drawdowns = [-11.42, -6.31, r_main["max_drawdown_pct"]]
win_rates = [49.42, 67.80, r_main["win_rate"]]
trades_cnt = [259, 59, r_main["total_trades"]]
x = np.arange(len(years)); width = 0.35
bars1 = ax3.bar(x - width/2, returns, width, label="净收益率 (%)", color=GREEN, alpha=0.85)
bars2 = ax3.bar(x + width/2, [abs(d) for d in drawdowns], width, label="最大回撤 (绝对值%)", color=RED, alpha=0.85)
for bar, val in zip(bars1, returns):
    ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
             f"{val:+.1f}%", ha="center", fontsize=10, color=GREEN, fontweight="bold")
for bar, val in zip(bars2, drawdowns):
    ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
             f"{val:.1f}%", ha="center", fontsize=10, color=RED, fontweight="bold")
ax3.set_xticks(x)
ax3.set_xticklabels([f"{y}年\n{t}笔 | 胜率{wr:.0f}%" for y,t,wr in zip(years,trades_cnt,win_rates)],
                     fontsize=10, color=TEXT)
ax3.set_ylabel("百分比 (%)", fontsize=10)
ax3.set_title("ButterflyBot · 三年回测对比（2024-2026）", fontsize=13)
style_ax(ax3)
ax3.legend(fontsize=10, facecolor=BG2, edgecolor=GRID, labelcolor=TEXT)
fig3.tight_layout()
path_compare = "reports/backtest/backtest_3year_compare.png"
fig3.savefig(path_compare, dpi=150, bbox_inches="tight", facecolor=BG)
print(f"✅ 三年对比图已保存: {path_compare}")
plt.close(fig3)

# ── 汇总输出 ──
print("\n" + "="*70)
print("三年回测汇总（最优配置：阈值0.12，仓位50%，止损-1.5%，止盈+3%）")
print("="*70)
print(f"  2024年: +82.37%, 回撤-11.42%, 259笔, 胜率49.42%, PF=1.437, 硬止损:否")
print(f"  2025年: +38.38%, 回撤-6.31%,   59笔, 胜率67.80%, PF=2.339, 硬止损:否")
r0 = results_2026[0]
print(f"  2026年: {r0['net_return_pct']:+.2f}%, 回撤{r0['max_drawdown_pct']:.2f}%, "
      f"{r0['total_trades']}笔, 胜率{r0['win_rate']:.2f}%, PF={r0['profit_factor']}, "
      f"硬止损:{'是' if r0['hard_stopped'] else '否'}")
