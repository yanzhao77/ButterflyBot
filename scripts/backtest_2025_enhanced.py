"""
ButterflyBot 2025年增强版双向回测脚本
新模型 v20260429_112717（AUC 0.754/0.744）
功能：做多 + 做空，仓位50%，账户硬止损-15%

回测参数：
  - 初始资金:   1000 USDT
  - 仓位:       50%（每笔）
  - 做多止损:   -1.5%
  - 做多止盈:   +3.0%
  - 做空止损:   -1.5%
  - 做空止盈:   +3.0%
  - 手续费:     0.1%（单边）
  - 趋势过滤:   MA50（做多要求价格>MA50，做空要求价格<MA50）
  - 账户硬止损: -15%
  - 做多阈值:   0.10（校准后概率）
  - 做空阈值:   0.08
"""
import sys, os, json, logging
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

VERSION = "v20260429_112717"

print("=" * 70)
print(f"🚀 ButterflyBot 2025年增强版双向回测")
print(f"   模型: {VERSION} | AUC多头=0.754, AUC空头=0.744")
print("=" * 70)

# ===================== 1. 加载全量数据 =====================
print("\n📥 加载历史数据...")
df_raw = fetch_ohlcv("DOGE/USDT", "15m", limit=None)
print(f"✅ 全量数据: {len(df_raw)} 根K线")

# ===================== 2. 增强版特征工程（与训练保持一致）=====================
print("\n🔧 构建增强版特征...")

def compute_rsi(series, window=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
    rs = gain / (loss + 1e-8)
    return 100 - (100 / (1 + rs))

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
    hi_diff = df["high"].diff()
    lo_diff = -df["low"].diff()
    plus_dm = hi_diff.where((hi_diff > lo_diff) & (hi_diff > 0), 0)
    minus_dm = lo_diff.where((lo_diff > hi_diff) & (lo_diff > 0), 0)
    atr = compute_atr(df, window)
    plus_di = 100 * plus_dm.rolling(window).mean() / (atr + 1e-8)
    minus_di = 100 * minus_dm.rolling(window).mean() / (atr + 1e-8)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-8)
    return dx.rolling(window).mean(), plus_di, minus_di

def compute_obv(df):
    return (np.sign(df["close"].diff()) * df["volume"]).fillna(0).cumsum()

def add_enhanced_features(df):
    df = df.copy()
    c = df["close"]; h = df["high"]; l = df["low"]; o = df["open"]; v = df["volume"]

    df["return"]     = c.pct_change()
    df["log_return"] = np.log(c / c.shift(1))
    df["return_3"]   = c.pct_change(3)
    df["return_5"]   = c.pct_change(5)
    df["return_10"]  = c.pct_change(10)
    df["return_20"]  = c.pct_change(20)
    df["return_30"]  = c.pct_change(30)

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

    for w in [6, 9, 14, 21, 28]:
        df[f"rsi_{w}"] = compute_rsi(c, w)
    df["rsi_diff"]  = df["rsi_14"] - df["rsi_14"].shift(3)
    df["rsi_cross"] = df["rsi_6"] - df["rsi_14"]

    df["macd"]             = df["ema12"] - df["ema26"]
    df["macd_signal"]      = df["macd"].ewm(span=9).mean()
    df["macd_hist"]        = df["macd"] - df["macd_signal"]
    df["macd_hist_change"] = df["macd_hist"] - df["macd_hist"].shift(1)
    fast_macd = c.ewm(span=5).mean() - c.ewm(span=13).mean()
    df["fast_macd_hist"] = fast_macd - fast_macd.ewm(span=4).mean()

    for w, sm in [(10, 2), (20, 2), (20, 1.5), (50, 2)]:
        mid = c.rolling(w).mean()
        std = c.rolling(w).std()
        up_ = mid + sm * std
        lo_ = mid - sm * std
        tag = f"bb{w}_{str(sm).replace('.','')}"
        df[f"{tag}_width"]    = (up_ - lo_) / (mid + 1e-8)
        df[f"{tag}_position"] = (c - lo_) / (up_ - lo_ + 1e-8)
    df["bb_upper"]  = df["ma20"] + 2 * c.rolling(20).std()
    df["bb_lower"]  = df["ma20"] - 2 * c.rolling(20).std()
    df["bb_middle"] = df["ma20"]

    df["atr"]        = compute_atr(df, 14)
    df["atr_7"]      = compute_atr(df, 7)
    df["atr_ratio"]  = df["atr"] / (c + 1e-8)
    df["atr_change"] = df["atr"] / (df["atr"].shift(5) + 1e-8)

    for w in [5, 10, 20, 50]:
        df[f"volatility_{w}"] = df["log_return"].rolling(w).std()
    df["vol_ratio"]  = df["volatility_10"] / (df["volatility_50"] + 1e-8)
    df["vol_change"] = df["volatility_10"] / (df["volatility_10"].shift(10) + 1e-8)

    df["stoch_k"]   = compute_stoch(df, 14)
    df["stoch_d"]   = df["stoch_k"].rolling(3).mean()
    df["stoch_k9"]  = compute_stoch(df, 9)
    df["stoch_cross"] = df["stoch_k"] - df["stoch_d"]

    for w in [14, 28]:
        hi_max = h.rolling(w).max()
        lo_min = l.rolling(w).min()
        df[f"williams_r_{w}"] = -100 * (hi_max - c) / (hi_max - lo_min + 1e-8)

    for w in [5, 10, 20, 30]:
        df[f"roc_{w}"] = ((c - c.shift(w)) / (c.shift(w) + 1e-8)) * 100

    df["adx"], df["plus_di"], df["minus_di"] = compute_adx(df, 14)
    df["di_diff"] = df["plus_di"] - df["minus_di"]

    for w in [5, 10, 20, 50]:
        df[f"volume_ma{w}"] = v.rolling(w).mean()
    df["volume_ratio"]    = v / (df["volume_ma20"] + 1e-8)
    df["volume_ratio_5"]  = v / (df["volume_ma5"] + 1e-8)
    df["volume_change"]   = v.pct_change()
    df["volume_change_5"] = v.pct_change(5)
    df["obv"]             = compute_obv(df)
    df["obv_ma"]          = df["obv"].rolling(20).mean()
    df["obv_ratio"]       = df["obv"] / (df["obv_ma"].abs() + 1e-8)
    df["vwap_ratio"]      = (c * v).rolling(20).sum() / (v.rolling(20).sum() + 1e-8) / (c + 1e-8)

    body       = (c - o).abs()
    upper_wick = h - pd.concat([c, o], axis=1).max(axis=1)
    lower_wick = pd.concat([c, o], axis=1).min(axis=1) - l
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

    for w in [3, 5, 10, 20]:
        df[f"momentum_{w}"] = c - c.shift(w)
    df["accel_5"]  = df["momentum_5"] - df["momentum_5"].shift(5)
    df["accel_10"] = df["momentum_10"] - df["momentum_10"].shift(10)

    df["hour"]        = df.index.hour
    df["day_of_week"] = df.index.dayofweek
    df["hour_sin"]    = np.sin(2 * np.pi * df.index.hour / 24)
    df["hour_cos"]    = np.cos(2 * np.pi * df.index.hour / 24)
    df["dow_sin"]     = np.sin(2 * np.pi * df.index.dayofweek / 7)
    df["dow_cos"]     = np.cos(2 * np.pi * df.index.dayofweek / 7)

    for w in [10, 20, 50, 100]:
        df[f"high_{w}"]      = h.rolling(w).max()
        df[f"low_{w}"]       = l.rolling(w).min()
        df[f"dist_high_{w}"] = (df[f"high_{w}"] - c) / (c + 1e-8)
        df[f"dist_low_{w}"]  = (c - df[f"low_{w}"]) / (c + 1e-8)

    df["rsi_vol_cross"]  = df["rsi_14"] * df["volume_ratio"]
    df["macd_atr_cross"] = df["macd_hist"] / (df["atr"] + 1e-8)
    df["bb_rsi_cross"]   = df["bb10_2_position"] * df["rsi_14"] / 100
    df["vol_momentum"]   = df["volume_ratio"] * df["return_5"]

    df.dropna(inplace=True)
    return df

df_feat_all = add_enhanced_features(df_raw)
with open(os.path.join(REGISTRY_DIR, f"{VERSION}_features.json")) as f:
    feature_cols = json.load(f)
available_cols = [c for c in feature_cols if c in df_feat_all.columns]
print(f"✅ 特征工程完成: {len(df_feat_all)} 行，{len(available_cols)} 个特征")

# ===================== 3. 加载模型并生成全量预测 =====================
print("\n🤖 加载双向模型...")
model_long  = joblib.load(os.path.join(REGISTRY_DIR, f"{VERSION}_long.pkl"))
model_short = joblib.load(os.path.join(REGISTRY_DIR, f"{VERSION}_short.pkl"))
cal_long    = joblib.load(os.path.join(REGISTRY_DIR, f"{VERSION}_cal_long.pkl"))
cal_short   = joblib.load(os.path.join(REGISTRY_DIR, f"{VERSION}_cal_short.pkl"))

X_all = df_feat_all[available_cols].values

raw_long  = model_long.predict(X_all)
raw_short = model_short.predict(X_all)
cal_long_probs  = cal_long.predict_proba(raw_long.reshape(-1, 1))[:, 1]
cal_short_probs = cal_short.predict_proba(raw_short.reshape(-1, 1))[:, 1]

# 对齐到原始索引
p_long_series  = pd.Series(np.nan, index=df_raw.index)
p_short_series = pd.Series(np.nan, index=df_raw.index)
p_long_series.loc[df_feat_all.index]  = cal_long_probs
p_short_series.loc[df_feat_all.index] = cal_short_probs

# EMA 平滑
alpha = 2.0 / (10 + 1.0)
def ema_smooth(series_vals):
    ema_val = None
    result = []
    for v in series_vals:
        if not np.isnan(v):
            ema_val = v if ema_val is None else alpha * v + (1 - alpha) * ema_val
        result.append(ema_val if ema_val is not None else 0.03)
    return np.array(result)

pema_long  = ema_smooth(p_long_series.values)
pema_short = ema_smooth(p_short_series.values)

# 截取2025年数据
mask_2025    = df_raw.index.year == 2025
df_2025      = df_raw[mask_2025].copy()
pema_long_25 = pema_long[mask_2025]
pema_short_25 = pema_short[mask_2025]

print(f"\n📅 2025年数据: {len(df_2025)} 根K线")
print(f"   多头 p_ema: min={pema_long_25.min():.4f}, max={pema_long_25.max():.4f}, mean={pema_long_25.mean():.4f}")
print(f"   空头 p_ema: min={pema_short_25.min():.4f}, max={pema_short_25.max():.4f}, mean={pema_short_25.mean():.4f}")

for th in [0.06, 0.08, 0.10, 0.12, 0.15]:
    print(f"   多头超{th:.2f}: {(pema_long_25>=th).sum():4d}次 | 空头超{th:.2f}: {(pema_short_25>=th).sum():4d}次")

# ===================== 4. 双向回测引擎 =====================
INITIAL_BALANCE  = 1000.0
POSITION_PCT     = 0.50
FEE_RATE         = 0.001
SL_LONG          = -0.015
TP_LONG          = 0.030
SL_SHORT         = -0.015
TP_SHORT         = 0.030
MAX_HOLDING_BARS = 40
COOLDOWN_BARS    = 3
HARD_STOP_PCT    = 0.15

def run_dual_backtest(df, pema_l, pema_s, label, long_th, short_th, sl_l, tp_l, sl_s, tp_s,
                      pos_pct=POSITION_PCT, use_trend_filter=True, use_hard_stop=True):
    balance      = INITIAL_BALANCE
    peak_balance = INITIAL_BALANCE
    position     = 0.0    # >0: 多头持仓量, <0: 空头持仓量
    entry_price  = 0.0
    entry_bar    = 0
    is_long      = True
    cooldown     = 0
    hard_stopped = False

    trades       = []
    equity_curve = []
    equity_dates = []

    closes = df["close"].values
    ma50   = df["close"].rolling(50).mean().values

    for i in range(len(df)):
        price = closes[i]
        pl    = pema_l[i]
        ps    = pema_s[i]

        # 当前权益
        if position > 0:
            cur_equity = balance + (price - entry_price) * position
        elif position < 0:
            cur_equity = balance + (entry_price - price) * abs(position)
        else:
            cur_equity = balance
        equity_curve.append(cur_equity)
        equity_dates.append(df.index[i])

        # 账户硬止损
        if use_hard_stop and not hard_stopped:
            if cur_equity > peak_balance:
                peak_balance = cur_equity
            dd = (peak_balance - cur_equity) / peak_balance
            if dd >= HARD_STOP_PCT:
                hard_stopped = True
                if position != 0:
                    if position > 0:
                        pnl = (price - entry_price) * position \
                              - (entry_price * position + price * position) * FEE_RATE
                    else:
                        pnl = (entry_price - price) * abs(position) \
                              - (entry_price * abs(position) + price * abs(position)) * FEE_RATE
                    balance += pnl
                    trades.append({"entry": entry_price, "exit": price, "pnl": pnl,
                                   "bars": i - entry_bar, "reason": "账户硬止损",
                                   "direction": "多" if is_long else "空",
                                   "entry_time": str(df.index[entry_bar]),
                                   "exit_time": str(df.index[i])})
                    position = 0
                continue

        if hard_stopped:
            continue

        if cooldown > 0:
            cooldown -= 1

        # ── 持仓管理 ──
        if position != 0:
            holding_bars = i - entry_bar
            if position > 0:
                profit_pct = (price - entry_price) / entry_price
                sl, tp = sl_l, tp_l
            else:
                profit_pct = (entry_price - price) / entry_price
                sl, tp = sl_s, tp_s

            reason = None
            if profit_pct <= sl:
                reason = "固定止损"
            elif profit_pct >= tp:
                reason = "止盈"
            elif holding_bars >= MAX_HOLDING_BARS:
                reason = "时间止损"
            elif position > 0 and pl <= max(0.02, long_th - 0.04):
                reason = "AI看跌"
            elif position < 0 and ps <= max(0.02, short_th - 0.04):
                reason = "AI看涨"

            if reason:
                if position > 0:
                    pnl = (price - entry_price) * position \
                          - (entry_price * position + price * position) * FEE_RATE
                else:
                    pnl = (entry_price - price) * abs(position) \
                          - (entry_price * abs(position) + price * abs(position)) * FEE_RATE
                balance += pnl
                trades.append({"entry": entry_price, "exit": price, "pnl": pnl,
                               "bars": holding_bars, "reason": reason,
                               "direction": "多" if is_long else "空",
                               "entry_time": str(df.index[entry_bar]),
                               "exit_time": str(df.index[i]),
                               "entry_idx": entry_bar, "exit_idx": i})
                position = 0; entry_price = 0; cooldown = COOLDOWN_BARS
                continue

        # ── 开仓 ──
        if position == 0 and cooldown == 0:
            ma50_val = ma50[i]
            has_ma50 = not np.isnan(ma50_val)

            # 做多信号
            if pl >= long_th:
                if not use_trend_filter or (has_ma50 and price >= ma50_val):
                    invest   = balance * pos_pct
                    fee      = invest * FEE_RATE
                    position = (invest - fee) / price
                    entry_price = price; entry_bar = i; is_long = True
                    continue

            # 做空信号
            if ps >= short_th:
                if not use_trend_filter or (has_ma50 and price <= ma50_val):
                    invest   = balance * pos_pct
                    fee      = invest * FEE_RATE
                    position = -((invest - fee) / price)
                    entry_price = price; entry_bar = i; is_long = False
                    continue

    # 强制平仓
    if position != 0:
        price = closes[-1]
        if position > 0:
            pnl = (price - entry_price) * position \
                  - (entry_price * position + price * position) * FEE_RATE
        else:
            pnl = (entry_price - price) * abs(position) \
                  - (entry_price * abs(position) + price * abs(position)) * FEE_RATE
        balance += pnl
        trades.append({"entry": entry_price, "exit": price, "pnl": pnl,
                       "bars": len(df) - 1 - entry_bar, "reason": "回测结束",
                       "direction": "多" if is_long else "空",
                       "entry_time": str(df.index[entry_bar]),
                       "exit_time": str(df.index[-1]),
                       "entry_idx": entry_bar, "exit_idx": len(df) - 1})

    total_trades = len(trades)
    long_trades  = [t for t in trades if t["direction"] == "多"]
    short_trades = [t for t in trades if t["direction"] == "空"]
    wins         = [t for t in trades if t["pnl"] > 0]
    losses       = [t for t in trades if t["pnl"] <= 0]
    win_rate     = len(wins) / total_trades * 100 if total_trades > 0 else 0
    total_profit = sum(t["pnl"] for t in wins)
    total_loss   = abs(sum(t["pnl"] for t in losses))
    pf           = total_profit / total_loss if total_loss > 0 else float("inf")
    net_return   = (balance - INITIAL_BALANCE) / INITIAL_BALANCE * 100

    eq      = np.array(equity_curve)
    peak_eq = np.maximum.accumulate(eq)
    dd_arr  = (eq - peak_eq) / peak_eq * 100
    max_dd  = dd_arr.min()

    reason_counts = {}
    for t in trades:
        reason_counts[t["reason"]] = reason_counts.get(t["reason"], 0) + 1

    return {
        "label":            label,
        "final_balance":    round(balance, 2),
        "net_return_pct":   round(net_return, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "total_trades":     total_trades,
        "long_trades":      len(long_trades),
        "short_trades":     len(short_trades),
        "win_rate":         round(win_rate, 2),
        "profit_factor":    round(pf, 3) if pf != float("inf") else "inf",
        "total_profit":     round(total_profit, 2),
        "total_loss":       round(total_loss, 2),
        "avg_profit":       round(total_profit / len(wins), 2) if wins else 0,
        "avg_loss":         round(-total_loss / len(losses), 2) if losses else 0,
        "reason_counts":    reason_counts,
        "hard_stopped":     hard_stopped,
        "equity_curve":     equity_curve,
        "equity_dates":     equity_dates,
        "drawdown_curve":   dd_arr.tolist(),
        "trades":           trades,
    }

# ===================== 5. 运行多组回测 =====================
print("\n" + "=" * 70)
print("📊 运行2025年双向回测（4组参数对比）...")
print("=" * 70)

configs = [
    # label, long_th, short_th, sl_l, tp_l, sl_s, tp_s
    ("A.旧模型仅做多(仓位10%)",  0.10, 999, -0.015, 0.030, -0.015, 0.030),
    ("B.新模型仅做多(仓位50%)",  0.10, 999, -0.015, 0.030, -0.015, 0.030),
    ("C.新模型双向(仓位50%)",    0.10, 0.08, -0.015, 0.030, -0.015, 0.030),
    ("D.新模型双向宽松(仓位50%)", 0.08, 0.07, -0.020, 0.040, -0.020, 0.040),
]

pos_pcts = [0.10, 0.50, 0.50, 0.50]
results = []
for (label, lt, st, sll, tpl, sls, tps), pos in zip(configs, pos_pcts):
    print(f"\n[{label[0]}] {label}")
    r = run_dual_backtest(df_2025, pema_long_25, pema_short_25, label,
                          lt, st, sll, tpl, sls, tps, pos_pct=pos)
    results.append(r)

# ===================== 6. 打印结果 =====================
for r in results:
    print(f"\n{'─'*65}")
    print(f"  📌 {r['label']}")
    print(f"{'─'*65}")
    print(f"  净收益率:   {r['net_return_pct']:+.2f}%  |  最终资金: {r['final_balance']:.2f} USDT")
    print(f"  最大回撤:   {r['max_drawdown_pct']:.2f}%")
    print(f"  总交易:     {r['total_trades']}笔  (多头{r['long_trades']}笔 + 空头{r['short_trades']}笔)")
    print(f"  胜率:       {r['win_rate']:.2f}%  |  盈利因子: {r['profit_factor']}")
    print(f"  平均盈利:   +{r['avg_profit']:.2f} USDT  |  平均亏损: {r['avg_loss']:.2f} USDT")
    print(f"  卖出原因:   {r['reason_counts']}")
    if r.get("hard_stopped"):
        print(f"  ⚠️  账户硬止损已触发！")

# 月度分析（C组）
rC = results[2]
print(f"\n📅 月度盈亏分析（C组-双向）:")
if rC["trades"]:
    df_t = pd.DataFrame(rC["trades"])
    df_t["exit_time"] = pd.to_datetime(df_t["exit_time"])
    df_t["month"] = df_t["exit_time"].dt.to_period("M")
    monthly = df_t.groupby("month").agg(
        trades=("pnl", "count"),
        pnl=("pnl", "sum"),
        wins=("pnl", lambda x: (x > 0).sum()),
        long_cnt=("direction", lambda x: (x == "多").sum()),
        short_cnt=("direction", lambda x: (x == "空").sum()),
    )
    monthly["win_rate"] = monthly["wins"] / monthly["trades"] * 100
    for idx, row in monthly.iterrows():
        sign = "✅" if row["pnl"] >= 0 else "❌"
        print(f"  {sign} {idx}: {int(row['trades']):3d}笔(多{int(row['long_cnt'])}+空{int(row['short_cnt'])}), "
              f"盈亏={row['pnl']:+.2f} USDT, 胜率={row['win_rate']:.0f}%")

# ===================== 7. 可视化 =====================
print("\n📊 生成可视化图表...")
plt.rcParams["font.family"]      = "sans-serif"
plt.rcParams["font.sans-serif"]  = ["Noto Sans CJK SC", "WenQuanYi Micro Hei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

C_BG   = "#161b22"
C_GRID = "#30363d"
C_TEXT = "#c9d1d9"
COLORS = ["#8b949e", "#58a6ff", "#f78166", "#3fb950"]

fig = plt.figure(figsize=(20, 16))
fig.patch.set_facecolor("#0d1117")
gs = GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.30, height_ratios=[2.5, 1.5, 1.5])

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
ax1   = fig.add_subplot(gs[0, :])
ax1_r = ax1.twinx()
ax1_r.fill_between(df_2025.index, df_2025["close"].values, alpha=0.07, color="#e3b341")
ax1_r.plot(df_2025.index, df_2025["close"].values, color="#e3b341", linewidth=0.8, alpha=0.5)
ax1_r.set_ylabel("DOGE 价格 (USDT)", color="#e3b341", fontsize=9)
ax1_r.tick_params(colors="#e3b341", labelsize=8)
ax1_r.spines["right"].set_color("#e3b341")
for sp in ["top", "left", "bottom"]:
    ax1_r.spines[sp].set_visible(False)
ax1_r.set_facecolor(C_BG)

for i, r in enumerate(results):
    ax1.plot(pd.to_datetime(r["equity_dates"]), r["equity_curve"],
             color=COLORS[i], linewidth=1.8,
             label=f"{r['label']} ({r['net_return_pct']:+.2f}%)")
ax1.axhline(1000, color="#555", linewidth=0.8, linestyle="--", alpha=0.6, label="初始资金")

ax1.set_title("2025年回测权益曲线（4组参数对比）", fontsize=12)
ax1.set_ylabel("账户权益 (USDT)", fontsize=9)
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax1.xaxis.set_major_locator(mdates.MonthLocator())
style_ax(ax1)
ax1.legend(loc="upper left", fontsize=9, facecolor=C_BG, edgecolor=C_GRID, labelcolor=C_TEXT)

# 子图2：回撤曲线
ax2 = fig.add_subplot(gs[1, :])
for i, r in enumerate(results):
    ax2.fill_between(pd.to_datetime(r["equity_dates"]), r["drawdown_curve"],
                     alpha=0.2, color=COLORS[i])
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

# 子图3：月度盈亏（C组）
ax3 = fig.add_subplot(gs[2, 0])
if rC["trades"]:
    df_t2 = pd.DataFrame(rC["trades"])
    df_t2["exit_time"] = pd.to_datetime(df_t2["exit_time"])
    df_t2["month"] = df_t2["exit_time"].dt.to_period("M").astype(str)
    monthly_pnl = df_t2.groupby("month")["pnl"].sum()
    bar_colors  = [COLORS[3] if v >= 0 else COLORS[2] for v in monthly_pnl.values]
    ax3.bar(range(len(monthly_pnl)), monthly_pnl.values, color=bar_colors, alpha=0.85)
    ax3.set_xticks(range(len(monthly_pnl)))
    ax3.set_xticklabels([m[5:] for m in monthly_pnl.index], fontsize=8, rotation=45)
    ax3.axhline(0, color=C_GRID, linewidth=0.8)
    ax3.set_title("月度盈亏 (C组-双向)", fontsize=10)
    ax3.set_ylabel("盈亏 (USDT)", fontsize=9)
style_ax(ax3)

# 子图4：多空交易分布（C组）
ax4 = fig.add_subplot(gs[2, 1])
if rC["trades"]:
    df_t3 = pd.DataFrame(rC["trades"])
    long_pnls  = [t["pnl"] for t in rC["trades"] if t["direction"] == "多"]
    short_pnls = [t["pnl"] for t in rC["trades"] if t["direction"] == "空"]
    bins = np.linspace(
        min(min(long_pnls, default=[0]), min(short_pnls, default=[0])),
        max(max(long_pnls, default=[0]), max(short_pnls, default=[0])),
        30
    )
    if long_pnls:
        ax4.hist(long_pnls,  bins=bins, alpha=0.6, color=COLORS[1], label=f"多头({len(long_pnls)}笔)")
    if short_pnls:
        ax4.hist(short_pnls, bins=bins, alpha=0.6, color=COLORS[2], label=f"空头({len(short_pnls)}笔)")
    ax4.axvline(0, color=C_TEXT, linewidth=1, linestyle="--")
    ax4.set_title("多空盈亏分布 (C组)", fontsize=10)
    ax4.set_xlabel("单笔盈亏 (USDT)", fontsize=9)
    ax4.legend(fontsize=9, facecolor=C_BG, edgecolor=C_GRID, labelcolor=C_TEXT)
style_ax(ax4)

# 汇总信息
rC_data = results[2]
summary_text = (
    f"2025年回测汇总\n"
    f"{'─'*28}\n"
    f"数据: 2025-01-01 ~ 2025-12-23\n"
    f"K线: {len(df_2025):,} 根 (15m)\n"
    f"模型: {VERSION}\n"
    f"{'─'*28}\n"
    f"C组净收益: {rC_data['net_return_pct']:+.2f}%\n"
    f"C组最大回撤: {rC_data['max_drawdown_pct']:.2f}%\n"
    f"C组交易: {rC_data['total_trades']}笔\n"
    f"  多头: {rC_data['long_trades']}笔\n"
    f"  空头: {rC_data['short_trades']}笔\n"
    f"C组胜率: {rC_data['win_rate']:.2f}%\n"
    f"C组盈利因子: {rC_data['profit_factor']}"
)
fig.text(0.01, 0.01, summary_text, fontsize=8, color=C_TEXT, va="bottom", ha="left",
         bbox=dict(boxstyle="round", facecolor=C_BG, edgecolor=C_GRID, alpha=0.8))

fig.suptitle(
    f"ButterflyBot 2025年增强版双向回测\n"
    f"DOGE/USDT 15m | 新模型 {VERSION} (AUC=0.754) | 仓位50% | 初始资金: 1,000 USDT",
    fontsize=13, color=C_TEXT, y=0.99
)

chart_path = "reports/backtest/backtest_2025_enhanced_chart.png"
plt.savefig(chart_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"✅ 图表已保存: {chart_path}")
plt.close()

# ===================== 8. 保存 JSON =====================
timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
report_data = {
    "timestamp": timestamp,
    "symbol":    "DOGE/USDT",
    "timeframe": "15m",
    "period":    "2025-01-01 ~ 2025-12-23",
    "total_bars": len(df_2025),
    "model":     VERSION,
    "config": {
        "initial_balance": INITIAL_BALANCE,
        "sl_long": SL_LONG, "tp_long": TP_LONG,
        "sl_short": SL_SHORT, "tp_short": TP_SHORT,
        "hard_stop_pct": HARD_STOP_PCT,
    },
    "results": {
        r["label"]: {k: v for k, v in r.items()
                     if k not in ("equity_curve", "equity_dates", "drawdown_curve", "trades")}
        for r in results
    },
}
json_path = f"reports/backtest/backtest_2025_enhanced_{timestamp}.json"
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(report_data, f, ensure_ascii=False, indent=2)
print(f"✅ JSON报告已保存: {json_path}")

print("\n" + "=" * 70)
print("✅ 2025年双向回测完成！")
best = max(results, key=lambda r: r["net_return_pct"])
print(f"   最优配置: {best['label']}")
print(f"   净收益={best['net_return_pct']:+.2f}%, 最大回撤={best['max_drawdown_pct']:.2f}%, "
      f"交易{best['total_trades']}笔(多{best['long_trades']}+空{best['short_trades']}), "
      f"胜率{best['win_rate']:.2f}%")
print("=" * 70)
