"""
ButterflyBot 最终配置回测（阈值0.12）
模型: v20260429_112717 (AUC=0.754)
参数: 仅做多, 仓位50%, 阈值0.12, 止损-1.5%, 止盈+3%, 账户硬止损-15%
数据: 2024年 + 2025年
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
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from datetime import datetime

logging.basicConfig(level=logging.WARNING)

from butterfly_bot.data.fetcher import fetch_ohlcv
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
print(f"🚀 ButterflyBot 最终配置回测（阈值0.12）")
print(f"   模型: {VERSION} | 仓位: {POSITION_PCT*100:.0f}% | 阈值: {LONG_TH}")
print(f"   止损: {SL_LONG*100:.1f}% | 止盈: {TP_LONG*100:.1f}% | 硬止损: -{HARD_STOP_PCT*100:.0f}%")
print("=" * 70)

# ===================== 1. 加载数据 =====================
print("\n📥 加载历史数据...")
df_raw = fetch_ohlcv("DOGE/USDT", "15m", limit=None)
print(f"✅ 全量: {len(df_raw)} 根K线 ({df_raw.index[0]} ~ {df_raw.index[-1]})")

# ===================== 2. 增强版特征工程 =====================
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
    c = df["close"]; h = df["high"]; l = df["low"]; o = df["open"]; v = df["volume"]
    df["return"] = c.pct_change(); df["log_return"] = np.log(c / c.shift(1))
    for p in [3,5,10,20,30]: df[f"return_{p}"] = c.pct_change(p)
    for w in [5,10,20,50,100,200]: df[f"ma{w}"] = c.rolling(w).mean()
    for s in [9,12,21,26,55]: df[f"ema{s}"] = c.ewm(span=s).mean()
    df["ma_diff_short"] = df["ma5"]-df["ma20"]; df["ma_diff_long"] = df["ma20"]-df["ma50"]
    df["ma_diff_200"] = df["ma50"]-df["ma200"]
    df["price_to_ma20"]  = (c-df["ma20"])/(df["ma20"]+1e-8)
    df["price_to_ma50"]  = (c-df["ma50"])/(df["ma50"]+1e-8)
    df["price_to_ma100"] = (c-df["ma100"])/(df["ma100"]+1e-8)
    df["price_to_ma200"] = (c-df["ma200"])/(df["ma200"]+1e-8)
    df["ema_cross"] = df["ema12"]-df["ema26"]; df["ema_cross_9_21"] = df["ema9"]-df["ema21"]
    for w in [6,9,14,21,28]: df[f"rsi_{w}"] = compute_rsi(c, w)
    df["rsi_diff"] = df["rsi_14"]-df["rsi_14"].shift(3); df["rsi_cross"] = df["rsi_6"]-df["rsi_14"]
    df["macd"] = df["ema12"]-df["ema26"]; df["macd_signal"] = df["macd"].ewm(span=9).mean()
    df["macd_hist"] = df["macd"]-df["macd_signal"]; df["macd_hist_change"] = df["macd_hist"]-df["macd_hist"].shift(1)
    fast_macd = c.ewm(span=5).mean()-c.ewm(span=13).mean()
    df["fast_macd_hist"] = fast_macd - fast_macd.ewm(span=4).mean()
    for w, sm in [(10,2),(20,2),(20,1.5),(50,2)]:
        mid = c.rolling(w).mean(); std = c.rolling(w).std()
        up_ = mid+sm*std; lo_ = mid-sm*std
        tag = f"bb{w}_{str(sm).replace('.','')}"
        df[f"{tag}_width"] = (up_-lo_)/(mid+1e-8); df[f"{tag}_position"] = (c-lo_)/(up_-lo_+1e-8)
    df["bb_upper"] = df["ma20"]+2*c.rolling(20).std()
    df["bb_lower"] = df["ma20"]-2*c.rolling(20).std(); df["bb_middle"] = df["ma20"]
    df["atr"] = compute_atr(df,14); df["atr_7"] = compute_atr(df,7)
    df["atr_ratio"] = df["atr"]/(c+1e-8); df["atr_change"] = df["atr"]/(df["atr"].shift(5)+1e-8)
    for w in [5,10,20,50]: df[f"volatility_{w}"] = df["log_return"].rolling(w).std()
    df["vol_ratio"] = df["volatility_10"]/(df["volatility_50"]+1e-8)
    df["vol_change"] = df["volatility_10"]/(df["volatility_10"].shift(10)+1e-8)
    df["stoch_k"] = compute_stoch(df,14); df["stoch_d"] = df["stoch_k"].rolling(3).mean()
    df["stoch_k9"] = compute_stoch(df,9); df["stoch_cross"] = df["stoch_k"]-df["stoch_d"]
    for w in [14,28]:
        hi_max = h.rolling(w).max(); lo_min = l.rolling(w).min()
        df[f"williams_r_{w}"] = -100*(hi_max-c)/(hi_max-lo_min+1e-8)
    for w in [5,10,20,30]: df[f"roc_{w}"] = ((c-c.shift(w))/(c.shift(w)+1e-8))*100
    df["adx"], df["plus_di"], df["minus_di"] = compute_adx(df,14)
    df["di_diff"] = df["plus_di"]-df["minus_di"]
    for w in [5,10,20,50]: df[f"volume_ma{w}"] = v.rolling(w).mean()
    df["volume_ratio"] = v/(df["volume_ma20"]+1e-8); df["volume_ratio_5"] = v/(df["volume_ma5"]+1e-8)
    df["volume_change"] = v.pct_change(); df["volume_change_5"] = v.pct_change(5)
    df["obv"] = compute_obv(df); df["obv_ma"] = df["obv"].rolling(20).mean()
    df["obv_ratio"] = df["obv"]/(df["obv_ma"].abs()+1e-8)
    df["vwap_ratio"] = (c*v).rolling(20).sum()/(v.rolling(20).sum()+1e-8)/(c+1e-8)
    body = (c-o).abs(); upper_wick = h-pd.concat([c,o],axis=1).max(axis=1)
    lower_wick = pd.concat([c,o],axis=1).min(axis=1)-l; total_range = h-l+1e-8
    df["body_ratio"] = body/total_range; df["upper_wick_ratio"] = upper_wick/total_range
    df["lower_wick_ratio"] = lower_wick/total_range; df["close_position"] = (c-l)/total_range
    df["high_low_ratio"] = total_range/(c+1e-8); df["is_bullish"] = (c>o).astype(int)
    df["hammer_score"] = (lower_wick/total_range)*(1-body/total_range)
    df["shooting_star"] = (upper_wick/total_range)*(1-body/total_range)
    df["consecutive_bull"] = (c>o).astype(int).rolling(3).sum()
    df["consecutive_bear"] = (c<o).astype(int).rolling(3).sum()
    for w in [3,5,10,20]: df[f"momentum_{w}"] = c-c.shift(w)
    df["accel_5"] = df["momentum_5"]-df["momentum_5"].shift(5)
    df["accel_10"] = df["momentum_10"]-df["momentum_10"].shift(10)
    df["hour"] = df.index.hour; df["day_of_week"] = df.index.dayofweek
    df["hour_sin"] = np.sin(2*np.pi*df.index.hour/24); df["hour_cos"] = np.cos(2*np.pi*df.index.hour/24)
    df["dow_sin"] = np.sin(2*np.pi*df.index.dayofweek/7); df["dow_cos"] = np.cos(2*np.pi*df.index.dayofweek/7)
    for w in [10,20,50,100]:
        df[f"high_{w}"] = h.rolling(w).max(); df[f"low_{w}"] = l.rolling(w).min()
        df[f"dist_high_{w}"] = (df[f"high_{w}"]-c)/(c+1e-8)
        df[f"dist_low_{w}"]  = (c-df[f"low_{w}"])/(c+1e-8)
    df["rsi_vol_cross"]  = df["rsi_14"]*df["volume_ratio"]
    df["macd_atr_cross"] = df["macd_hist"]/(df["atr"]+1e-8)
    df["bb_rsi_cross"]   = df["bb10_2_position"]*df["rsi_14"]/100
    df["vol_momentum"]   = df["volume_ratio"]*df["return_5"]
    df.dropna(inplace=True)
    return df

print("\n🔧 构建增强版特征...")
df_feat_all = add_enhanced_features(df_raw)
with open(os.path.join(REGISTRY_DIR, f"{VERSION}_features.json")) as f:
    feature_cols = json.load(f)
available_cols = [c for c in feature_cols if c in df_feat_all.columns]
print(f"✅ 特征: {len(available_cols)} 个，样本: {len(df_feat_all)} 行")

# ===================== 3. 模型预测 =====================
print("\n🤖 生成预测概率...")
model_long = joblib.load(os.path.join(REGISTRY_DIR, f"{VERSION}_long.pkl"))
cal_long   = joblib.load(os.path.join(REGISTRY_DIR, f"{VERSION}_cal_long.pkl"))
X_all      = df_feat_all[available_cols].values
raw_long   = model_long.predict(X_all)
cal_probs  = cal_long.predict_proba(raw_long.reshape(-1,1))[:,1]

p_long_series = pd.Series(np.nan, index=df_raw.index)
p_long_series.loc[df_feat_all.index] = cal_probs

alpha = 2.0/(10+1.0)
def ema_smooth(vals):
    ema_val = None; result = []
    for v in vals:
        if not np.isnan(v):
            ema_val = v if ema_val is None else alpha*v+(1-alpha)*ema_val
        result.append(ema_val if ema_val is not None else 0.03)
    return np.array(result)

pema_all = ema_smooth(p_long_series.values)

# ===================== 4. 回测引擎 =====================
def run_backtest(df, pema, label, long_th=LONG_TH, sl=SL_LONG, tp=TP_LONG, pos_pct=POSITION_PCT):
    balance = INITIAL_BAL; peak_balance = INITIAL_BAL
    position = 0.0; entry_price = 0.0; entry_bar = 0
    cooldown = 0; hard_stopped = False
    trades = []; equity_curve = []; equity_dates = []
    closes = df["close"].values; ma50 = df["close"].rolling(50).mean().values

    for i in range(len(df)):
        price = closes[i]; pl = pema[i]
        cur_equity = balance + (price-entry_price)*position if position > 0 else balance
        equity_curve.append(cur_equity); equity_dates.append(df.index[i])

        if not hard_stopped:
            if cur_equity > peak_balance: peak_balance = cur_equity
            if (peak_balance-cur_equity)/peak_balance >= HARD_STOP_PCT:
                hard_stopped = True
                if position > 0:
                    pnl = (price-entry_price)*position - (entry_price*position+price*position)*FEE_RATE
                    balance += pnl
                    trades.append({"entry":entry_price,"exit":price,"pnl":pnl,
                                   "bars":i-entry_bar,"reason":"账户硬止损",
                                   "entry_time":str(df.index[entry_bar]),"exit_time":str(df.index[i]),
                                   "entry_idx":entry_bar,"exit_idx":i,"direction":"long"})
                    position = 0
                continue
        if hard_stopped: continue
        if cooldown > 0: cooldown -= 1

        if position > 0:
            profit_pct = (price-entry_price)/entry_price
            holding_bars = i-entry_bar; reason = None
            if profit_pct <= sl: reason = "固定止损"
            elif profit_pct >= tp: reason = "止盈"
            elif holding_bars >= MAX_HOLD_BARS: reason = "时间止损"
            elif pl <= max(0.02, long_th-0.04): reason = "AI看跌"
            if reason:
                pnl = (price-entry_price)*position - (entry_price*position+price*position)*FEE_RATE
                balance += pnl
                trades.append({"entry":entry_price,"exit":price,"pnl":pnl,
                               "bars":holding_bars,"reason":reason,
                               "entry_time":str(df.index[entry_bar]),"exit_time":str(df.index[i]),
                               "entry_idx":entry_bar,"exit_idx":i,"direction":"long"})
                position = 0; cooldown = COOLDOWN_BARS; continue

        if position == 0 and cooldown == 0:
            ma50_val = ma50[i]
            if pl >= long_th and not np.isnan(ma50_val) and price >= ma50_val:
                invest = balance*pos_pct; fee = invest*FEE_RATE
                position = (invest-fee)/price; entry_price = price; entry_bar = i

    if position > 0:
        price = closes[-1]
        pnl = (price-entry_price)*position - (entry_price*position+price*position)*FEE_RATE
        balance += pnl
        trades.append({"entry":entry_price,"exit":price,"pnl":pnl,
                       "bars":len(df)-1-entry_bar,"reason":"回测结束",
                       "entry_time":str(df.index[entry_bar]),"exit_time":str(df.index[-1]),
                       "entry_idx":entry_bar,"exit_idx":len(df)-1,"direction":"long"})

    wins = [t for t in trades if t["pnl"]>0]; losses = [t for t in trades if t["pnl"]<=0]
    total = len(trades); win_rate = len(wins)/total*100 if total>0 else 0
    tot_profit = sum(t["pnl"] for t in wins); tot_loss = abs(sum(t["pnl"] for t in losses))
    pf = tot_profit/tot_loss if tot_loss>0 else float("inf")
    net_return = (balance-INITIAL_BAL)/INITIAL_BAL*100
    eq = np.array(equity_curve); peak_eq = np.maximum.accumulate(eq)
    dd_arr = (eq-peak_eq)/peak_eq*100

    reason_counts = {}
    for t in trades: reason_counts[t["reason"]] = reason_counts.get(t["reason"],0)+1

    return {
        "label": label, "final_balance": round(balance,2),
        "net_return_pct": round(net_return,2), "max_drawdown_pct": round(dd_arr.min(),2),
        "total_trades": total, "win_rate": round(win_rate,2),
        "profit_factor": round(pf,3) if pf!=float("inf") else "inf",
        "total_profit": round(tot_profit,2), "total_loss": round(tot_loss,2),
        "avg_profit": round(tot_profit/len(wins),2) if wins else 0,
        "avg_loss": round(-tot_loss/len(losses),2) if losses else 0,
        "reason_counts": reason_counts, "hard_stopped": hard_stopped,
        "equity_curve": equity_curve, "equity_dates": equity_dates,
        "drawdown_curve": dd_arr.tolist(), "trades": trades,
    }

# 分年回测
results = {}
for year in [2024, 2025]:
    mask = df_raw.index.year == year
    df_y = df_raw[mask].copy(); pema_y = pema_all[mask]
    r = run_backtest(df_y, pema_y, str(year))
    results[year] = r
    hs = "⚠️ 硬止损触发" if r["hard_stopped"] else "✅"
    print(f"  {hs} [{year}] 净收益={r['net_return_pct']:+.2f}%, 回撤={r['max_drawdown_pct']:.2f}%, "
          f"交易={r['total_trades']}笔, 胜率={r['win_rate']:.2f}%, PF={r['profit_factor']}")

# 保存结果
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
report = {
    "timestamp": timestamp, "symbol": "DOGE/USDT", "timeframe": "15m",
    "model": VERSION, "long_th": LONG_TH, "sl": SL_LONG, "tp": TP_LONG,
    "position_pct": POSITION_PCT, "hard_stop_pct": HARD_STOP_PCT,
    "results": {
        str(y): {k:v for k,v in r.items() if k not in ("equity_curve","equity_dates","drawdown_curve","trades")}
        for y,r in results.items()
    }
}
json_path = f"reports/backtest/backtest_final_012_{timestamp}.json"
with open(json_path,"w",encoding="utf-8") as f:
    json.dump(report,f,ensure_ascii=False,indent=2)
print(f"\n✅ 数据已保存: {json_path}")
print(f"   2024: {results[2024]['net_return_pct']:+.2f}%  |  2025: {results[2025]['net_return_pct']:+.2f}%")

# 导出供图表脚本使用
import pickle
with open(f"reports/backtest/backtest_final_012_data.pkl","wb") as f:
    pickle.dump({
        "results": results,
        "df_2024": df_raw[df_raw.index.year==2024].copy(),
        "df_2025": df_raw[df_raw.index.year==2025].copy(),
        "pema_2024": pema_all[df_raw.index.year==2024],
        "pema_2025": pema_all[df_raw.index.year==2025],
        "timestamp": timestamp,
    }, f)
print("✅ 中间数据已缓存，供图表脚本使用")
