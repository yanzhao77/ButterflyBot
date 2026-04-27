#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化对比回测脚本（批量预测版，高效）
对比：固定阈值(0.70) vs 动态阈值(85%分位数) + 移动止损
数据：使用缓存数据（2025年11-12月，约1000根K线）
"""
import sys
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 只显示ERROR以上
logging.basicConfig(level=logging.ERROR)

print("=" * 70)
print("🚀 ButterflyBot 优化对比回测（批量预测版）")
print("=" * 70)

# ===================== 加载数据 =====================
from butterfly_bot.data.fetcher import fetch_ohlcv

print("\n📥 正在加载数据（使用缓存）...")
data = fetch_ohlcv("DOGE/USDT", "15m", limit=None)  # 使用全部缓存数据（约5000根K线）

if data is None or data.empty:
    print("❌ 无法获取数据，退出")
    sys.exit(1)

print(f"✅ 数据加载成功: {len(data)} 根K线")
print(f"   时间范围: {data.index[0]} ~ {data.index[-1]}")

# ===================== 批量特征工程 + 批量预测 =====================
print("\n🤖 正在进行批量特征工程和模型预测...")

from butterfly_bot.data.features import add_features, get_feature_columns
from butterfly_bot.model.model_registry import load_latest_model_path
import butterfly_bot.config.settings as cfg
import joblib

# 加载模型
model_path = load_latest_model_path()
if model_path is None or not Path(model_path).exists():
    print("❌ 无法加载模型")
    sys.exit(1)

model = joblib.load(model_path)
print(f"✅ 模型加载成功: {Path(model_path).name}")

# 批量特征工程（一次性处理所有数据）
try:
    features_df = add_features(data.copy())
    features_df = features_df.dropna()
    print(f"✅ 特征工程完成: {len(features_df)} 行特征")
except Exception as e:
    print(f"❌ 特征工程失败: {e}")
    sys.exit(1)

# 批量预测
try:
    feature_cols = get_feature_columns()
    # 只保留模型需要的特征列
    available_cols = [c for c in feature_cols if c in features_df.columns]
    X = features_df[available_cols].values
    raw_probs = model.predict(X)  # LGBModel.predict() 直接返回概率数组
    print(f"✅ 批量预测完成: {len(raw_probs)} 个预测值")
except Exception as e:
    print(f"❌ 批量预测失败: {e}")
    sys.exit(1)

# 对齐预测结果与原始数据
# features_df 的索引是原始数据的子集（特征工程可能删除了部分行）
aligned_probs = pd.Series(np.nan, index=data.index)
aligned_probs.loc[features_df.index] = raw_probs

# 用逐步EMA平滑（与AISignalCore一致，span=10，alpha=2/(10+1)）
# 注意：NaN位置（特征工程删除的行）不更新EMA，保持上一个值
PROB_EMA_SPAN = 10
alpha = 2.0 / (PROB_EMA_SPAN + 1.0)
p_ema = None
pema_list = []
for prob in aligned_probs.values:
    if not np.isnan(prob):
        if p_ema is None:
            p_ema = prob
        else:
            p_ema = alpha * prob + (1 - alpha) * p_ema
    pema_list.append(p_ema if p_ema is not None else 0.3)

predictions = np.array(pema_list)

print(f"   p_ema 统计: min={predictions.min():.4f}, max={predictions.max():.4f}, "
      f"mean={predictions.mean():.4f}")
print(f"   85%分位数={np.quantile(predictions, 0.85):.4f}, "
      f"90%分位数={np.quantile(predictions, 0.90):.4f}")

# ===================== 回测引擎 =====================
TAKE_PROFIT_PCT = 0.06
STOP_LOSS_PCT   = 0.03
TRAILING_ACT    = 0.03   # 盈利3%启动移动止损
TRAILING_LOCK2  = 0.02   # 盈利5%后锁定+2%
DYN_Q_HIGH      = 0.85
DYN_Q_LOW       = 0.20
DYN_WINDOW      = 300
FIXED_BUY_TH    = 0.70
FIXED_SELL_TH   = 0.45

class SimpleBacktest:
    def __init__(self, mode="fixed", initial_balance=1000.0):
        self.mode = mode
        self.balance = initial_balance
        self.initial_balance = initial_balance
        self.position = 0.0
        self.entry_price = 0.0
        self.trades = []
        self.pema_hist = []
        self.holding_bars = 0
        self.cooldown = 0
        self.max_profit_pct = 0.0

    def get_threshold(self, i):
        if self.mode == "dynamic" and len(self.pema_hist) >= DYN_WINDOW:
            recent = self.pema_hist[-DYN_WINDOW:]
            return float(np.quantile(recent, DYN_Q_HIGH)), float(np.quantile(recent, DYN_Q_LOW))
        return FIXED_BUY_TH, FIXED_SELL_TH

    def run(self, data, predictions):
        fee_rate = 0.001
        prices = data['close'].values
        closes = data['close'].values

        for i in range(len(data)):
            price = prices[i]
            p_ema = predictions[i]
            self.pema_hist.append(p_ema)

            if self.cooldown > 0:
                self.cooldown -= 1
                if self.position > 0:
                    self.holding_bars += 1
                continue

            buy_th, sell_th = self.get_threshold(i)

            # 持仓中：检查卖出条件
            if self.position > 0:
                self.holding_bars += 1
                profit_pct = (price - self.entry_price) / self.entry_price

                # 更新最高盈利
                if profit_pct > self.max_profit_pct:
                    self.max_profit_pct = profit_pct

                # 1. 止盈
                if profit_pct >= TAKE_PROFIT_PCT:
                    self._sell(price, fee_rate, f"止盈(+{profit_pct*100:.2f}%)", i)
                    continue

                # 2. 移动止损
                if self.max_profit_pct >= 0.05:
                    if profit_pct <= TRAILING_LOCK2:
                        self._sell(price, fee_rate, f"移动止损-锁利(最高{self.max_profit_pct*100:.1f}%)", i)
                        continue
                elif self.max_profit_pct >= TRAILING_ACT:
                    if profit_pct <= 0.0:
                        self._sell(price, fee_rate, f"移动止损-保本(最高{self.max_profit_pct*100:.1f}%)", i)
                        continue
                else:
                    if profit_pct <= -STOP_LOSS_PCT:
                        self._sell(price, fee_rate, f"固定止损({profit_pct*100:.2f}%)", i)
                        continue

                # 3. 时间止损
                if self.holding_bars >= 50:
                    self._sell(price, fee_rate, f"时间止损(持仓{self.holding_bars}根)", i)
                    continue

                # 4. AI看跌
                if p_ema <= sell_th:
                    self._sell(price, fee_rate, f"AI看跌(p_ema={p_ema:.3f})", i)
                    continue

            # 空仓：检查买入条件
            else:
                if p_ema >= buy_th:
                    # MA20趋势过滤
                    if i >= 20:
                        ma20 = closes[max(0, i-20):i].mean()
                        if price < ma20:
                            continue
                    self._buy(price, fee_rate, f"AI看涨(p_ema={p_ema:.3f},th={buy_th:.3f})", i)

        # 强制平仓
        if self.position > 0:
            last_price = prices[-1]
            profit_pct = (last_price - self.entry_price) / self.entry_price
            self._sell(last_price, fee_rate, f"回测结束({profit_pct*100:.2f}%)", len(data)-1)

        return self._report()

    def _buy(self, price, fee_rate, reason, idx):
        invest = self.balance * 0.25
        fee = invest * fee_rate
        cost = invest + fee
        if cost > self.balance:
            return
        self.position = invest / price
        self.entry_price = price
        self.balance -= cost
        self.holding_bars = 0
        self.max_profit_pct = 0.0
        self.cooldown = 5

    def _sell(self, price, fee_rate, reason, idx):
        if self.position <= 0:
            return
        revenue = self.position * price
        fee = revenue * fee_rate
        net = revenue - fee
        profit_pct = (price - self.entry_price) / self.entry_price
        profit = net - (self.position * self.entry_price)
        self.trades.append({
            "profit": profit,
            "profit_pct": profit_pct,
            "holding_bars": self.holding_bars,
            "reason": reason.split("(")[0]
        })
        self.balance += net
        self.position = 0
        self.entry_price = 0.0
        self.holding_bars = 0
        self.max_profit_pct = 0.0
        self.cooldown = 5

    def _report(self):
        if not self.trades:
            return {"mode": self.mode, "initial": self.initial_balance,
                    "final": self.balance, "trades": 0,
                    "net_pct": (self.balance - self.initial_balance) / self.initial_balance * 100,
                    "win_rate": 0, "profit_factor": 0, "avg_profit": 0, "avg_loss": 0,
                    "total_profit": 0, "total_loss": 0, "trade_list": []}
        wins = [t for t in self.trades if t["profit"] > 0]
        losses = [t for t in self.trades if t["profit"] <= 0]
        tp = sum(t["profit"] for t in wins)
        tl = abs(sum(t["profit"] for t in losses))
        pf = tp / tl if tl > 0 else float('inf')
        reasons = {}
        for t in self.trades:
            reasons[t["reason"]] = reasons.get(t["reason"], 0) + 1
        return {
            "mode": self.mode,
            "initial": self.initial_balance,
            "final": round(self.balance, 2),
            "trades": len(self.trades),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": round(len(wins) / len(self.trades) * 100, 2),
            "profit_factor": round(pf, 3),
            "total_profit": round(tp, 2),
            "total_loss": round(tl, 2),
            "avg_profit": round(tp / len(wins), 2) if wins else 0,
            "avg_loss": round(tl / len(losses), 2) if losses else 0,
            "net_pct": round((self.balance - self.initial_balance) / self.initial_balance * 100, 2),
            "reasons": reasons,
            "trade_list": self.trades
        }

# ===================== 运行对比回测 =====================
print("\n" + "=" * 70)
print("📊 运行固定阈值回测 (CONFIDENCE_THRESHOLD=0.70 + 移动止损)...")
r_fixed = SimpleBacktest(mode="fixed").run(data, predictions)

print("📊 运行动态阈值回测 (85%分位数 + 移动止损)...")
r_dyn = SimpleBacktest(mode="dynamic").run(data, predictions)

print("📊 运行固定阈值回测（无移动止损，纯固定止损，用于对比）...")
# 临时关闭移动止损：将TRAILING_ACT设为很高的值
_orig = TRAILING_ACT
TRAILING_ACT = 999  # 实际上禁用移动止损
r_no_trailing = SimpleBacktest(mode="fixed").run(data, predictions)
TRAILING_ACT = _orig

# ===================== 打印结果 =====================
def print_result(r, title):
    print(f"\n{'─'*55}")
    print(f"  📌 {title}")
    print(f"{'─'*55}")
    print(f"  初始资金:   1000.00 USDT")
    print(f"  最终资金:   {r['final']:.2f} USDT")
    print(f"  净收益率:   {r['net_pct']:+.2f}%")
    print(f"  总交易次数: {r['trades']}")
    if r['trades'] > 0:
        print(f"  胜率:       {r['win_rate']:.2f}%")
        print(f"  盈利因子:   {r['profit_factor']:.3f}")
        print(f"  总盈利:     +{r['total_profit']:.2f} USDT")
        print(f"  总亏损:     -{r['total_loss']:.2f} USDT")
        print(f"  平均盈利:   +{r['avg_profit']:.2f} USDT/笔")
        print(f"  平均亏损:   -{r['avg_loss']:.2f} USDT/笔")
        if r.get("reasons"):
            print(f"  卖出原因:")
            for k, v in sorted(r["reasons"].items(), key=lambda x: -x[1]):
                print(f"    {k}: {v}次 ({v/r['trades']*100:.1f}%)")

print_result(r_no_trailing, "基准：固定阈值(0.70) + 固定止损(-3%)")
print_result(r_fixed,       "优化1：固定阈值(0.70) + 移动止损")
print_result(r_dyn,         "优化2：动态阈值(85%分位) + 移动止损")

# ===================== 对比总结 =====================
print(f"\n{'='*70}")
print("📈 优化效果汇总对比")
print(f"{'='*70}")
print(f"{'指标':<22} {'基准(固定+固定止损)':<20} {'优化1(固定+移动止损)':<20} {'优化2(动态+移动止损)'}")
print(f"{'─'*85}")

def fmt(v, is_pct=False):
    if is_pct:
        return f"{v:+.2f}%"
    return f"{v:.2f}"

print(f"{'净收益率':<22} {fmt(r_no_trailing['net_pct'], True):<20} {fmt(r_fixed['net_pct'], True):<20} {fmt(r_dyn['net_pct'], True)}")
print(f"{'最终资金(USDT)':<22} {fmt(r_no_trailing['final']):<20} {fmt(r_fixed['final']):<20} {fmt(r_dyn['final'])}")
print(f"{'总交易次数':<22} {r_no_trailing['trades']:<20} {r_fixed['trades']:<20} {r_dyn['trades']}")
if r_fixed['trades'] > 0 and r_no_trailing['trades'] > 0:
    print(f"{'胜率':<22} {fmt(r_no_trailing['win_rate'], True):<20} {fmt(r_fixed['win_rate'], True):<20} {fmt(r_dyn['win_rate'], True)}")
    print(f"{'盈利因子':<22} {r_no_trailing['profit_factor']:<20.3f} {r_fixed['profit_factor']:<20.3f} {r_dyn['profit_factor']:.3f}")
    print(f"{'平均盈利(USDT)':<22} +{r_no_trailing['avg_profit']:<19.2f} +{r_fixed['avg_profit']:<19.2f} +{r_dyn['avg_profit']:.2f}")
    print(f"{'平均亏损(USDT)':<22} -{r_no_trailing['avg_loss']:<19.2f} -{r_fixed['avg_loss']:<19.2f} -{r_dyn['avg_loss']:.2f}")

print(f"\n{'─'*85}")
print("移动止损贡献:")
if r_fixed['trades'] > 0 and r_no_trailing['trades'] > 0:
    trailing_contrib = r_fixed['net_pct'] - r_no_trailing['net_pct']
    dynamic_contrib  = r_dyn['net_pct'] - r_fixed['net_pct']
    total_contrib    = r_dyn['net_pct'] - r_no_trailing['net_pct']
    print(f"  移动止损贡献:  {trailing_contrib:+.2f}% (固定阈值+移动止损 vs 基准)")
    print(f"  动态阈值贡献:  {dynamic_contrib:+.2f}% (动态阈值 vs 固定阈值)")
    print(f"  综合优化贡献:  {total_contrib:+.2f}% (最优 vs 基准)")

# ===================== 保存报告 =====================
import os
os.makedirs("reports/backtest", exist_ok=True)
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
report = {
    "timestamp": ts,
    "symbol": "DOGE/USDT",
    "timeframe": "15m",
    "data_range": f"{data.index[0]} ~ {data.index[-1]}",
    "total_candles": len(data),
    "p_ema_stats": {
        "min": float(predictions.min()),
        "max": float(predictions.max()),
        "mean": float(predictions.mean()),
        "q85": float(np.quantile(predictions, 0.85)),
        "q90": float(np.quantile(predictions, 0.90)),
    },
    "baseline_fixed_stop": r_no_trailing,
    "optimized_trailing_stop": r_fixed,
    "optimized_dynamic_threshold": r_dyn,
}
report_path = f"reports/backtest/optimized_comparison_{ts}.json"
with open(report_path, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, default=str)

print(f"\n✅ 报告已保存: {report_path}")
print("=" * 70)
