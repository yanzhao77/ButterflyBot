# 🤖 AI Quant Trading（Butterfly）

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Backtrader](https://img.shields.io/badge/Backtrader-1.9.76-orange)](https://www.backtrader.com/)

---
*庄生晓梦迷蝴蝶，望帝春心托杜鹃。*
*此情可待成追忆，只是当时已惘然。*
---

---

一个可落地的加密量化框架：LightGBM 信号 + Prophet 趋势 + 规则引擎 的融合模型，支持分页抓取、模型注册管理、保守风控回测，以及基于 AISignalCore 的实时/模拟交易。

---

## 📁 项目结构

```
ai-quant-trading/
├── config/               # 配置文件
├── data/                 # 数据获取与特征工程
├── models/               # 模型训练、加载与版本管理
├── strategies/           # 策略核心 + Backtrader 适配
├── backtest/             # 回测入口与指标
├── live/                 # 实时/模拟交易 Runner
└── requirements.txt      # 依赖库
```
---

## ⚙️ 快速开始

### 1. 安装依赖

```bash
python -m venv .venv && . .venv/Scripts/activate  # Windows: .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> **注意**：`prophet` 在部分系统需额外安装 `cmdstanpy`，如遇问题请参考 [Prophet 安装指南](https://facebook.github.io/prophet/docs/installation.html)。

---

### 1.a 在 Windows (PowerShell) 下的安装与常见问题

如果你在 Windows/PowerShell 上运行，下面的步骤能帮你快速安装依赖并避免常见问题：

- 推荐先创建并激活虚拟环境（PowerShell）：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

- 然后安装项目依赖：

```powershell
python -m pip install -r requirements.txt
```

- Prophet 在某些系统需要额外的后端（cmdstanpy / C++ 编译器）。如果安装 prophet 失败，尝试：

```powershell
python -m pip install "cmdstanpy>=1.0"
python -m pip install prophet
```

如果仍失败，可考虑使用 conda（更稳定）：

```powershell
conda create -n butterfly python=3.11
conda activate butterfly
conda install -c conda-forge prophet
python -m pip install -r requirements.txt
```

- 若遇到 LightGBM 安装问题（Windows 上常见），可以先安装二进制 wheel：

```powershell
python -m pip install lightgbm
```

- FastAPI/uvicorn 已加入 requirements，启动 API 的示例命令在下文。


### 2) 训练模型

```bash
python model/train.py
```

自动分页抓取更长历史，生成目标（`TARGET_SHIFT/THRESHOLD`），并将最佳模型登记为 latest。

---

### 3) 回测（默认 AISignalStrategy）

```bash
python backtest/run_backtest.py
```

回测按 `RETRAIN_SINCE_DAYS` 计算 since 并分页抓取，避免只拿到 1000 根。

---

### 4) 查看模型版本

```bash
ls model/registry/          # 查看所有模型
cat model/latest_model.txt  # 查看当前最优版本
```

---

## 🔧 配置说明（config/settings.py）

### `config/settings.py`

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `TIMEFRAME` | `"1m/5m/15m/1h/4h/1d"` | 交易周期 |
| `INITIAL_CASH` | `1000.0` | 回测初始资金（USDT） |
| `RETRAIN_SINCE_DAYS` / `RETRAIN_LIMIT` | `180` / `100000` | 抓数窗口与上限 |
| `MAX_POSITION_RATIO` | `0.2~0.5` | 单次资金占比上限 |
| `STOP_LOSS_PCT` / `TAKE_PROFIT_PCT` | `0.01/0.02` | 百分比止损/止盈 |
| `CONFIDENCE_THRESHOLD` / `SELL_THRESHOLD` | `0.6/0.5` | 固定阈值（兜底） |
| `USE_QUANTILE_THRESH` + `PROB_Q_HIGH/LOW/WINDOW` | `True/0.9/0.55/500` | 分位数自适应阈值 |
| `PROB_EMA_SPAN` / `REQUIRE_P_EMA_UP` | `10/True` | 概率平滑与动量过滤 |
| `TIME_STOP_BARS` / `COOLDOWN_BARS` | `30/5` | 时间止损与冷却 |
| `TRADE_ONLY_ON_CANDLE_CLOSE` | `True` | 实时仅闭合K线交易 |

> 修改 `TIMEFRAME` 后，系统会自动调整 Prophet 使用策略（≤15m 时禁用）。

---

## 🧠 模型融合逻辑

系统采用 **加权集成** 方式融合三种信号：

| 组件 | 权重范围 | 说明 |
|------|--------|------|
| **LightGBM** | 0.3 ~ 0.6 | 基于多因子的主模型，权重由 AUC 决定 |
| **Prophet** | 0.0 ~ 0.4 | 趋势预测，短周期可自动禁用 |
| **规则引擎** | ≥0.1 | 基于 RSI<30 / MACD 金叉等经典信号 |

权重计算规则：
- AUC ≥ 0.7 → LGBM 权重 0.6
- AUC ≥ 0.6 → LGBM 权重 0.5
- AUC ≥ 0.55 → LGBM 权重 0.4
- 否则 → LGBM 权重 0.3

---

## 🔄 自动重训练机制

- 回测结束后，系统对比 **当前模型 AUC** 与 **回测 AUC**
- 若 `回测 AUC > 当前 AUC + 0.02`，则触发重训练
- 训练后自动选择历史 **AUC 最高** 的模型作为 `latest`

回测模块已预留自动重训练入口（默认关闭）。

---

## 📊 回测指标

回测结果保存于 `backtest/strategy_metrics.json`，包含：

```json
{
  "auc": 0.6821,
  "win_rate": 0.583,
  "win_loss_ratio": 2.1,
  "total_trades": 120
}
```

这些指标用于评估策略稳健性。

---

## 🚀 实时/模拟交易（LiveRunner）

```bash
python live/live_runner.py
```

- `USE_REAL_MONEY=False` 为模拟；设 True 并配置 `API_KEY/API_SECRET` 进入实盘。
- `TRADE_ONLY_ON_CANDLE_CLOSE=False` 可在未闭合K线时按最新价评估并下单（调试用）。

---

## 🚀 扩展方向

- [ ] 实盘交易接口（Binance API）
- [ ] Telegram / 邮件交易通知
- [ ] Web 可视化看板（Streamlit / Dash）
- [ ] 多币种并行策略
- [ ] 风险控制模块（最大回撤止损）

---

## 📜 许可证

本项目采用 [MIT 许可证](LICENSE)。

---

## 🙌 致谢

- [Backtrader](https://www.backtrader.com/)：回测框架
- [LightGBM](https://lightgbm.readthedocs.io/)：高效梯度提升
- [Prophet](https://facebook.github.io/prophet/)：时间序列预测
- [CCXT](https://ccxt.trade/)：统一交易所 API

---

> 💡 **提示**：本系统仅供学习与研究使用，**不构成投资建议**。实盘交易前请充分测试并评估风险。