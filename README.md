# 🤖 Butterfly Bot：基于多模型融合的智能量化交易系统

*庄生晓梦迷蝴蝶，望帝春心托杜鹃。*
*此情可待成追忆，只是当时已惘然。*

[![Python](https://img.shields.io/badge/Python-3.13.7%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Backtrader](https://img.shields.io/badge/Backtrader-1.9.76-orange)](https://www.backtrader.com/)

---

## 📌 简介

**Butterfly Bot** 是一个轻量级、可扩展的量化交易系统，融合了 **LightGBM 机器学习模型**、**Prophet 趋势预测** 与 **技术指标规则引擎**，通过动态权重集成策略生成交易信号。系统支持：

- ✅ **多因子融合预测**（LGBM + Prophet + RSI/MACD 规则）
- ✅ **动态权重调整**（根据回测 AUC 自动分配模型权重）
- ✅ **模型版本管理**（带元数据、AUC、训练时间等）
- ✅ **自动重训练机制**（回测性能提升时自动触发）
- ✅ **时间框架自适应**（高频禁用 Prophet，避免过拟合）
- ✅ **完整回测流水线**（胜率、盈亏比、AUC 等指标）

适用于 **BTC/USDT** 等主流加密货币在 **1m ~ 1d** 时间框架下的日内或波段交易。

---

## 📁 项目结构

```
butterfly_bot/
├── config/               # 配置文件
├── data/                 # 数据获取与特征工程
├── models/               # 模型训练、加载与版本管理
├── strategies/           # AI 交易策略
├── backtest/             # 回测执行与指标计算
├── utils/                # 工具函数（自动重训练、时间框架处理）
├── live/                 # （预留）实盘交易接口
└── requirements.txt      # 依赖库
```
### python版本是3.13.7
---

## ⚙️ 快速开始

### 1. 安装依赖

```bash
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


### 2. 首次训练模型

```bash
python model/train.py
```

将生成首个模型版本（如 `v20251103_1520.pkl`）并创建 `models/latest_model.txt`。

---

### 3. 运行回测

```bash
python backtest/run_backtest.py
```

输出示例：
```
💼 初始资金: 10,000.00 USDT
💰 最终资金: 12,345.67 USDT
📈 收益率: 23.46%
📈 回测完成 | AUC: 0.6821, 胜率: 58.3%
🔍 当前模型 AUC: 0.6210 | 回测 AUC: 0.6821
🚀 启动自动重训练...
✅ 模型已保存: ./models/registry/v20251103_1530.pkl
🏆 最优模型更新为: v20251103_1530
```

系统会自动判断是否需要重训练，并更新最优模型。

---

### 4. 查看模型版本

```bash
ls model/registry/          # 查看所有模型
cat model/latest_model.txt  # 查看当前最优版本
```

---

## 🔧 配置说明

### `config/settings.py`

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `TIMEFRAME` | `"1h"` | 交易周期（支持 `"1m"`, `"5m"`, `"15m"`, `"1h"`, `"4h"`, `"1d"`） |
| `INITIAL_CASH` | `10000.0` | 回测初始资金（USDT） |
| `MODEL_METRICS_PATH` | `"./backtest/strategy_metrics.json"` | 回测指标保存路径 |

> 修改 `TIMEFRAME` 后，系统会自动调整 Prophet 使用策略（≤15m 时禁用）。

---

## 🧠 模型融合逻辑

系统采用 **加权集成** 方式融合三种信号：

| 组件 | 权重范围 | 说明 |
|------|--------|------|
| **LightGBM** | 0.3 ~ 0.6 | 基于多因子的主模型，权重由 AUC 决定 |
| **Prophet** | 0.0 ~ 0.4 | 趋势预测，仅在 ≥15m 周期启用 |
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

可通过修改 `utils/auto_retrain.py` 中的 `threshold_delta_auc` 调整敏感度。

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

这些指标用于驱动模型权重与重训练决策。

---

## 🚀 扩展方向（未来计划）

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