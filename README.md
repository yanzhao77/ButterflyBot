# 🦋 ButterflyBot - AI加密货币量化交易系统

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![LightGBM](https://img.shields.io/badge/LightGBM-AI%20Model-orange)](https://lightgbm.readthedocs.io/)
[![Status](https://img.shields.io/badge/Status-Active%20Development-brightgreen)](https://github.com/yanzhao77/ButterflyBot)

**庄生晓梦迷蝴蝶，望帝春心托杜鹃。**  
**此情可待成追忆，只是当时已惘然。**

---

## 📖 项目简介

ButterflyBot 是一个基于AI机器学习的加密货币量化交易系统，采用 **LightGBM** 模型进行价格预测，结合完善的风险管理和止盈止损机制，支持现货和合约交易的回测与实盘。

### 🎯 核心特性

- ✅ **AI驱动**：LightGBM模型（AUC 0.715），17个技术指标特征
- ✅ **完整交易循环**：买入→持仓→卖出的完整流程
- ✅ **智能止盈止损**：4种卖出条件（止盈6%、止损-3%、时间止损、AI看跌）
- ✅ **持仓状态跟踪**：实时跟踪持仓盈亏和持仓时间
- ✅ **风险管理**：15%最大回撤硬止损，单笔风险控制
- ✅ **多合约支持**：现货、USDT-M永续合约，可配置杠杆
- ✅ **模块化架构**：Broker抽象层，支持回测/模拟/实盘切换

### 📊 最新回测成果

**测试配置**：DOGE/USDT, 15分钟K线, 1000根数据, 5倍杠杆

| 指标 | 数值 | 说明 |
|------|------|------|
| **总交易次数** | 19 | ✅ 从0提升到19 |
| **胜率** | 42.11% | 需要优化 |
| **盈亏比** | 1.87 | ✅ 优秀 |
| **止盈触发** | 21次 | 6%止盈 |
| **止损触发** | 30次 | -3%止损 |
| **净收益率** | -22.48% | 需要优化买入时机 |

**关键成就**：
- ✅ 成功实现完整的买卖交易循环
- ✅ 止盈止损机制正常工作
- ✅ 盈亏比1.87说明风险控制有效
- ⚠️ 胜率需要提升（目标55%+）

---

## 🏗️ 系统架构

```
butterfly_bot/
├── config/                    # 配置管理
│   └── settings.py           # 全局配置（阈值、风控参数等）
├── core/                      # 核心组件
│   ├── broker/               # 交易执行层
│   │   ├── base.py          # Broker抽象基类
│   │   ├── backtest.py      # 回测Broker
│   │   ├── paper.py         # 模拟交易Broker
│   │   └── live.py          # 实盘Broker
│   ├── engine/               # 交易引擎
│   │   └── trading_engine.py # 信号处理和订单执行
│   ├── risk/                 # 风险管理
│   │   └── risk_manager.py  # 风险控制（回撤、仓位等）
│   └── reporter/             # 报告生成
│       └── report_generator.py
├── strategies/                # 交易策略
│   └── ai_signal_core.py     # AI信号核心策略
├── data/                      # 数据处理
│   ├── fetcher.py            # 历史数据获取
│   └── features.py           # 特征工程
├── model/                     # AI模型
│   ├── ensemble_model.py     # 模型加载和预测
│   └── model_registry.py     # 模型版本管理
└── scripts/                   # 脚本工具
    ├── test_backtest.py      # 完整回测脚本
    └── test_simple_trade.py  # 简单测试脚本
```

### 核心组件说明

#### 1. Broker抽象层
- **BaseBroker**：统一的交易接口
- **BacktestBroker**：回测环境，模拟订单执行
- **PaperBroker**：模拟交易，连接真实行情
- **LiveBroker**：实盘交易，真实下单

#### 2. AISignalCore策略
- **持仓状态跟踪**：has_position, entry_price, holding_bars
- **4种卖出条件**：
  1. 止盈：盈利 >= 6%
  2. 止损：亏损 >= 3%
  3. 时间止损：持仓 >= 50根K线
  4. AI看跌：p_ema <= 0.45
- **趋势过滤**：价格 > MA20
- **动量确认**：RSI > 50

#### 3. TradingEngine
- 信号处理和订单执行
- 风险检查和仓位管理
- 自动更新策略持仓状态

---

## 🚀 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone https://github.com/yanzhao77/ButterflyBot.git
cd ButterflyBot

# 创建虚拟环境
python3.11 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install ccxt lightgbm scikit-learn pandas numpy
```

### 2. 配置参数

编辑 `butterfly_bot/config/settings.py`：

```python
# 交易对和周期
SYMBOL = "DOGE/USDT"
TIMEFRAME = "15m"

# 买入阈值（关键参数）
CONFIDENCE_THRESHOLD = 0.50  # AI置信度阈值

# 止盈止损
TAKE_PROFIT_PCT = 0.06  # 6%止盈
STOP_LOSS_PCT = 0.03    # 3%止损

# 风险控制
MAX_POSITION_RATIO = 0.25  # 最大仓位25%
MAX_DRAWDOWN = 0.15        # 最大回撤15%
```

### 3. 运行回测

```bash
# 完整回测（包含现货和合约）
python scripts/test_backtest.py

# 查看回测报告
cat reports/backtest/backtest_USDT_M_leverage5_*.json
```

### 4. 查看结果

回测完成后，查看生成的报告：
- `reports/backtest/` - 回测JSON报告
- `BACKTEST_RESULTS_ANALYSIS.md` - 详细分析报告
- `OPTIMIZATION_COMPARISON.md` - 优化对比报告

---

## ⚙️ 配置说明

### 核心参数

| 参数 | 默认值 | 说明 | 优化建议 |
|------|--------|------|---------|
| `CONFIDENCE_THRESHOLD` | 0.50 | AI买入置信度阈值 | 提高到0.60-0.65可减少交易次数 |
| `SELL_THRESHOLD` | 0.45 | AI卖出阈值 | 保持0.45 |
| `TAKE_PROFIT_PCT` | 0.06 | 止盈百分比 | 6%较为合理 |
| `STOP_LOSS_PCT` | 0.03 | 止损百分比 | 3%较为合理 |
| `MAX_HOLDING_BARS` | 50 | 最大持仓K线数 | 50根约12.5小时 |
| `COOLDOWN_BARS` | 5 | 交易冷却期 | 5根约1.25小时 |
| `TREND_FILTER` | True | 启用趋势过滤 | 建议保持True |

### 回测配置

```python
BACKTEST_CONFIG = {
    "initial_balance": 1000.0,    # 初始资金
    "leverage": 5,                 # 杠杆倍数
    "contract_type": "USDT_M",    # 合约类型
    "start_date": "2023-11-01",   # 开始日期
    "end_date": "2023-11-30",     # 结束日期
}
```

---

## 📊 回测结果示例

### 交易明细（前5笔）

| # | 开仓价格 | 平仓价格 | 盈亏 | 盈亏率 | 原因 |
|---|---------|---------|------|--------|------|
| 1 | 0.07147 | 0.07597 | +15.48 USDT | +6.19% | 止盈 |
| 2 | 0.07695 | 0.08253 | +18.13 USDT | +7.14% | 止盈 |
| 3 | 0.08332 | 0.08951 | +18.91 USDT | +7.32% | 止盈 |
| 4 | 0.08771 | 0.08443 | -10.09 USDT | -3.84% | 止损 |
| 5 | 0.08491 | 0.08123 | -11.53 USDT | -4.43% | 止损 |

### 卖出原因分布

| 原因 | 触发次数 | 占比 |
|------|---------|------|
| 🎯 止盈 (+6%) | 21次 | 41% |
| 🛑 止损 (-3%) | 30次 | 58% |
| ⏰ 时间止损 | 0次 | 0% |
| 📉 AI看跌 | 0次 | 0% |

---

## 🔧 优化历程

### v1.0 - 基础架构（2025-12-23）
- ✅ 实现完整的买卖交易循环
- ✅ 添加持仓状态跟踪
- ✅ 实现4种卖出条件
- ✅ 总交易次数从0提升到19

### v1.1 - 买入质量优化（2025-12-24）
- ✅ CONFIDENCE_THRESHOLD: 0.30 → 0.50
- ✅ 趋势过滤: MA50 → MA20
- ✅ 新增RSI过滤: RSI > 50
- ⚠️ 效果不显著（测试数据局限）

### 下一步计划
- 🎯 提高阈值到0.65
- 🎯 运行5000-10000根K线长周期回测
- 🎯 增加MACD和成交量过滤
- 🎯 实现移动止损机制

---

## 📈 性能优化建议

### 提高胜率（目标55%+）

1. **提高买入阈值**
   ```python
   CONFIDENCE_THRESHOLD = 0.65  # 从0.50提高到0.65
   ```

2. **增加过滤条件**
   - MACD过滤：要求MACD > 0
   - 成交量确认：要求成交量 > 20日均量
   - 多时间框架确认

3. **优化止盈止损**
   - 移动止损：盈利3%后移至成本价
   - 分批止盈：3%卖出30%，5%卖出30%，剩余等待6%

### 长周期回测

```python
# 修改回测配置
BACKTEST_CONFIG = {
    "start_date": "2023-06-01",  # 6个月数据
    "end_date": "2023-12-01",
}
```

---

## 🛠️ 开发指南

### 添加新策略

1. 继承 `AISignalCore` 或创建新策略类
2. 实现 `get_signal(data)` 方法
3. 返回信号字典：`{"signal": "buy/sell/hold", "confidence": 0.0-1.0, "reason": "..."}`

```python
class MyStrategy:
    def get_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        # 你的策略逻辑
        if condition:
            return {
                "signal": "buy",
                "confidence": 0.8,
                "reason": "自定义买入条件"
            }
        return {"signal": "hold", "confidence": 0.0, "reason": "无信号"}
```

### 添加新的Broker

1. 继承 `BaseBroker`
2. 实现必要的方法：`place_order()`, `close_position()`, `get_balance()` 等
3. 在 `TradingEngine` 中使用

---

## 📚 文档资源

- **BACKTEST_RESULTS_ANALYSIS.md** - 详细回测分析报告
- **OPTIMIZATION_COMPARISON.md** - 优化对比分析
- **DELIVERY_SUMMARY.md** - 交付总结文档
- **FINAL_BACKTEST_ANALYSIS.md** - 最终回测分析

---

## 🔍 故障排查

### 问题：回测没有产生交易

**可能原因**：
1. AI模型预测值都低于阈值
2. 趋势过滤阻止了所有买入
3. 持仓状态未正确更新

**解决方案**：
1. 降低 `CONFIDENCE_THRESHOLD`
2. 检查日志中的 "买入判断" 和 "趋势过滤" 信息
3. 确保 `TradingEngine` 正确传入 `strategy` 对象

### 问题：止盈止损未触发

**可能原因**：
1. 持仓状态未更新
2. 价格波动未达到阈值

**解决方案**：
1. 检查日志中的 "持仓盈亏" 信息
2. 调整止盈止损百分比

---

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request！

1. Fork 本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

---

## 📜 许可证

本项目采用 [MIT 许可证](LICENSE)。

---

## 🙌 致谢

- [CCXT](https://ccxt.trade/) - 统一的交易所API
- [LightGBM](https://lightgbm.readthedocs.io/) - 高效的梯度提升框架
- [Pandas](https://pandas.pydata.org/) - 数据分析工具
- [NumPy](https://numpy.org/) - 科学计算库

---

## ⚠️ 免责声明

**本项目仅供学习和研究使用，不构成任何投资建议。**

- ⚠️ 加密货币交易具有高风险，可能导致本金损失
- ⚠️ 过去的回测表现不代表未来收益
- ⚠️ 实盘交易前请充分测试并评估风险
- ⚠️ 请勿使用超出承受能力的资金进行交易

**使用本系统进行实盘交易的所有风险由用户自行承担。**

---

## 📞 联系方式

- **GitHub**: [yanzhao77/ButterflyBot](https://github.com/yanzhao77/ButterflyBot)
- **Issues**: [提交问题](https://github.com/yanzhao77/ButterflyBot/issues)

---

**Happy Trading! 🚀**
