# 阶段1快速启动指南

## 🚀 5分钟快速开始

本指南帮助你快速开始阶段1小资金实盘测试。

---

## 准备工作（30分钟）

### 1. 交易所准备

**Binance账户设置：**

1. 注册Binance账户：https://www.binance.com
2. 完成KYC认证
3. 充值$100-200 USDT
4. 生成API密钥：
   - 登录 → 个人中心 → API管理
   - 创建新的API密钥
   - **重要：** 仅勾选"现货交易"权限，不要勾选"提现"
   - 保存API Key和Secret Key

**安全设置：**
- ✅ 启用双重认证（2FA）
- ✅ 设置API白名单IP（如果有固定IP）
- ✅ 设置API每日提现限额为0
- ❌ 不要将API密钥分享给任何人

### 2. 环境准备

**安装依赖：**

```bash
cd /home/ubuntu/ButterflyBot

# 安装Python依赖
pip3 install ccxt pandas numpy joblib scikit-learn lightgbm

# 验证安装
python3 -c "import ccxt; print('ccxt version:', ccxt.__version__)"
```

**配置API密钥：**

```bash
# 创建环境变量文件
cat > .env << 'EOF'
BINANCE_API_KEY=your_api_key_here
BINANCE_API_SECRET=your_api_secret_here
EOF

# 设置权限（重要！）
chmod 600 .env
```

**测试API连接：**

```bash
python3 << 'EOF'
import ccxt
import os

# 读取API密钥
api_key = os.getenv('BINANCE_API_KEY', 'your_key')
api_secret = os.getenv('BINANCE_API_SECRET', 'your_secret')

# 创建交易所实例
exchange = ccxt.binance({
    'apiKey': api_key,
    'secret': api_secret,
})

# 测试连接
try:
    balance = exchange.fetch_balance()
    print("✅ API连接成功！")
    print(f"USDT余额: {balance['USDT']['free']}")
except Exception as e:
    print(f"❌ API连接失败: {e}")
EOF
```

### 3. 数据准备

**下载最新数据：**

```bash
# 运行数据获取脚本
python3 fetch_1year_data.py

# 验证数据
python3 -c "import pandas as pd; df = pd.read_csv('cached_data/binance_DOGE_USDT_15m.csv'); print(f'✅ {len(df)} rows, latest: {df.iloc[-1][\"timestamp\"]}')"
```

**加载模型：**

```bash
# 检查模型文件
ls -lh models/registry/*balanced.pkl

# 验证模型
python3 -c "import joblib; model = joblib.load('models/registry/v20251126_2157_balanced.pkl'); print('✅ 模型加载成功')"
```

---

## 启动测试（5分钟）

### 方法1：使用监控脚本（推荐）

**创建启动脚本：**

```bash
cat > start_stage1.sh << 'EOF'
#!/bin/bash
# 阶段1启动脚本

echo "=========================================="
echo "阶段1实盘测试启动"
echo "=========================================="

# 加载环境变量
source .env

# 检查环境
echo "检查环境..."
python3 quick_check.sh

# 启动监控
echo "启动监控..."
python3 stage1_monitor.py &

# 启动交易（这里需要你自己实现live_trading.py）
echo "启动交易..."
# python3 live_trading.py

echo "=========================================="
echo "系统已启动！"
echo "监控地址: http://localhost:8000"
echo "日志文件: stage1_trading.log"
echo "=========================================="
EOF

chmod +x start_stage1.sh
```

**启动：**

```bash
./start_stage1.sh
```

### 方法2：手动启动

**步骤：**

1. 打开终端1 - 运行监控
```bash
python3 stage1_monitor.py
```

2. 打开终端2 - 运行交易
```bash
# 这里需要你实现live_trading.py
# python3 live_trading.py
```

3. 打开终端3 - 查看日志
```bash
tail -f stage1_trading.log
```

---

## 每日操作流程

### 早上（08:30-09:00）

**1. 打开每日检查清单：**

```bash
# 查看清单
cat DAILY_CHECKLIST.md

# 或者在浏览器中打开
# 使用Markdown阅读器
```

**2. 执行交易前检查：**

```bash
# 运行快速检查脚本
./quick_check.sh

# 检查账户余额
python3 << 'EOF'
import ccxt
exchange = ccxt.binance({'apiKey': 'xxx', 'secret': 'xxx'})
balance = exchange.fetch_balance()
print(f"USDT: {balance['USDT']['free']}")
print(f"DOGE: {balance['DOGE']['free']}")
EOF

# 查看市场行情
python3 << 'EOF'
import ccxt
exchange = ccxt.binance()
ticker = exchange.fetch_ticker('DOGE/USDT')
print(f"DOGE/USDT: {ticker['last']}")
print(f"24h变化: {ticker['percentage']}%")
print(f"24h成交量: {ticker['quoteVolume']}")
EOF
```

**3. 启动交易系统：**

```bash
./start_stage1.sh
```

### 白天（09:00-18:00）

**每小时检查：**

```bash
# 查看当前状态
python3 << 'EOF'
from stage1_monitor import Stage1Monitor
monitor = Stage1Monitor()

# 获取统计
stats = monitor.get_current_stats()
today = monitor.get_today_stats()

print(f"今日交易: {today['trades']}次")
print(f"今日盈亏: ${today['pnl']:+.2f}")
print(f"总胜率: {stats['win_rate']*100:.1f}%")
print(f"总盈亏: ${stats['total_pnl']:+.2f}")

# 风控检查
risk = monitor.check_risk_control()
if risk['should_pause']:
    print("⚠️ 建议暂停交易！")
else:
    print("✅ 风控正常")
EOF
```

### 晚上（18:00-19:00）

**1. 停止交易：**

```bash
# 停止交易程序
# pkill -f live_trading.py

# 或者手动停止
```

**2. 生成每日报告：**

```bash
python3 << 'EOF'
from stage1_monitor import Stage1Monitor
monitor = Stage1Monitor()
report = monitor.generate_daily_report()
print(report)
EOF
```

**3. 复盘分析：**

- 查看所有交易记录
- 分析盈利/亏损原因
- 识别最佳/最差交易
- 记录经验教训
- 填写交易日志

**4. 备份数据：**

```bash
# 备份到本地
cp -r stage1_data stage1_data_backup_$(date +%Y%m%d)

# 备份到云端（可选）
# rsync -av stage1_data/ user@server:/backup/stage1_data/
```

---

## 每周操作流程

### 周日总结

**1. 生成周报：**

```bash
python3 << 'EOF'
from stage1_monitor import Stage1Monitor
monitor = Stage1Monitor()
report = monitor.generate_weekly_report()
print(report)
EOF
```

**2. 深度分析：**

- 回顾本周所有交易
- 分析盈亏模式
- 评估策略有效性
- 识别改进机会

**3. 参数调整（如需要）：**

```bash
# 编辑配置文件
nano config_stage1.py

# 修改参数，例如：
# CONFIDENCE_THRESHOLD = 0.10  # 如果交易太频繁
# TAKE_PROFIT_PCT = 0.03  # 如果止盈太难触发
```

**4. 进度评估：**

- 对比阶段1目标
- 评估是否达标
- 预测能否进入阶段2
- 制定下周计划

---

## 常见问题

### Q1：如何查看当前持仓？

```bash
python3 << 'EOF'
import ccxt
exchange = ccxt.binance({'apiKey': 'xxx', 'secret': 'xxx'})
balance = exchange.fetch_balance()
doge = balance['DOGE']['free']
if doge > 0:
    print(f"持仓: {doge} DOGE")
else:
    print("无持仓")
EOF
```

### Q2：如何手动平仓？

```bash
python3 << 'EOF'
import ccxt
exchange = ccxt.binance({'apiKey': 'xxx', 'secret': 'xxx'})

# 查看持仓
balance = exchange.fetch_balance()
doge = balance['DOGE']['free']

if doge > 0:
    # 市价卖出
    order = exchange.create_market_sell_order('DOGE/USDT', doge)
    print(f"已平仓: {doge} DOGE")
else:
    print("无持仓")
EOF
```

### Q3：如何暂停交易？

```bash
# 方法1：停止程序
pkill -f live_trading.py

# 方法2：修改配置
# 在config_stage1.py中添加：
# TRADING_ENABLED = False
```

### Q4：如何查看交易记录？

```bash
# 查看JSON文件
cat stage1_data/trades.json | python3 -m json.tool

# 或者使用监控脚本
python3 << 'EOF'
from stage1_monitor import Stage1Monitor
import json
monitor = Stage1Monitor()
print(json.dumps(monitor.trades, indent=2))
EOF
```

### Q5：如何恢复数据？

```bash
# 从备份恢复
cp -r stage1_data_backup_20251128/* stage1_data/

# 验证数据
python3 << 'EOF'
from stage1_monitor import Stage1Monitor
monitor = Stage1Monitor()
print(f"交易记录: {len(monitor.trades)}笔")
print(f"权益记录: {len(monitor.equity_curve)}条")
EOF
```

---

## 应急处理

### 连续亏损3次

```bash
# 1. 立即暂停交易
pkill -f live_trading.py

# 2. 分析原因
python3 << 'EOF'
from stage1_monitor import Stage1Monitor
monitor = Stage1Monitor()
recent = monitor.trades[-5:]
for t in recent:
    print(f"{t['type']} {t['return']*100:+.2f}% ${t['pnl']:+.2f}")
EOF

# 3. 等待1小时后手动恢复
```

### 单日亏损超限

```bash
# 1. 立即停止交易
pkill -f live_trading.py

# 2. 平掉所有持仓（如有）
# 使用Q2的方法

# 3. 生成分析报告
python3 << 'EOF'
from stage1_monitor import Stage1Monitor
monitor = Stage1Monitor()
report = monitor.generate_daily_report()
print(report)
EOF

# 4. 次日决定是否继续
```

### 系统故障

```bash
# 1. 停止所有程序
pkill -f python3

# 2. 检查持仓
# 使用Q1的方法

# 3. 必要时手动平仓
# 使用Q2的方法

# 4. 检查数据完整性
python3 << 'EOF'
from stage1_monitor import Stage1Monitor
monitor = Stage1Monitor()
print(f"✅ 数据完整: {len(monitor.trades)}笔交易")
EOF

# 5. 重新启动
./start_stage1.sh
```

---

## 文件结构

```
ButterflyBot/
├── STAGE1_IMPLEMENTATION_PLAN.md  # 详细实施计划
├── STAGE1_QUICK_START.md          # 本文件
├── DAILY_CHECKLIST.md             # 每日检查清单
├── config_stage1.py               # 阶段1配置
├── stage1_monitor.py              # 监控系统
├── live_trading.py                # 实盘交易（需实现）
├── start_stage1.sh                # 启动脚本
├── quick_check.sh                 # 快速检查
├── .env                           # API密钥（不要提交到Git）
├── stage1_data/                   # 数据目录
│   ├── trades.json                # 交易记录
│   ├── equity.json                # 权益曲线
│   ├── daily.json                 # 每日统计
│   ├── daily_report_*.txt         # 每日报告
│   └── weekly_report_*.txt        # 每周报告
└── stage1_trading.log             # 交易日志
```

---

## 重要提示

### ⚠️ 安全

- **不要** 将API密钥提交到Git
- **不要** 分享API密钥给任何人
- **不要** 设置API提现权限
- **务必** 启用2FA
- **务必** 定期备份数据

### ⚠️ 风控

- **严格** 执行每日检查清单
- **严格** 遵守风控参数
- **严格** 记录所有交易
- **不要** 因短期盈亏改变策略
- **不要** 存在侥幸心理

### ⚠️ 纪律

- **每天** 完成所有检查
- **每笔** 交易都要记录
- **每日** 必须复盘
- **每周** 必须总结
- **保持** 冷静和理性

---

## 联系支持

**遇到问题？**

1. 查看文档：
   - STAGE1_IMPLEMENTATION_PLAN.md
   - DAILY_CHECKLIST.md
   - README.md

2. 检查日志：
   - stage1_trading.log
   - stage1_data/daily_report_*.txt

3. 查看代码：
   - stage1_monitor.py
   - config_stage1.py

---

## 下一步

**测试成功后：**

1. 撰写详细测试报告
2. 总结经验教训
3. 调整阶段2参数
4. 增加资金至$500-1,000
5. 进入阶段2：扩大测试

**测试失败后：**

1. 深度分析失败原因
2. 调整策略或参数
3. 重新进行模拟测试
4. 验证改进效果
5. 决定是否重新开始

---

**祝测试顺利！记住：小资金测试的目的是学习和验证，不是赚大钱。** 🚀

**保持冷静，严格执行，积累经验！** 💪
