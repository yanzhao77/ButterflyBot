# ButterflyBot 重构架构设计

## 🎯 重构目标

1. **生产级架构** - 可部署在服务器上持续运行
2. **Broker抽象层** - 统一接口，支持回测/模拟/实盘切换
3. **永续合约支持** - USDT-M / COIN-M 永续合约
4. **杠杆交易** - 可配置杠杆倍数
5. **硬性止损** - 账户回撤15%自动暂停
6. **官方SDK** - binance-connector + binance-futures-connector
7. **完整测试** - 回测和实盘测试
8. **详细报告** - 交易频率、准确率、最大回撤分析

---

## 📐 新架构设计

### 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                    ButterflyBot v2.0                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌───────────────────────────────────────────────────────┐ │
│  │              Trading Engine (交易引擎)                 │ │
│  ├───────────────────────────────────────────────────────┤ │
│  │  - 策略执行                                           │ │
│  │  - 信号生成                                           │ │
│  │  - 订单管理                                           │ │
│  │  - 状态持久化                                         │ │
│  └───────────────────────────────────────────────────────┘ │
│                          ↓↑                                 │
│  ┌───────────────────────────────────────────────────────┐ │
│  │              Broker Interface (经纪商接口)             │ │
│  ├───────────────────────────────────────────────────────┤ │
│  │  抽象层：统一接口                                      │ │
│  │  - get_balance()                                      │ │
│  │  - get_position()                                     │ │
│  │  │  - place_order()                                     │ │
│  │  - get_klines()                                       │ │
│  │  - set_leverage()                                     │ │
│  └───────────────────────────────────────────────────────┘ │
│           ↓                ↓                ↓               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │  Backtest   │  │   Paper     │  │    Live     │        │
│  │   Broker    │  │   Broker    │  │   Broker    │        │
│  ├─────────────┤  ├─────────────┤  ├─────────────┤        │
│  │ 历史数据    │  │ 模拟账户    │  │ 真实API     │        │
│  │ 模拟撮合    │  │ 实时数据    │  │ 实时交易    │        │
│  │ 无延迟      │  │ 模拟延迟    │  │ 真实延迟    │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
│                                                             │
│  ┌───────────────────────────────────────────────────────┐ │
│  │           Risk Manager (风险管理器)                    │ │
│  ├───────────────────────────────────────────────────────┤ │
│  │  - 账户回撤监控（硬性止损15%）                         │ │
│  │  - 单笔风险控制                                        │ │
│  │  - 杠杆倍数限制                                        │ │
│  │  - 持仓比例限制                                        │ │
│  │  - 连续亏损保护                                        │ │
│  └───────────────────────────────────────────────────────┘ │
│                                                             │
│  ┌───────────────────────────────────────────────────────┐ │
│  │           Reporter (报告生成器)                        │ │
│  ├───────────────────────────────────────────────────────┤ │
│  │  - 交易频率分析                                        │ │
│  │  - 预测准确率统计                                      │ │
│  │  - 最大回撤计算                                        │ │
│  │  - 盈亏比分析                                          │ │
│  │  - 做多/做空分析                                       │ │
│  │  - 杠杆效果分析                                        │ │
│  └───────────────────────────────────────────────────────┘ │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔧 核心模块设计

### 1. Broker 抽象层

**目的：** 统一回测、模拟、实盘的接口，方便切换和测试

```python
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from enum import Enum

class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"

class ContractType(Enum):
    SPOT = "spot"              # 现货
    USDT_M = "usdt_m"          # USDT本位永续
    COIN_M = "coin_m"          # 币本位永续

class BaseBroker(ABC):
    """经纪商抽象基类"""
    
    @abstractmethod
    def get_balance(self, asset: str = "USDT") -> float:
        """获取余额"""
        pass
    
    @abstractmethod
    def get_position(self, symbol: str) -> Dict:
        """获取持仓
        Returns:
            {
                'size': float,        # 持仓数量（正=多，负=空）
                'entry_price': float, # 开仓均价
                'leverage': int,      # 杠杆倍数
                'unrealized_pnl': float,  # 未实现盈亏
            }
        """
        pass
    
    @abstractmethod
    def place_order(
        self,
        symbol: str,
        side: OrderSide,
        amount: float,
        order_type: OrderType = OrderType.MARKET,
        price: Optional[float] = None
    ) -> Dict:
        """下单
        Returns:
            {
                'order_id': str,
                'filled_price': float,
                'filled_amount': float,
                'fee': float,
            }
        """
        pass
    
    @abstractmethod
    def get_klines(
        self,
        symbol: str,
        interval: str,
        limit: int = 500
    ) -> pd.DataFrame:
        """获取K线数据"""
        pass
    
    @abstractmethod
    def set_leverage(self, symbol: str, leverage: int):
        """设置杠杆倍数（仅永续合约）"""
        pass
    
    @abstractmethod
    def close_position(self, symbol: str):
        """平仓"""
        pass
```

### 2. Broker 实现类

#### BacktestBroker（回测）

```python
class BacktestBroker(BaseBroker):
    """回测经纪商
    
    特点：
    - 使用历史数据
    - 模拟订单撮合
    - 无网络延迟
    - 精确计算手续费和滑点
    """
    
    def __init__(
        self,
        initial_balance: float = 1000.0,
        commission: float = 0.001,  # 0.1%
        slippage: float = 0.0005,   # 0.05%
        contract_type: ContractType = ContractType.SPOT
    ):
        self.balance = initial_balance
        self.initial_balance = initial_balance
        self.positions = {}
        self.orders = []
        self.trades = []
        self.commission = commission
        self.slippage = slippage
        self.contract_type = contract_type
        self.leverage = 1
        
        # 历史数据
        self.klines_data = {}
        self.current_index = 0
    
    def load_historical_data(self, symbol: str, df: pd.DataFrame):
        """加载历史数据"""
        self.klines_data[symbol] = df
    
    def step(self):
        """推进一个时间步（用于回测循环）"""
        self.current_index += 1
    
    def get_current_price(self, symbol: str) -> float:
        """获取当前价格"""
        df = self.klines_data[symbol]
        return df.iloc[self.current_index]['close']
    
    def place_order(self, symbol, side, amount, order_type=OrderType.MARKET, price=None):
        """模拟下单"""
        current_price = self.get_current_price(symbol)
        
        # 模拟滑点
        if side == OrderSide.BUY:
            filled_price = current_price * (1 + self.slippage)
        else:
            filled_price = current_price * (1 - self.slippage)
        
        # 计算手续费
        cost = filled_price * amount
        fee = cost * self.commission
        
        # 更新余额和持仓
        if side == OrderSide.BUY:
            required = cost + fee
            if self.contract_type == ContractType.SPOT:
                if required > self.balance:
                    raise ValueError(f"余额不足: {self.balance} < {required}")
                self.balance -= required
            else:  # 永续合约
                margin = required / self.leverage
                if margin > self.balance:
                    raise ValueError(f"保证金不足: {self.balance} < {margin}")
                self.balance -= fee  # 只扣手续费
            
            # 更新持仓
            if symbol not in self.positions:
                self.positions[symbol] = {
                    'size': 0,
                    'entry_price': 0,
                    'leverage': self.leverage
                }
            
            pos = self.positions[symbol]
            total_cost = pos['size'] * pos['entry_price'] + amount * filled_price
            pos['size'] += amount
            pos['entry_price'] = total_cost / pos['size'] if pos['size'] > 0 else 0
        
        else:  # SELL
            # 平仓逻辑
            if symbol in self.positions:
                pos = self.positions[symbol]
                if amount > pos['size']:
                    amount = pos['size']
                
                # 计算盈亏
                if self.contract_type == ContractType.SPOT:
                    pnl = (filled_price - pos['entry_price']) * amount - fee
                    self.balance += filled_price * amount - fee
                else:  # 永续合约
                    pnl = (filled_price - pos['entry_price']) * amount * self.leverage - fee
                    self.balance += pnl
                
                pos['size'] -= amount
                if pos['size'] <= 0:
                    del self.positions[symbol]
        
        # 记录交易
        trade = {
            'timestamp': self.klines_data[symbol].index[self.current_index],
            'symbol': symbol,
            'side': side.value,
            'amount': amount,
            'price': filled_price,
            'fee': fee,
            'balance': self.balance
        }
        self.trades.append(trade)
        
        return {
            'order_id': f"backtest_{len(self.trades)}",
            'filled_price': filled_price,
            'filled_amount': amount,
            'fee': fee
        }
```

#### PaperBroker（模拟盘）

```python
class PaperBroker(BaseBroker):
    """模拟盘经纪商
    
    特点：
    - 使用实时数据
    - 模拟账户
    - 模拟延迟
    - 不消耗真实资金
    """
    
    def __init__(
        self,
        initial_balance: float = 1000.0,
        contract_type: ContractType = ContractType.SPOT
    ):
        self.balance = initial_balance
        self.initial_balance = initial_balance
        self.positions = {}
        self.orders = []
        self.trades = []
        self.contract_type = contract_type
        self.leverage = 1
        
        # 使用真实API获取数据（但不交易）
        from binance.spot import Spot
        self.client = Spot()
    
    def get_klines(self, symbol, interval, limit=500):
        """获取实时K线"""
        # 转换符号格式
        binance_symbol = symbol.replace('/', '')
        klines = self.client.klines(binance_symbol, interval, limit=limit)
        
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('timestamp')
        df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
        
        return df
    
    def place_order(self, symbol, side, amount, order_type=OrderType.MARKET, price=None):
        """模拟下单（不执行真实交易）"""
        # 获取当前价格
        binance_symbol = symbol.replace('/', '')
        ticker = self.client.ticker_price(binance_symbol)
        current_price = float(ticker['price'])
        
        # 模拟滑点和手续费
        slippage = 0.0005
        commission = 0.001
        
        if side == OrderSide.BUY:
            filled_price = current_price * (1 + slippage)
        else:
            filled_price = current_price * (1 - slippage)
        
        cost = filled_price * amount
        fee = cost * commission
        
        # 更新模拟账户（逻辑同BacktestBroker）
        # ... (省略，与BacktestBroker相同)
        
        return {
            'order_id': f"paper_{len(self.trades)}",
            'filled_price': filled_price,
            'filled_amount': amount,
            'fee': fee
        }
```

#### LiveBroker（实盘）

```python
class LiveBroker(BaseBroker):
    """实盘经纪商
    
    特点：
    - 真实API
    - 真实交易
    - 真实延迟
    - 消耗真实资金
    """
    
    def __init__(
        self,
        api_key: str,
        api_secret: str,
        contract_type: ContractType = ContractType.SPOT,
        testnet: bool = False
    ):
        self.api_key = api_key
        self.api_secret = api_secret
        self.contract_type = contract_type
        self.testnet = testnet
        
        # 根据合约类型选择客户端
        if contract_type == ContractType.SPOT:
            from binance.spot import Spot
            self.client = Spot(
                api_key=api_key,
                api_secret=api_secret,
                base_url="https://testnet.binance.vision" if testnet else None
            )
        elif contract_type == ContractType.USDT_M:
            from binance.um_futures import UMFutures
            self.client = UMFutures(
                key=api_key,
                secret=api_secret,
                base_url="https://testnet.binancefuture.com" if testnet else None
            )
        elif contract_type == ContractType.COIN_M:
            from binance.cm_futures import CMFutures
            self.client = CMFutures(
                key=api_key,
                secret=api_secret
            )
    
    def get_balance(self, asset="USDT"):
        """获取真实余额"""
        if self.contract_type == ContractType.SPOT:
            account = self.client.account()
            for balance in account['balances']:
                if balance['asset'] == asset:
                    return float(balance['free'])
        else:  # 永续合约
            account = self.client.account()
            for asset_info in account['assets']:
                if asset_info['asset'] == asset:
                    return float(asset_info['availableBalance'])
        return 0.0
    
    def get_position(self, symbol):
        """获取真实持仓"""
        if self.contract_type == ContractType.SPOT:
            # 现货没有持仓概念
            base = symbol.split('/')[0]
            balance = self.get_balance(base)
            return {
                'size': balance,
                'entry_price': 0,
                'leverage': 1,
                'unrealized_pnl': 0
            }
        else:  # 永续合约
            binance_symbol = symbol.replace('/', '')
            positions = self.client.get_position_risk(symbol=binance_symbol)
            for pos in positions:
                if pos['symbol'] == binance_symbol:
                    return {
                        'size': float(pos['positionAmt']),
                        'entry_price': float(pos['entryPrice']),
                        'leverage': int(pos['leverage']),
                        'unrealized_pnl': float(pos['unRealizedProfit'])
                    }
        return {'size': 0, 'entry_price': 0, 'leverage': 1, 'unrealized_pnl': 0}
    
    def place_order(self, symbol, side, amount, order_type=OrderType.MARKET, price=None):
        """执行真实交易"""
        binance_symbol = symbol.replace('/', '')
        
        params = {
            'symbol': binance_symbol,
            'side': side.value.upper(),
            'type': order_type.value.upper(),
            'quantity': amount
        }
        
        if order_type == OrderType.LIMIT and price:
            params['price'] = price
            params['timeInForce'] = 'GTC'
        
        # 执行订单
        if self.contract_type == ContractType.SPOT:
            order = self.client.new_order(**params)
        else:  # 永续合约
            order = self.client.new_order(**params)
        
        return {
            'order_id': order['orderId'],
            'filled_price': float(order.get('avgPrice', price or 0)),
            'filled_amount': float(order.get('executedQty', amount)),
            'fee': 0  # 需要单独查询
        }
    
    def set_leverage(self, symbol, leverage):
        """设置杠杆（仅永续合约）"""
        if self.contract_type in [ContractType.USDT_M, ContractType.COIN_M]:
            binance_symbol = symbol.replace('/', '')
            self.client.change_leverage(symbol=binance_symbol, leverage=leverage)
```

---

### 3. RiskManager（风险管理器）

```python
class RiskManager:
    """风险管理器
    
    功能：
    1. 账户回撤监控（硬性止损）
    2. 单笔风险控制
    3. 杠杆倍数限制
    4. 持仓比例限制
    5. 连续亏损保护
    """
    
    def __init__(
        self,
        initial_balance: float,
        max_drawdown_pct: float = 0.15,  # 最大回撤15%
        max_position_ratio: float = 0.25,  # 最大仓位25%
        max_leverage: int = 5,  # 最大杠杆5倍
        max_consecutive_losses: int = 5  # 最大连续亏损5次
    ):
        self.initial_balance = initial_balance
        self.peak_balance = initial_balance
        self.current_balance = initial_balance
        
        self.max_drawdown_pct = max_drawdown_pct
        self.max_position_ratio = max_position_ratio
        self.max_leverage = max_leverage
        self.max_consecutive_losses = max_consecutive_losses
        
        self.consecutive_losses = 0
        self.is_paused = False
        self.pause_reason = ""
    
    def update_balance(self, balance: float):
        """更新余额"""
        self.current_balance = balance
        if balance > self.peak_balance:
            self.peak_balance = balance
    
    def get_current_drawdown(self) -> float:
        """计算当前回撤"""
        return (self.peak_balance - self.current_balance) / self.peak_balance
    
    def check_hard_stop(self) -> bool:
        """检查硬性止损"""
        drawdown = self.get_current_drawdown()
        
        if drawdown >= self.max_drawdown_pct:
            self.is_paused = True
            self.pause_reason = f"账户回撤{drawdown:.2%}超过限制{self.max_drawdown_pct:.2%}"
            return True
        
        return False
    
    def check_position_size(self, position_value: float) -> bool:
        """检查仓位大小"""
        ratio = position_value / self.current_balance
        if ratio > self.max_position_ratio:
            return False
        return True
    
    def check_leverage(self, leverage: int) -> bool:
        """检查杠杆倍数"""
        return leverage <= self.max_leverage
    
    def record_trade_result(self, pnl: float):
        """记录交易结果"""
        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
        
        if self.consecutive_losses >= self.max_consecutive_losses:
            self.is_paused = True
            self.pause_reason = f"连续亏损{self.consecutive_losses}次"
    
    def can_trade(self) -> tuple[bool, str]:
        """是否可以交易"""
        if self.is_paused:
            return False, self.pause_reason
        
        if self.check_hard_stop():
            return False, self.pause_reason
        
        return True, ""
```

---

### 4. Reporter（报告生成器）

```python
class Reporter:
    """报告生成器
    
    生成详细的交易报告，包括：
    - 交易频率分析
    - 预测准确率统计
    - 最大回撤计算
    - 盈亏比分析
    - 做多/做空分析
    - 杠杆效果分析
    """
    
    def __init__(self, trades: List[Dict], initial_balance: float):
        self.trades = pd.DataFrame(trades)
        self.initial_balance = initial_balance
    
    def generate_report(self) -> Dict:
        """生成完整报告"""
        report = {
            'summary': self._summary_stats(),
            'frequency': self._frequency_analysis(),
            'accuracy': self._accuracy_analysis(),
            'drawdown': self._drawdown_analysis(),
            'profit_loss': self._profit_loss_analysis(),
            'direction': self._direction_analysis(),
            'leverage': self._leverage_analysis()
        }
        
        return report
    
    def _summary_stats(self) -> Dict:
        """总体统计"""
        final_balance = self.trades['balance'].iloc[-1]
        total_return = (final_balance - self.initial_balance) / self.initial_balance
        
        return {
            'initial_balance': self.initial_balance,
            'final_balance': final_balance,
            'total_return': total_return,
            'total_trades': len(self.trades),
            'winning_trades': len(self.trades[self.trades['pnl'] > 0]),
            'losing_trades': len(self.trades[self.trades['pnl'] < 0]),
            'win_rate': len(self.trades[self.trades['pnl'] > 0]) / len(self.trades)
        }
    
    def _frequency_analysis(self) -> Dict:
        """交易频率分析"""
        # 计算交易间隔
        self.trades['timestamp'] = pd.to_datetime(self.trades['timestamp'])
        intervals = self.trades['timestamp'].diff().dt.total_seconds() / 3600  # 小时
        
        return {
            'avg_interval_hours': intervals.mean(),
            'median_interval_hours': intervals.median(),
            'min_interval_hours': intervals.min(),
            'max_interval_hours': intervals.max(),
            'trades_per_day': len(self.trades) / (intervals.sum() / 24)
        }
    
    def _accuracy_analysis(self) -> Dict:
        """预测准确率分析"""
        # 假设trades中有'predicted'和'actual'字段
        if 'predicted' in self.trades.columns and 'actual' in self.trades.columns:
            correct = (self.trades['predicted'] == self.trades['actual']).sum()
            accuracy = correct / len(self.trades)
            
            return {
                'accuracy': accuracy,
                'correct_predictions': correct,
                'total_predictions': len(self.trades)
            }
        return {}
    
    def _drawdown_analysis(self) -> Dict:
        """回撤分析"""
        equity = self.trades['balance']
        peak = equity.expanding().max()
        drawdown = (peak - equity) / peak
        
        max_dd = drawdown.max()
        max_dd_idx = drawdown.idxmax()
        max_dd_date = self.trades.loc[max_dd_idx, 'timestamp']
        
        return {
            'max_drawdown': max_dd,
            'max_drawdown_date': max_dd_date,
            'avg_drawdown': drawdown.mean()
        }
    
    def _profit_loss_analysis(self) -> Dict:
        """盈亏分析"""
        wins = self.trades[self.trades['pnl'] > 0]['pnl']
        losses = self.trades[self.trades['pnl'] < 0]['pnl']
        
        return {
            'avg_win': wins.mean() if len(wins) > 0 else 0,
            'avg_loss': losses.mean() if len(losses) > 0 else 0,
            'profit_factor': abs(wins.sum() / losses.sum()) if len(losses) > 0 else float('inf'),
            'expectancy': self.trades['pnl'].mean()
        }
    
    def _direction_analysis(self) -> Dict:
        """做多/做空分析"""
        longs = self.trades[self.trades['side'] == 'buy']
        shorts = self.trades[self.trades['side'] == 'sell']
        
        return {
            'long_trades': len(longs),
            'short_trades': len(shorts),
            'long_win_rate': len(longs[longs['pnl'] > 0]) / len(longs) if len(longs) > 0 else 0,
            'short_win_rate': len(shorts[shorts['pnl'] > 0]) / len(shorts) if len(shorts) > 0 else 0,
            'long_pnl': longs['pnl'].sum() if len(longs) > 0 else 0,
            'short_pnl': shorts['pnl'].sum() if len(shorts) > 0 else 0
        }
    
    def _leverage_analysis(self) -> Dict:
        """杠杆效果分析"""
        if 'leverage' in self.trades.columns:
            return {
                'avg_leverage': self.trades['leverage'].mean(),
                'max_leverage': self.trades['leverage'].max(),
                'leverage_distribution': self.trades['leverage'].value_counts().to_dict()
            }
        return {}
```

---

## 📂 新目录结构

```
ButterflyBot/
├── core/                          # 核心模块
│   ├── __init__.py
│   ├── broker/                    # Broker抽象层
│   │   ├── __init__.py
│   │   ├── base.py               # BaseBroker抽象类
│   │   ├── backtest.py           # BacktestBroker
│   │   ├── paper.py              # PaperBroker
│   │   └── live.py               # LiveBroker
│   ├── engine/                    # 交易引擎
│   │   ├── __init__.py
│   │   ├── trading_engine.py     # 主引擎
│   │   └── order_manager.py      # 订单管理
│   ├── risk/                      # 风险管理
│   │   ├── __init__.py
│   │   └── risk_manager.py       # RiskManager
│   └── reporter/                  # 报告生成
│       ├── __init__.py
│       └── reporter.py            # Reporter
│
├── strategies/                    # 策略模块
│   ├── __init__.py
│   ├── ai_signal_core.py         # AI信号策略（保留）
│   └── base_strategy.py          # 策略基类
│
├── model/                         # 模型模块（保留）
│   ├── __init__.py
│   ├── lgb_model.py
│   └── train_balanced.py
│
├── data/                          # 数据模块（保留）
│   ├── __init__.py
│   ├── features.py
│   └── fetcher.py
│
├── config/                        # 配置模块
│   ├── __init__.py
│   ├── settings.py               # 基础配置
│   ├── backtest_config.py        # 回测配置
│   ├── paper_config.py           # 模拟盘配置
│   └── live_config.py            # 实盘配置
│
├── scripts/                       # 运行脚本
│   ├── run_backtest.py           # 运行回测
│   ├── run_paper.py              # 运行模拟盘
│   └── run_live.py               # 运行实盘
│
├── tests/                         # 测试模块
│   ├── test_broker.py
│   ├── test_risk_manager.py
│   └── test_reporter.py
│
├── reports/                       # 报告输出
│   ├── backtest/
│   ├── paper/
│   └── live/
│
├── requirements.txt
└── README.md
```

---

## 🚀 使用流程

### 1. 回测

```python
from core.broker.backtest import BacktestBroker, ContractType
from core.engine.trading_engine import TradingEngine
from strategies.ai_signal_core import AISignalCore

# 创建回测Broker
broker = BacktestBroker(
    initial_balance=1000,
    contract_type=ContractType.USDT_M,  # 使用USDT本位永续
    commission=0.001,
    slippage=0.0005
)

# 设置杠杆
broker.set_leverage("DOGE/USDT", 5)

# 加载历史数据
df = pd.read_csv('data.csv')
broker.load_historical_data("DOGE/USDT", df)

# 创建策略
strategy = AISignalCore(...)

# 创建交易引擎
engine = TradingEngine(broker, strategy)

# 运行回测
engine.run_backtest()

# 生成报告
report = engine.generate_report()
print(report)
```

### 2. 模拟盘

```python
from core.broker.paper import PaperBroker

# 创建模拟盘Broker
broker = PaperBroker(
    initial_balance=1000,
    contract_type=ContractType.USDT_M
)

# 其余同回测
engine = TradingEngine(broker, strategy)
engine.run_live()  # 持续运行
```

### 3. 实盘

```python
from core.broker.live import LiveBroker

# 创建实盘Broker
broker = LiveBroker(
    api_key="xxx",
    api_secret="xxx",
    contract_type=ContractType.USDT_M,
    testnet=False  # 使用真实环境
)

# 其余同模拟盘
engine = TradingEngine(broker, strategy)
engine.run_live()
```

---

## ✅ 下一步实施计划

1. **Phase 1：核心模块实现**
   - Broker抽象层
   - RiskManager
   - Reporter

2. **Phase 2：引擎实现**
   - TradingEngine
   - OrderManager

3. **Phase 3：测试**
   - 单元测试
   - 集成测试
   - 回测验证

4. **Phase 4：部署**
   - Docker容器化
   - 监控告警
   - 日志系统

5. **Phase 5：优化**
   - 性能优化
   - 策略优化
   - 风控优化
