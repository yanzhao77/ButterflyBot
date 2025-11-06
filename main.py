# main.py (在项目根目录)
"""
量化交易系统主入口
"""

import warnings

warnings.filterwarnings("ignore")

import argparse
import sys
import os

# 添加当前目录到 Python 路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train import train_and_evaluate, main as train_main
from backtest.strategy import AISignalCore
from model.model_registry import load_latest_model


def train_command(args):
    """训练命令"""
    print("🚀 开始训练模型...")
    version, auc = train_and_evaluate(
        symbol=args.symbol,
        timeframe=args.timeframe,
        limit=args.limit,
        since_days=args.since_days
    )
    print(f"✅ 训练完成！版本: {version}, AUC: {auc:.4f}")


def backtest_command(args):
    """回测命令"""
    print("📊 开始回测...")
    # 加载最新模型
    model = load_latest_model()
    if model is None:
        print("❌ 没有找到训练好的模型，请先运行训练命令")
        return

    # 创建策略并运行回测
    strategy = AISignalCore(model=model)
    # 这里添加回测逻辑
    print("✅ 回测完成")


def predict_command(args):
    """预测命令"""
    print("🔮 开始预测...")
    model = load_latest_model()
    if model is None:
        print("❌ 没有找到训练好的模型，请先运行训练命令")
        return

    # 这里添加预测逻辑
    print("✅ 预测完成")


def main():
    parser = argparse.ArgumentParser(description="量化交易系统")
    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # 训练命令
    train_parser = subparsers.add_parser("train", help="训练模型")
    train_parser.add_argument("--symbol", type=str, default="DOGE/USDT", help="交易对")
    train_parser.add_argument("--timeframe", type=str, default="1h", help="K线周期")
    train_parser.add_argument("--limit", type=int, default=10000, help="K线数量")
    train_parser.add_argument("--since_days", type=int, default=365, help="历史天数")

    # 回测命令
    backtest_parser = subparsers.add_parser("backtest", help="运行回测")
    backtest_parser.add_argument("--symbol", type=str, default="DOGE/USDT", help="交易对")
    backtest_parser.add_argument("--timeframe", type=str, default="1h", help="K线周期")
    backtest_parser.add_argument("--period", type=str, default="30d", help="回测周期")

    # 预测命令
    predict_parser = subparsers.add_parser("predict", help="运行预测")
    predict_parser.add_argument("--symbol", type=str, default="DOGE/USDT", help="交易对")
    predict_parser.add_argument("--timeframe", type=str, default="1h", help="K线周期")

    args = parser.parse_args()

    if args.command == "train":
        train_command(args)
    elif args.command == "backtest":
        backtest_command(args)
    elif args.command == "predict":
        predict_command(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()