import json

class ReportGenerator:
    def __init__(self, broker, risk_manager, engine):
        self.broker = broker
        self.risk_manager = risk_manager
        self.engine = engine

    def generate_report(self):
        report = {
            "initial_balance": self.broker.initial_balance,
            "final_balance": self.broker.get_balance(),
            "trades": self.broker.trades
        }
        return report

    def print_report(self, report):
        print("====== Backtest Report ======")
        print(f'Initial Balance: {report["initial_balance"]}')
        print(f'Final Balance: {report["final_balance"]}')
        print(f'Total Trades: {len(report["trades"])}')

    def save_report(self, report, path):
        with open(path, "w") as f:
            json.dump(report, f, indent=4)
