import pandas as pd

class SimpleMACrossoverStrategy:
    def __init__(self, short_window=20, long_window=50):
        self.short_window = short_window
        self.long_window = long_window

    def get_signal(self, df: pd.DataFrame):
        df = df.copy()
        if len(df) < self.long_window:
            return {"signal": "hold", "confidence": 0.0, "reason": "Data insufficient"}

        df['short_ma'] = df['close'].rolling(window=self.short_window).mean()
        df['long_ma'] = df['close'].rolling(window=self.long_window).mean()

        latest = df.iloc[-1]
        previous = df.iloc[-2]

        print(f"short_ma: {latest['short_ma']}, long_ma: {latest['long_ma']}")

        # Buy signal: short MA crosses above long MA
        if latest['short_ma'] > latest['long_ma'] and previous['short_ma'] <= previous['long_ma']:
            return {"signal": "buy", "confidence": 1.0, "reason": "MA Crossover"}

        # Sell signal: short MA crosses below long MA
        if latest['short_ma'] < latest['long_ma'] and previous['short_ma'] >= previous['long_ma']:
            return {"signal": "sell", "confidence": 1.0, "reason": "MA Crossover"}

        return {"signal": "hold", "confidence": 0.0, "reason": "No crossover"}
