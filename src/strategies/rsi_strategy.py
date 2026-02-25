from ta.momentum import RSIIndicator


class RSIStrategy:

    def __init__(self, window=14, overbought=70, oversold=30):
        self.window = window
        self.overbought = overbought
        self.oversold = oversold

    def generate_signals(self, data):

        df = data.copy()

        # Use lowercase column name
        close = df["close"]

        rsi = RSIIndicator(close=close, window=self.window)
        df["rsi"] = rsi.rsi()

        df["signal"] = 0

        # Long when oversold
        df.loc[df["rsi"] < self.oversold, "signal"] = 1

        # Short when overbought
        df.loc[df["rsi"] > self.overbought, "signal"] = -1

        return df
