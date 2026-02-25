import pandas as pd
from .base_strategy import BaseStrategy


class MovingAverageStrategy(BaseStrategy):

    def __init__(self, fast=20, slow=50):
        super().__init__(name="MovingAverage")
        self.fast = fast
        self.slow = slow

    def generate_signals(self, df):

        df = df.copy()

        df["fast_ma"] = df["close"].rolling(self.fast).mean()
        df["slow_ma"] = df["close"].rolling(self.slow).mean()

        df["signal"] = 0
        df.loc[df["fast_ma"] > df["slow_ma"], "signal"] = 1
        df.loc[df["fast_ma"] < df["slow_ma"], "signal"] = -1

        return df
