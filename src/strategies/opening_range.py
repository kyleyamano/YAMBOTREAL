import pandas as pd
import numpy as np
from .base_strategy import BaseStrategy


class OpeningRangeBreakout(BaseStrategy):

    def __init__(self, start="08:30", end="08:45", r_multiple=1.5):
        super().__init__(name="OpeningRangeFiltered")
        self.start = start
        self.end = end
        self.r_multiple = r_multiple

    def generate_signals(self, df):

        df = df.copy()
        df["signal"] = 0
        df["exit"] = 0

        df["date"] = df.index.date

        grouped = df.groupby("date")

        daily_or_sizes = []

        # First pass: compute OR sizes
        for date, group in grouped:

            or_window = group.between_time(self.start, self.end)

            if len(or_window) == 0:
                continue

            or_high = or_window["high"].max()
            or_low = or_window["low"].min()
            or_size = or_high - or_low

            daily_or_sizes.append((date, or_size))

        or_df = pd.DataFrame(daily_or_sizes, columns=["date", "or_size"])
        or_df["rolling_median"] = or_df["or_size"].rolling(20).median()

        or_lookup = dict(zip(or_df["date"], zip(or_df["or_size"], or_df["rolling_median"])))

        # Second pass: generate trades
        for date, group in grouped:

            if date not in or_lookup:
                continue

            or_size, median_size = or_lookup[date]

            if pd.isna(median_size):
                continue

            # Volatility filter
            if or_size <= median_size:
                continue

            or_window = group.between_time(self.start, self.end)
            if len(or_window) == 0:
                continue

            or_high = or_window["high"].max()
            or_low = or_window["low"].min()

            risk = or_high - or_low
            target_long = or_high + risk * self.r_multiple
            target_short = or_low - risk * self.r_multiple

            after_or = group.between_time(self.end, "16:00")

            for idx, row in after_or.iterrows():

                # Long breakout
                if row["close"] > or_high:
                    df.loc[idx, "signal"] = 1
                    break

                # Short breakout
                elif row["close"] < or_low:
                    df.loc[idx, "signal"] = -1
                    break

        return df
