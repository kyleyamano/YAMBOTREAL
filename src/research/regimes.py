import pandas as pd
import numpy as np


def add_regime_labels(df, atr_window=50, slope_window=100):

    data = df.copy()

    # Ensure lowercase columns
    data.columns = [c.lower() for c in data.columns]

    # ATR
    high_low = data["high"] - data["low"]
    high_close = abs(data["high"] - data["close"].shift())
    low_close = abs(data["low"] - data["close"].shift())

    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    data["atr"] = true_range.rolling(atr_window).mean()

    # Volatility percentile
    data["atr_pct"] = data["atr"].rolling(1000).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1]
    )

    # Trend slope
    data["slope"] = (
        data["close"]
        .rolling(slope_window)
        .apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
    )

    # Regime classification
    conditions = [
        (data["atr_pct"] < 0.3) & (abs(data["slope"]) < 0.02),
        (data["atr_pct"] > 0.7) & (abs(data["slope"]) > 0.05),
    ]

    choices = [
        "LOW_VOL_MEAN_REVERT",
        "HIGH_VOL_TREND"
    ]

    data["regime"] = np.select(conditions, choices, default="NEUTRAL")

    return data
