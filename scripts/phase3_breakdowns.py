import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
from src.data.loader import load_parquet
from src.research.regimes import add_regime_labels


def main():

    print("Loading trades...")
    trades = pd.read_csv("results/trades.csv", parse_dates=["entry_time", "exit_time"])

    print("Loading market data...")
    data = load_parquet("data/raw/mnq_1m_full.parquet")

    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index, utc=True)

    # ADD REGIME LABELS HERE
    print("Recomputing regime labels...")
    data = add_regime_labels(data)

    data = data.reset_index().rename(columns={"index": "datetime"})

    # Merge regime onto trades
    trades = trades.merge(
        data[["datetime", "regime"]],
        left_on="entry_time",
        right_on="datetime",
        how="left"
    )

    print("\n==============================")
    print("PnL by Regime")
    print("==============================")
    print(trades.groupby("regime")["net_pnl"].agg(["count", "sum", "mean"]))

    print("\n==============================")
    print("Long vs Short")
    print("==============================")
    print(trades.groupby("side")["net_pnl"].agg(["count", "sum", "mean"]))

    print("\n==============================")
    print("PnL by Hour")
    print("==============================")

    trades["hour"] = trades["entry_time"].dt.hour
    print(trades.groupby("hour")["net_pnl"].agg(["count", "sum", "mean"]))

    print("\nBreakdown complete.")


if __name__ == "__main__":
    main()
