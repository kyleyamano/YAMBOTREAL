import pandas as pd
import numpy as np

print("Loading trades...")
trades = pd.read_csv("results/trades.csv")

if len(trades) == 0:
    print("No trades found.")
    exit()

print("\nTotal Trades:", len(trades))
print("Win Rate:", (trades["net_pnl"] > 0).mean())

avg_loss = abs(trades[trades["net_pnl"] < 0]["net_pnl"].mean())

if np.isnan(avg_loss):
    print("No losing trades found.")
    exit()

trades["R"] = trades["net_pnl"] / avg_loss

print("\n===== R MULTIPLE DISTRIBUTION =====")
print(">= 1R:", (trades["R"] >= 1).mean())
print(">= 2R:", (trades["R"] >= 2).mean())
print(">= 3R:", (trades["R"] >= 3).mean())
print(">= 4R:", (trades["R"] >= 4).mean())
print(">= 5R:", (trades["R"] >= 5).mean())

print("\n===== WINNER TAIL =====")
winners = trades[trades["net_pnl"] > 0]

print("Median Winner:", winners["net_pnl"].median())
print("90th Percentile Winner:", winners["net_pnl"].quantile(0.90))
print("95th Percentile Winner:", winners["net_pnl"].quantile(0.95))
print("Max Winner:", winners["net_pnl"].max())

if "bars_held" in trades.columns:
    print("\n===== HOLD TIME =====")
    print("Avg Bars Held (Winners):", winners["bars_held"].mean())
    print("Avg Bars Held (Losers):", trades[trades["net_pnl"] < 0]["bars_held"].mean())
