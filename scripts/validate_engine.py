
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np

from src.backtesting.engine import BacktestEngine, ContractSpecs, ExecutionConfig, RiskConfig, SizingConfig

# --- Dummy Strategy ---
class AlwaysLongOneBar:
    def generate_signals(self, df):
        # Enter long once, then flat
        sig = pd.Series(0, index=df.index)
        sig.iloc[1] = 1
        sig.iloc[2] = 0
        return sig

# --- Synthetic Data ---
dates = pd.date_range("2024-01-01", periods=5, freq="T", tz="UTC")
data = pd.DataFrame({
    "open":  [100, 101, 102, 103, 104],
    "high":  [101, 102, 103, 104, 105],
    "low":   [99, 100, 101, 102, 103],
    "close": [100.5, 101.5, 102.5, 103.5, 104.5],
}, index=dates)

engine = BacktestEngine(
    data=data,
    strategy=AlwaysLongOneBar(),
    initial_capital=10000,
    specs=ContractSpecs(
        tick_size=1.0,
        tick_value=1.0,
        commission_per_side=1.0
    ),
    execution=ExecutionConfig(
        fill_mode="next_open",
        slippage_ticks=1.0
    ),
    risk=RiskConfig(),
    sizing=SizingConfig(
        mode="fixed",
        fixed_qty=1
    ),
    export_artifacts=False
)

result = engine.run()
trades = result["trades"]

print(trades)

# --- Validation Checks ---
assert len(trades) == 1, "Expected exactly one trade."

trade = trades.iloc[0]

# Entry should be at bar 2 open + 1 tick slippage
expected_entry = 102 + 1  # open at index 2 is 102
assert trade["entry_price"] == expected_entry, "Entry fill incorrect."

# Exit should be at bar 3 open - 1 tick slippage (signal exit)
expected_exit = 103 - 1
assert trade["exit_price"] == expected_exit, "Exit fill incorrect."

# Commission applied once per side
assert trade["commission_entry"] == 1.0
assert trade["commission_exit"] == 1.0

print("\nALL TESTS PASSED")
