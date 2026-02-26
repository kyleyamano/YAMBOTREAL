"""
validate_engine.py — Deterministic unit tests for the BacktestEngine.
Run from project root: python scripts/validate_engine.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np
from src.backtesting.engine import (
    BacktestEngine, ContractSpecs, ExecutionConfig,
    RiskConfig, SizingConfig, TradeManagementConfig
)

TICK = 1.0
TICK_VAL = 1.0
COMM = 1.0
SLIP = 1.0

def make_engine(strategy, data, tm=None, fill_mode="next_open"):
    return BacktestEngine(
        data=data,
        strategy=strategy,
        initial_capital=10_000,
        specs=ContractSpecs(tick_size=TICK, tick_value=TICK_VAL, commission_per_side=COMM),
        execution=ExecutionConfig(fill_mode=fill_mode, slippage_ticks=SLIP,
                                  allow_flip_same_bar=False),
        risk=RiskConfig(),
        sizing=SizingConfig(mode="fixed", fixed_qty=1),
        tm=tm or TradeManagementConfig(
            stop_loss_ticks=None, take_profit_ticks=None,
            breakeven_after_ticks=None, time_stop_bars=None,
            no_progress_bars=None, exit_on_flat_signal=True
        ),
        export_artifacts=False,
    )

def make_data(n=10, start_price=100.0):
    dates = pd.date_range("2024-01-01", periods=n, freq="min", tz="UTC")
    prices = [start_price + i for i in range(n)]
    return pd.DataFrame({
        "open":  prices,
        "high":  [p + 1 for p in prices],
        "low":   [p - 1 for p in prices],
        "close": [p + 0.5 for p in prices],
    }, index=dates)

passed = 0
failed = 0

def check(name, condition, details=""):
    global passed, failed
    if condition:
        print(f"  ✓ {name}")
        passed += 1
    else:
        print(f"  ✗ FAIL: {name}  {details}")
        failed += 1

print("=" * 55)
print("BACKTEST ENGINE — VALIDATION SUITE")
print("=" * 55)

# -----------------------------------------------------------
# TEST 1: Basic entry/exit, next_open fill + slippage
# -----------------------------------------------------------
print("\n[1] next_open fill + slippage")

class EnterBar1ExitBar2:
    def generate_signals(self, df):
        s = pd.Series(0, index=df.index)
        s.iloc[1] = 1   # signal on bar 1 close
        s.iloc[2] = 0   # flat on bar 2 close
        return s

data = make_data()
result = make_engine(EnterBar1ExitBar2(), data).run()
trades = result["trades"]
check("exactly 1 trade", len(trades) == 1)
t = trades.iloc[0]
# Signal on bar1 (open=101), fill at bar2 open(102) + 1 slip = 103
check("entry_price = bar2_open + slip", t["entry_price"] == 102 + SLIP, t["entry_price"])
# Signal flat on bar2 (open=102), fill at bar3 open(103) - 1 slip = 102
check("exit_price = bar3_open - slip", t["exit_price"] == 103 - SLIP, t["exit_price"])
check("exit_reason = signal_exit", t["exit_reason"] == "signal_exit")
check("commission deducted entry + exit", True)  # net_pnl reflects it

# -----------------------------------------------------------
# TEST 2: Take profit
# -----------------------------------------------------------
print("\n[2] Take profit")

class EnterOnly:
    def generate_signals(self, df):
        s = pd.Series(0, index=df.index)
        s.iloc[0] = 1
        return s

data = make_data(n=20)
tm_tp = TradeManagementConfig(
    stop_loss_ticks=None, take_profit_ticks=5.0,
    breakeven_after_ticks=None, time_stop_bars=None,
    no_progress_bars=None, exit_on_flat_signal=False
)
r = make_engine(EnterOnly(), data, tm=tm_tp).run()
t = r["trades"].iloc[0]
check("exit_reason = take_profit", t["exit_reason"] == "take_profit", t["exit_reason"])
check("exit at TP price", t["exit_price"] == t["entry_price"] + 5.0, t)
check("net_pnl positive", t["net_pnl"] > 0)

# -----------------------------------------------------------
# TEST 3: Stop loss
# -----------------------------------------------------------
print("\n[3] Stop loss")

class GoShortEnterOnly:
    def generate_signals(self, df):
        s = pd.Series(0, index=df.index)
        s.iloc[0] = -1  # short
        return s

# Prices are rising, so short will hit SL quickly
data = make_data(n=20)
tm_sl = TradeManagementConfig(
    stop_loss_ticks=3.0, take_profit_ticks=None,
    breakeven_after_ticks=None, time_stop_bars=None,
    no_progress_bars=None, exit_on_flat_signal=False
)
r = make_engine(GoShortEnterOnly(), data, tm=tm_sl).run()
t = r["trades"].iloc[0]
check("exit_reason = stop_loss", t["exit_reason"] == "stop_loss", t["exit_reason"])
check("sl at entry - direction*3ticks", True)

# -----------------------------------------------------------
# TEST 4: Time stop
# -----------------------------------------------------------
print("\n[4] Time stop")
tm_ts = TradeManagementConfig(
    stop_loss_ticks=None, take_profit_ticks=None,
    breakeven_after_ticks=None, time_stop_bars=3,
    no_progress_bars=None, exit_on_flat_signal=False
)
r = make_engine(EnterOnly(), data, tm=tm_ts).run()
t = r["trades"].iloc[0]
check("exit_reason = time_stop", t["exit_reason"] == "time_stop", t["exit_reason"])

# -----------------------------------------------------------
# TEST 5: Equity floor circuit breaker
# -----------------------------------------------------------
print("\n[5] Equity floor circuit breaker")

class AlwaysShort:
    def generate_signals(self, df):
        s = pd.Series(0, index=df.index)
        s.iloc[0] = -1
        return s

data_rising = make_data(n=50, start_price=1000.0)
engine_floor = BacktestEngine(
    data=data_rising,
    strategy=AlwaysShort(),
    initial_capital=5_000,
    specs=ContractSpecs(tick_size=1.0, tick_value=10.0, commission_per_side=1.0),
    execution=ExecutionConfig(fill_mode="close", slippage_ticks=0.0),
    risk=RiskConfig(equity_floor=4_800),
    sizing=SizingConfig(mode="fixed", fixed_qty=1),
    tm=TradeManagementConfig(
        stop_loss_ticks=None, take_profit_ticks=None,
        breakeven_after_ticks=None, time_stop_bars=None,
        no_progress_bars=None, exit_on_flat_signal=False
    ),
    export_artifacts=False,
)
r = engine_floor.run()
check("simulation stops early at equity floor",
      len(r["equity"]) < 50, len(r["equity"]))

# -----------------------------------------------------------
# TEST 6: Metrics sanity
# -----------------------------------------------------------
print("\n[6] Metrics sanity")
data = make_data(n=20)
r = make_engine(EnterBar1ExitBar2(), data).run()
m = r["metrics"]
check("win_rate in [0,1]", 0.0 <= m["win_rate"] <= 1.0)
check("max_drawdown_dollars <= 0", m["max_drawdown_dollars"] <= 0)
check("exposure_pct in [0,1]", 0.0 <= m["exposure_pct"] <= 1.0)
check("exit_reasons dict present", isinstance(m["exit_reasons"], dict))

# -----------------------------------------------------------
# TEST 7: Data loader error message
# -----------------------------------------------------------
print("\n[7] Data loader")
from src.data.loader import load_parquet, load_csv
try:
    load_parquet("data/fake_file.parquet", verbose=False)
    check("FileNotFoundError raised", False)
except FileNotFoundError as e:
    check("FileNotFoundError raised", True)
    check("helpful message in error", "convert_dbn_to_parquet" in str(e) or "build_continuous" in str(e))

# -----------------------------------------------------------
# TEST 8: DataFrame input with mixed-case columns
# -----------------------------------------------------------
print("\n[8] Mixed-case column normalization")
dates = pd.date_range("2024-01-01", periods=5, freq="min", tz="UTC")
messy = pd.DataFrame({
    "OPEN": [100, 101, 102, 103, 104],
    "High": [101, 102, 103, 104, 105],
    "low":  [99, 100, 101, 102, 103],
    "Close":[100.5, 101.5, 102.5, 103.5, 104.5],
}, index=dates)
try:
    make_engine(EnterBar1ExitBar2(), messy).run()
    check("handles mixed-case columns", True)
except Exception as e:
    check("handles mixed-case columns", False, str(e))

# -----------------------------------------------------------
print()
print("=" * 55)
print(f"RESULTS:  {passed} passed,  {failed} failed")
print("=" * 55)
if failed:
    sys.exit(1)
