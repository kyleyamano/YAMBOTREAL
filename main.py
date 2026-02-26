# main.py

from dotenv import load_dotenv
import os
import sys
import traceback
import pandas as pd

# ==================================================
# ENVIRONMENT
# ==================================================

load_dotenv()
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)

# ==================================================
# IMPORTS
# ==================================================

from src.data.loader import load_parquet
from src.strategies.tjr_smc import TJRSMCStrategy

from src.backtesting.engine import (
    BacktestEngine,
    ContractSpecs,
    ExecutionConfig,
    RiskConfig,
    SizingConfig,
    TradeManagementConfig
)

# ==================================================
# GLOBAL CONFIG
# ==================================================

INSTRUMENT = "MNQ"
INITIAL_CAPITAL = 50_000

USE_REGIME_DATA = False

WALK_FORWARD_MODE = False
IN_SAMPLE_END = "2022-12-31"
OUT_SAMPLE_START = "2023-01-01"

PROP_MODE = False

RESULTS_DIR = "results"

# ==================================================
# DATA LOADING
# ==================================================

def load_data():

    if INSTRUMENT == "MNQ":
        path = "data/processed/mnq_continuous_1m.parquet"
    else:
        path = "data/processed/nq_continuous_1m.parquet"

    print(f"\nLoading dataset: {path}")
    df = load_parquet(path)

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)

    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    if df.index.duplicated().any():
        df = df[~df.index.duplicated(keep="last")]

    df = df.sort_index()

    print(f"Rows: {len(df):,}")
    print(f"Date Range: {df.index.min()} → {df.index.max()}")

    return df

# ==================================================
# STRATEGY BUILDER
# ==================================================

def build_strategy():

    return TJRSMCStrategy(
        sweep_buffer_points=2.0,
        sweep_max_bars=6,

        bos_swing_left=2,
        bos_swing_right=2,
        bos_lookback_swings=8,

        fvg_max_age_bars=60,
        entry_requires_bias=True,

        killzone_only=True,
        killzone_start_et="09:30",
        killzone_end_et="11:00",

        cooldown_bars=20,
        max_signals_per_day=2
    )

# ==================================================
# ENGINE BUILDER
# ==================================================

def build_engine(data, strategy):

    tick_value = 0.50 if INSTRUMENT == "MNQ" else 5.00

    specs = ContractSpecs(
        tick_size=0.25,
        tick_value=tick_value,
        commission_per_side=0.35
    )

    # More realistic slippage
    execution = ExecutionConfig(
        fill_mode="next_open",
        slippage_ticks=1.5,
        allow_flip_same_bar=False
    )

    # Conservative risk defaults
    if PROP_MODE:
        risk = RiskConfig(
            equity_floor=0,
            max_contracts=1,
            max_daily_loss=1_500,
            max_drawdown=3_000
        )
    else:
        risk = RiskConfig(
            equity_floor=0,
            max_contracts=1,
            max_daily_loss=None,
            max_drawdown=None
        )

    sizing = SizingConfig(
        mode="fixed",
        fixed_qty=1
    )

    # More realistic trade management
    tm = TradeManagementConfig(
        stop_loss_ticks=30,
        take_profit_ticks=75,   # 2.5R
        trailing_stop_ticks=None,
        breakeven_after_ticks=25,
        time_stop_bars=60,

        exit_on_flat_signal=False,

        no_progress_bars=10,
        no_progress_ticks=8,

        add_on_after_ticks=None,
        add_on_qty=0,
        max_adds=0
    )

    return BacktestEngine(
        data=data,
        strategy=strategy,
        initial_capital=INITIAL_CAPITAL,
        specs=specs,
        execution=execution,
        risk=risk,
        sizing=sizing,
        tm=tm,
        results_dir=RESULTS_DIR
    )

# ==================================================
# METRICS
# ==================================================

def print_metrics(metrics):

    print("\nBACKTEST COMPLETE")
    print("=" * 60)

    print(f"Initial Capital: ${INITIAL_CAPITAL:,.2f}")
    print(f"Final Equity:    ${metrics['final_equity']:,.2f}")
    print(f"Net Profit:      ${metrics['net_profit']:,.2f}")

    print("\nTrade Stats")
    print("-" * 60)
    print(f"Trades:        {metrics['total_trades']}")
    print(f"Win Rate:      {metrics['win_rate']:.2%}")
    print(f"Profit Factor: {metrics['profit_factor']:.2f}")
    print(f"Expectancy:    ${metrics['expectancy']:.2f}")
    print(f"Avg Win:       ${metrics['avg_win']:.2f}")
    print(f"Avg Loss:      ${metrics['avg_loss']:.2f}")

    print("\nRisk")
    print("-" * 60)
    print(f"Max DD:        ${metrics['max_drawdown_dollars']:,.2f}")
    print(f"Max DD %:      {metrics['max_drawdown_pct']:.2%}")
    print(f"Sharpe:        {metrics['annualized_sharpe']:.2f}")
    print(f"Sortino:       {metrics['annualized_sortino']:.2f}")
    print(f"Exposure:      {metrics['exposure_pct']:.2%}")

    print("=" * 60)

# ==================================================
# MAIN
# ==================================================

def main():

    print("=" * 60)
    print("TJR SMC RESEARCH LAB")
    print("=" * 60)

    try:
        data = load_data()
        strategy = build_strategy()

        engine = build_engine(data, strategy)

        print("\nRunning Backtest...")
        results = engine.run()

        print_metrics(results["metrics"])

    except Exception as e:
        print("\nERROR OCCURRED")
        print("-" * 60)
        print(str(e))
        traceback.print_exc()
        print("=" * 60)


if __name__ == "__main__":
    main()
