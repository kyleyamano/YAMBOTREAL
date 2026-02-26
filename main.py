# main.py
"""
TJR SMC Research Lab — Main Backtest Runner

Usage:
    python main.py

Edit the CONFIG section below to change instrument, capital, strategy params, etc.
All errors are caught and explained with fix instructions.
"""

from dotenv import load_dotenv
import os
import sys
import traceback
import time
import pandas as pd

load_dotenv()
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)

# ==================================================
# CONFIG — Edit these to change what runs
# ==================================================

INSTRUMENT      = "MNQ"       # "MNQ" or "NQ"
INITIAL_CAPITAL = 50_000      # starting equity in dollars
PROP_MODE       = False       # True = apply TopStep-style daily loss + drawdown limits

RESULTS_DIR     = "results"

# Strategy params
STRATEGY_CONFIG = dict(
    sweep_buffer_points  = 2.0,
    sweep_max_bars       = 6,
    bos_swing_left       = 2,
    bos_swing_right      = 2,
    bos_lookback_swings  = 8,
    fvg_max_age_bars     = 60,
    entry_requires_bias  = True,
    killzone_only        = True,
    killzone_start_et    = "09:30",
    killzone_end_et      = "11:00",
    cooldown_bars        = 20,
    max_signals_per_day  = 2,
)

# Trade management
SL_TICKS        = 30          # stop loss in ticks
TP_TICKS        = 75          # take profit in ticks (2.5R)
BREAKEVEN_TICKS = 25          # move SL to breakeven after this many ticks in profit
TIME_STOP_BARS  = 60          # exit trade after this many bars regardless
SLIPPAGE_TICKS  = 1.5         # realistic slippage per fill

# ==================================================
# IMPORTS (after sys.path set)
# ==================================================

def _import_or_die():
    """Catch import errors with clear fix instructions."""
    errors = []

    try:
        from src.data.loader import load_parquet
    except ImportError as e:
        errors.append(f"  src/data/loader.py missing or broken: {e}\n"
                      f"  → Make sure you extracted the latest zip and ran git pull")

    try:
        from src.strategies.tjr_smc import TJRSMCStrategy
    except ImportError as e:
        errors.append(f"  TJRSMCStrategy import failed: {e}\n"
                      f"  → Check src/strategies/tjr_smc.py and src/utils/smc.py exist")

    try:
        from src.backtesting.engine import (
            BacktestEngine, ContractSpecs, ExecutionConfig,
            RiskConfig, SizingConfig, TradeManagementConfig
        )
    except ImportError as e:
        errors.append(f"  Engine import failed: {e}\n"
                      f"  → Check src/backtesting/engine.py — run scripts/validate_engine.py to diagnose")

    if errors:
        print("\n[IMPORT ERROR] Cannot start — fix these issues first:")
        for err in errors:
            print(err)
        sys.exit(1)

    from src.data.loader import load_parquet
    from src.strategies.tjr_smc import TJRSMCStrategy
    from src.backtesting.engine import (
        BacktestEngine, ContractSpecs, ExecutionConfig,
        RiskConfig, SizingConfig, TradeManagementConfig
    )
    return load_parquet, TJRSMCStrategy, BacktestEngine, ContractSpecs, ExecutionConfig, RiskConfig, SizingConfig, TradeManagementConfig


# ==================================================
# DATA
# ==================================================

def load_data(load_parquet):
    path_map = {
        "MNQ": "data/processed/mnq_continuous_1m.parquet",
        "NQ":  "data/processed/nq_continuous_1m.parquet",
    }
    if INSTRUMENT not in path_map:
        raise ValueError(f"Unknown INSTRUMENT '{INSTRUMENT}'. Must be 'MNQ' or 'NQ'.")

    path = path_map[INSTRUMENT]
    print(f"\n[DATA] Loading {INSTRUMENT} from: {path}")

    try:
        df = load_parquet(path)
    except FileNotFoundError as e:
        print(f"\n[DATA ERROR] {e}")
        print("\nFix options:")
        print("  1. Run: python scripts/build_continuous_mnq.py")
        print("  2. Or:  python scripts/convert_dbn_to_parquet.py")
        print("  3. Make sure your raw Databento files are in data/raw/")
        sys.exit(1)

    # Data quality report
    print(f"\n[DATA] Quality Report")
    print(f"  Rows:        {len(df):,}")
    print(f"  Date range:  {df.index.min().date()} → {df.index.max().date()}")
    print(f"  Days:        {(df.index.max() - df.index.min()).days:,}")
    print(f"  Price range: {df['close'].min():.2f} – {df['close'].max():.2f}")

    # Warn about gaps
    expected_bars = (df.index.max() - df.index.min()).total_seconds() / 60
    actual_bars   = len(df)
    gap_pct       = (1 - actual_bars / expected_bars) * 100
    if gap_pct > 30:
        print(f"  WARNING: ~{gap_pct:.0f}% of expected bars are missing — data may be incomplete")
    else:
        print(f"  Completeness: ~{100 - gap_pct:.0f}% (gaps expected for overnight/weekend)")

    return df


# ==================================================
# STRATEGY
# ==================================================

def build_strategy(TJRSMCStrategy):
    print(f"\n[STRATEGY] TJR SMC")
    print(f"  Killzone:   {STRATEGY_CONFIG['killzone_start_et']} – {STRATEGY_CONFIG['killzone_end_et']} ET")
    print(f"  Bias filter: {'ON' if STRATEGY_CONFIG['entry_requires_bias'] else 'OFF'}")
    print(f"  Max trades/day: {STRATEGY_CONFIG['max_signals_per_day']}")

    try:
        return TJRSMCStrategy(**STRATEGY_CONFIG)
    except TypeError as e:
        print(f"\n[STRATEGY ERROR] Invalid config parameter: {e}")
        print("  → Check STRATEGY_CONFIG at the top of main.py")
        sys.exit(1)


# ==================================================
# ENGINE
# ==================================================

def build_engine(data, strategy, classes):
    BacktestEngine, ContractSpecs, ExecutionConfig, RiskConfig, SizingConfig, TradeManagementConfig = classes

    tick_value = 0.50 if INSTRUMENT == "MNQ" else 5.00

    specs = ContractSpecs(
        tick_size=0.25,
        tick_value=tick_value,
        commission_per_side=0.35
    )

    execution = ExecutionConfig(
        fill_mode="next_open",
        slippage_ticks=SLIPPAGE_TICKS,
        allow_flip_same_bar=False
    )

    if PROP_MODE:
        risk = RiskConfig(
            equity_floor=0,
            max_contracts=1,
            max_daily_loss=1_500,
            max_drawdown=3_000
        )
        print("\n[RISK] PROP MODE ON — daily loss $1,500 / max drawdown $3,000")
    else:
        risk = RiskConfig(equity_floor=0, max_contracts=1)

    sizing = SizingConfig(mode="fixed", fixed_qty=1)

    tm = TradeManagementConfig(
        stop_loss_ticks      = SL_TICKS,
        take_profit_ticks    = TP_TICKS,
        trailing_stop_ticks  = None,
        breakeven_after_ticks= BREAKEVEN_TICKS,
        time_stop_bars       = TIME_STOP_BARS,
        exit_on_flat_signal  = False,
        no_progress_bars     = 10,
        no_progress_ticks    = 8,
    )

    print(f"\n[ENGINE] Config")
    print(f"  SL: {SL_TICKS} ticks  TP: {TP_TICKS} ticks  ({TP_TICKS/SL_TICKS:.1f}R)")
    print(f"  Breakeven after: {BREAKEVEN_TICKS} ticks")
    print(f"  Time stop: {TIME_STOP_BARS} bars")
    print(f"  Fill mode: next_open  Slippage: {SLIPPAGE_TICKS} ticks")

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
# RESULTS PRINTER
# ==================================================

def print_results(metrics, elapsed):
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)

    # P&L
    pnl_sign = "+" if metrics['net_profit'] >= 0 else ""
    print(f"\nCapital")
    print(f"  Start:       ${INITIAL_CAPITAL:>12,.2f}")
    print(f"  End:         ${metrics['final_equity']:>12,.2f}")
    print(f"  Net P&L:     {pnl_sign}${metrics['net_profit']:>11,.2f}  "
          f"({pnl_sign}{metrics['net_profit']/INITIAL_CAPITAL:.1%})")

    # Trades
    print(f"\nTrades")
    print(f"  Total:       {metrics['total_trades']:>5}")

    if metrics['total_trades'] == 0:
        print("\n  NO TRADES GENERATED — check strategy config and data date range")
        _print_notrade_tips()
    else:
        print(f"  Win Rate:    {metrics['win_rate']:>6.1%}")
        print(f"  Profit Factor: {metrics['profit_factor']:>5.2f}  (>1.5 = decent, >2.0 = strong)")
        print(f"  Expectancy:  ${metrics['expectancy']:>8.2f} per trade")
        print(f"  Avg Win:     ${metrics['avg_win']:>8.2f}")
        print(f"  Avg Loss:    ${metrics['avg_loss']:>8.2f}")

        # Exit breakdown
        if metrics.get('exit_reasons'):
            print(f"\nExit Reasons")
            for reason, count in sorted(metrics['exit_reasons'].items(), key=lambda x: -x[1]):
                pct = count / metrics['total_trades']
                print(f"  {reason:<22} {count:>4}  ({pct:.0%})")

        # Risk
        print(f"\nRisk")
        print(f"  Max Drawdown:  ${metrics['max_drawdown_dollars']:>9,.2f}  "
              f"({metrics['max_drawdown_pct']:.1%})")
        print(f"  Sharpe:        {metrics['annualized_sharpe']:>6.2f}  (>1.0 = acceptable, >2.0 = strong)")
        print(f"  Sortino:       {metrics['annualized_sortino']:>6.2f}")
        print(f"  Calmar:        {metrics['calmar']:>6.2f}")
        print(f"  Exposure:      {metrics['exposure_pct']:>6.1%}")

        # Regime breakdown
        if metrics.get('regime_breakdown'):
            print(f"\nRegime Breakdown")
            for r in metrics['regime_breakdown']:
                print(f"  {r['regime']:<25}  {r['trades']:>3} trades  "
                      f"WR {r['win_rate']:.0%}  P&L ${r['total_pnl']:>8,.2f}")

        # Quick verdict
        print(f"\nVerdict")
        pf = metrics['profit_factor']
        wr = metrics['win_rate']
        sh = metrics['annualized_sharpe']
        if pf >= 1.5 and wr >= 0.4 and sh >= 1.0:
            print("  ✓ Edge looks real — worth walk-forward testing")
        elif pf >= 1.2:
            print("  ~ Marginal edge — needs more data or param tuning")
        else:
            print("  ✗ No edge detected — review strategy logic or params")

    print(f"\nCompleted in {elapsed:.1f}s")
    print(f"Results saved to: {RESULTS_DIR}/")
    print("=" * 60)


def _print_notrade_tips():
    print("\n  Common causes:")
    print("  1. Data date range doesn't overlap killzone hours")
    print("  2. entry_requires_bias=True but no days pass the bias filter")
    print("  3. sweep_buffer_points too large — try lowering to 0.5")
    print("  4. Data is not 1-minute bars (check rows vs date range)")
    print("\n  Try: set entry_requires_bias=False in STRATEGY_CONFIG and re-run")


# ==================================================
# MAIN
# ==================================================

def main():
    print("=" * 60)
    print("TJR SMC RESEARCH LAB")
    print(f"  Instrument:  {INSTRUMENT}")
    print(f"  Capital:     ${INITIAL_CAPITAL:,}")
    print(f"  Prop Mode:   {'ON' if PROP_MODE else 'OFF'}")
    print("=" * 60)

    # Step 1: imports
    load_parquet, TJRSMCStrategy, BacktestEngine, ContractSpecs, \
        ExecutionConfig, RiskConfig, SizingConfig, TradeManagementConfig = _import_or_die()

    engine_classes = (BacktestEngine, ContractSpecs, ExecutionConfig,
                      RiskConfig, SizingConfig, TradeManagementConfig)

    try:
        # Step 2: data
        data = load_data(load_parquet)

        # Step 3: strategy
        strategy = build_strategy(TJRSMCStrategy)

        # Step 4: engine
        engine = build_engine(data, strategy, engine_classes)

        # Step 5: run
        print(f"\n[RUN] Starting backtest on {len(data):,} bars...")
        t0 = time.time()
        results = engine.run()
        elapsed = time.time() - t0

        # Step 6: results
        print_results(results["metrics"], elapsed)

    except KeyboardInterrupt:
        print("\n\nBacktest interrupted by user.")

    except Exception as e:
        print(f"\n{'=' * 60}")
        print(f"[ERROR] {type(e).__name__}: {e}")
        print(f"{'=' * 60}")
        print("\nFull traceback:")
        traceback.print_exc()
        print(f"\n{'=' * 60}")
        print("Troubleshooting:")
        print("  1. Run: python scripts/validate_engine.py")
        print("  2. Check your data exists at the expected path")
        print("  3. Check STRATEGY_CONFIG params at top of main.py")
        print(f"{'=' * 60}")
        sys.exit(1)


if __name__ == "__main__":
    main()
