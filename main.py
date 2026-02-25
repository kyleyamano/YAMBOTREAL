from dotenv import load_dotenv
import os
import sys
import traceback
import pandas as pd

# --------------------------------------------------
# Load Environment Variables
# --------------------------------------------------
load_dotenv()

# Ensure project root is in path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)

# --------------------------------------------------
# Imports
# --------------------------------------------------
from src.data.loader import load_parquet
from src.research.regimes import add_regime_labels
from src.strategies.rsi_strategy import RSIStrategy
from src.backtesting.engine import BacktestEngine, ExecutionConfig, RiskConfig


def main():

    print("=" * 60)
    print("MNQ LAB - PROFESSIONAL BACKTEST")
    print("=" * 60)

    try:

        # --------------------------------------------------
        # Load Data
        # --------------------------------------------------
        print("\nLoading MNQ 1m parquet data...")
        data = load_parquet("data/raw/mnq_1m_full.parquet")

        # Ensure datetime index
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index, utc=True)

        # Remove duplicate timestamps
        if data.index.duplicated().any():
            print("WARNING: Duplicate timestamps detected. Removing duplicates...")
            data = data[~data.index.duplicated(keep="last")]

        print(f"Rows Loaded: {len(data):,}")
        print(f"Date Range: {data.index.min()} → {data.index.max()}")

        # --------------------------------------------------
        # Add Regime Labels
        # --------------------------------------------------
        print("\nClassifying market regimes...")
        data = add_regime_labels(data)

        print("\nRegime Distribution:")
        print(data["regime"].value_counts())

        # --------------------------------------------------
        # Strategy
        # --------------------------------------------------
        strategy = RSIStrategy(window=14)

        # --------------------------------------------------
        # Engine Configuration
        # --------------------------------------------------
        execution = ExecutionConfig(
            tick_size=0.25,
            slippage_ticks=1.0,
            commission_per_contract=0.50,
            execution_price="close",
            signal_shift=1
        )

        risk = RiskConfig(
            max_leverage_contracts=10,
            stop_if_equity_below=4_500
        )

        engine = BacktestEngine(
            data=data,
            strategy=strategy,
            initial_capital=150_000,
            contract_multiplier=2,
            contracts=1,
            execution=execution,
            risk=risk
        )

        # --------------------------------------------------
        # Run Backtest
        # --------------------------------------------------
        print("\nRunning backtest...")
        results = engine.run()

        # --------------------------------------------------
        # Results
        # --------------------------------------------------
        print("\nBACKTEST COMPLETE")
        print("-" * 60)

        print(f"Initial Capital: ${50_000:,.2f}")
        print(f"Final Equity:    ${results['final_equity']:,.2f}")
        print(f"Net Profit:      ${results['net_profit']:,.2f}")

        print("\nTrade Statistics")
        print("-" * 60)
        print(f"Total Trades: {results['total_trades']}")
        print(f"Win Rate:     {results['win_rate']:.2%}")
        print(f"Profit Factor:{results['profit_factor']:.2f}")
        print(f"Expectancy:   ${results['expectancy']:.2f}")
        print(f"Avg Win:      ${results['avg_win']:.2f}")
        print(f"Avg Loss:     ${results['avg_loss']:.2f}")

        print("\nRisk Metrics")
        print("-" * 60)
        print(f"Max Drawdown: {results['max_drawdown']:.2%}")
        print(f"Sharpe (approx): {results['sharpe_approx']:.2f}")

        print("=" * 60)

    except Exception as e:
        print("\nERROR OCCURRED")
        print("-" * 60)
        print(str(e))
        traceback.print_exc()
        print("=" * 60)


if __name__ == "__main__":
    main()
