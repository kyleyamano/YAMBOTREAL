# src/backtesting/engine.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd


# -----------------------------
# Helpers / Config
# -----------------------------

@dataclass
class ExecutionConfig:
    """
    Execution assumptions for a simple, robust bar-based backtester.
    - execution_price: "close" (default) or "open"
      If "close": fills are assumed at the current bar close (after shifting signals).
      If "open": fills are assumed at the current bar open (after shifting signals).
    - signal_shift: 1 means execute signals one bar later to avoid lookahead.
    """
    tick_size: float = 0.25
    slippage_ticks: float = 1.0
    commission_per_contract: float = 0.50  # per side, per contract
    execution_price: str = "close"         # "close" or "open"
    signal_shift: int = 1                  # avoid lookahead


@dataclass
class RiskConfig:
    """
    Optional safety rails.
    max_leverage_contracts is mostly a sanity limit (not margin-based).
    stop_if_equity_below stops sim if equity <= that value.
    """
    max_leverage_contracts: int = 100
    stop_if_equity_below: float = 0.0


class BacktestEngine:
    """
    A robust backtesting engine for single-instrument futures-style trading.
    Supports:
      - long/short/flat signals (-1, 0, +1)
      - slippage + commissions
      - mark-to-market equity curve each bar
      - trade blotter (entries/exits, pnl)
      - summary metrics
      - CSV export to results/

    Expected data columns (case-insensitive supported):
      - datetime index OR a datetime column
      - open/high/low/close (or Open/High/Low/Close)
      - optional volume

    Strategy contract:
      strategy.generate_signals(data) may return:
        - pd.Series of signals indexed like data
        - pd.DataFrame with a 'signal' column (or 'Signal')
        - pd.DataFrame containing data + 'signal'
    """

    def __init__(
        self,
        data: pd.DataFrame,
        strategy: Any,
        initial_capital: float = 50_000.0,
        contract_multiplier: float = 2.0,  # MNQ = $2 / point
        contracts: int = 1,
        execution: Optional[ExecutionConfig] = None,
        risk: Optional[RiskConfig] = None,
        results_dir: Union[str, Path] = "results",
        signal_col: str = "signal",
        allow_position_flip_same_bar: bool = True,
    ):
        self.data = data.copy()
        self.strategy = strategy
        self.initial_capital = float(initial_capital)
        self.contract_multiplier = float(contract_multiplier)
        self.contracts = int(contracts)
        self.execution = execution or ExecutionConfig()
        self.risk = risk or RiskConfig()
        self.results_dir = Path(results_dir)
        self.signal_col = signal_col
        self.allow_position_flip_same_bar = allow_position_flip_same_bar

        if self.contracts <= 0:
            raise ValueError("contracts must be >= 1")

        if self.contracts > self.risk.max_leverage_contracts:
            raise ValueError(
                f"contracts={self.contracts} exceeds max_leverage_contracts={self.risk.max_leverage_contracts}"
            )

        if self.execution.execution_price not in ("close", "open"):
            raise ValueError("execution_price must be 'close' or 'open'")

        if self.execution.signal_shift < 0:
            raise ValueError("signal_shift must be >= 0")

    # -----------------------------
    # Public API
    # -----------------------------

    def run(self) -> Dict[str, Any]:
        df = self._prepare_market_data(self.data)

        # Generate signals robustly
        sig = self._get_signal_series(df)

        # Shift signals to avoid lookahead
        if self.execution.signal_shift > 0:
            sig = sig.shift(self.execution.signal_shift)

        sig = sig.fillna(0).astype(int)

        # Align
        sig = sig.reindex(df.index).fillna(0).astype(int)

        equity_curve, trades = self._simulate(df, sig)

        # Metrics + outputs
        results = self._compute_results(equity_curve, trades)
        out = self._persist_outputs(df, sig, equity_curve, trades)

        results.update(out)
        results["execution_assumptions"] = {
            "tick_size": self.execution.tick_size,
            "slippage_ticks": self.execution.slippage_ticks,
            "commission_per_contract": self.execution.commission_per_contract,
            "contracts": self.contracts,
            "contract_multiplier": self.contract_multiplier,
            "execution_price": self.execution.execution_price,
            "signal_shift": self.execution.signal_shift,
            "notes": "signals executed after shift; bar-based fills with slippage+commission per side",
        }

        return results

    # -----------------------------
    # Data prep / validation
    # -----------------------------

    def _prepare_market_data(self, df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(df, pd.DataFrame) or df.empty:
            raise ValueError("data must be a non-empty pandas DataFrame")

        df = df.copy()

        # Normalize columns (case-insensitive)
        colmap = {c: c.lower() for c in df.columns}
        df.rename(columns=colmap, inplace=True)

        # Ensure datetime index
        if df.index.name is None or "datetime" in df.columns:
            if "datetime" in df.columns:
                df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
                df = df.dropna(subset=["datetime"]).set_index("datetime")
            else:
                # Try to interpret the index as datetime
                try:
                    df.index = pd.to_datetime(df.index, utc=True, errors="raise")
                except Exception as e:
                    raise ValueError("Data must have a datetime index or a 'datetime' column") from e

        # Sort + de-dup timestamps (keep last)
        df = df.sort_index()
        if df.index.has_duplicates:
            df = df[~df.index.duplicated(keep="last")]

        # Required OHLC columns
        required = ["open", "high", "low", "close"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}. Found columns: {list(df.columns)}")

        # Convert to numeric safely
        for c in required:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        df = df.dropna(subset=required)
        if df.empty:
            raise ValueError("After cleaning, data has no rows (check OHLC columns for NaNs).")

        return df

    # -----------------------------
    # Signal handling
    # -----------------------------

    def _get_signal_series(self, df: pd.DataFrame) -> pd.Series:
        out = self.strategy.generate_signals(df)

        # If strategy returns Series: interpret as signals directly
        if isinstance(out, pd.Series):
            sig = out.copy()

        # If DataFrame: try to find signal column
        elif isinstance(out, pd.DataFrame):
            tmp = out.copy()
            tmp_cols = {c: c.lower() for c in tmp.columns}
            tmp.rename(columns=tmp_cols, inplace=True)

            if self.signal_col.lower() in tmp.columns:
                sig = tmp[self.signal_col.lower()].copy()
            elif "signal" in tmp.columns:
                sig = tmp["signal"].copy()
            else:
                raise ValueError(
                    f"Strategy returned a DataFrame but no '{self.signal_col}'/'signal' column was found. "
                    f"Columns: {list(out.columns)}"
                )
        else:
            raise TypeError("strategy.generate_signals must return a pandas Series or DataFrame")

        # Align / clean signals to int in {-1,0,1}
        sig = sig.reindex(df.index)
        sig = pd.to_numeric(sig, errors="coerce").fillna(0)

        # Clip to [-1,1] then round to int
        sig = sig.clip(-1, 1).round().astype(int)

        return sig

    # -----------------------------
    # Simulation
    # -----------------------------

    def _simulate(self, df: pd.DataFrame, sig: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame]:
        tick = float(self.execution.tick_size)
        slip_ticks = float(self.execution.slippage_ticks)
        slip_points = slip_ticks * tick

        commission = float(self.execution.commission_per_contract) * self.contracts
        mult = self.contract_multiplier * self.contracts

        # Choose fill price series
        px = df["close"] if self.execution.execution_price == "close" else df["open"]

        cash = self.initial_capital
        position = 0  # -1 short, 0 flat, +1 long
        entry_price = np.nan
        entry_time = None

        trades = []
        equity_rows = []

        for ts, price in px.items():
            desired = int(sig.loc[ts])  # already aligned and int

            # Mark-to-market equity using current close (not fill px) for realism
            mkt_price = float(df.loc[ts, "close"])
            unreal = 0.0
            if position != 0 and np.isfinite(entry_price):
                unreal = (mkt_price - entry_price) * mult * position

            equity = cash + unreal
            equity_rows.append((ts, cash, unreal, equity, position))

            if equity <= self.risk.stop_if_equity_below:
                # Stop sim if equity blown
                break

            # Decide if we need to trade
            if desired == position:
                continue

            # If we have a position, we may need to exit first
            if position != 0:
                exit_fill = self._apply_slippage(price=float(price), side=-position, slip_points=slip_points)
                pnl_points = (exit_fill - entry_price) * position
                gross_pnl = pnl_points * mult

                # commissions per side
                net_pnl = gross_pnl - commission  # exit commission
                cash += net_pnl

                trades.append(
                    {
                        "entry_time": entry_time,
                        "exit_time": ts,
                        "side": "LONG" if position == 1 else "SHORT",
                        "contracts": self.contracts,
                        "entry_price": float(entry_price),
                        "exit_price": float(exit_fill),
                        "pnl_points": float(pnl_points),
                        "gross_pnl": float(gross_pnl),
                        "net_pnl": float(net_pnl),
                        "commission_exit": float(commission),
                        "slippage_points_assumed": float(slip_points),
                    }
                )

                # flat after exit
                position = 0
                entry_price = np.nan
                entry_time = None

                # If desired is 0, we are done
                if desired == 0:
                    continue

                # If flip is not allowed same bar, skip entering now
                if not self.allow_position_flip_same_bar:
                    continue

            # Enter new position if desired != 0
            if desired != 0:
                entry_fill = self._apply_slippage(price=float(price), side=desired, slip_points=slip_points)

                # Pay entry commission immediately
                cash -= commission

                position = desired
                entry_price = entry_fill
                entry_time = ts

                trades[-0:]  # no-op, keeps structure explicit

        equity_df = pd.DataFrame(
            equity_rows,
            columns=["datetime", "cash", "unrealized_pnl", "equity", "position"],
        ).set_index("datetime")

        trades_df = pd.DataFrame(trades)

        return equity_df, trades_df

    @staticmethod
    def _apply_slippage(price: float, side: int, slip_points: float) -> float:
        """
        side: +1 for buy/long entry or short exit
              -1 for sell/short entry or long exit
        For buys, worse price = higher; for sells, worse price = lower.
        """
        if side not in (-1, 1):
            raise ValueError("side must be -1 or +1")
        return float(price + slip_points) if side == 1 else float(price - slip_points)

    # -----------------------------
    # Metrics
    # -----------------------------

    def _compute_results(self, equity: pd.DataFrame, trades: pd.DataFrame) -> Dict[str, Any]:
        final_equity = float(equity["equity"].iloc[-1])
        net_profit = final_equity - self.initial_capital

        # --------------------------------------------------
        # Regime Breakdown (if regime column exists)
        # --------------------------------------------------
        regime_breakdown = []

        if "regime" in self.data.columns:

            merged = equity.copy()
            merged["regime"] = self.data["regime"].reindex(merged.index)

            for regime_name in merged["regime"].dropna().unique():

                subset = merged[merged["regime"] == regime_name]

                if len(subset) > 10:
                    start_eq = subset["equity"].iloc[0]
                    end_eq = subset["equity"].iloc[-1]

                    regime_return = (end_eq / start_eq) - 1.0

                    regime_breakdown.append({
                        "regime": regime_name,
                        "return": float(regime_return)
                    })

        total_trades = int(len(trades))
        if total_trades == 0:
            return {
                        return {
            "final_equity": final_equity,
            "net_profit": float(net_profit),
            "total_trades": total_trades,
            "win_rate": float(win_rate),
            "profit_factor": float(profit_factor),
            "expectancy": float(expectancy),
            "avg_win": float(avg_win),
            "avg_loss": float(avg_loss),
            "max_drawdown": float(max_dd),
            "sharpe_approx": float(sharpe),
            "regime_breakdown": regime_breakdown
        }
            }

        pnl = trades["net_pnl"].astype(float)
        wins = pnl[pnl > 0]
        losses = pnl[pnl < 0]

        win_rate = float((pnl > 0).mean())

        gross_profit = float(wins.sum()) if len(wins) else 0.0
        gross_loss = float(losses.sum()) if len(losses) else 0.0  # negative

        profit_factor = float(gross_profit / abs(gross_loss)) if gross_loss != 0 else float("inf")

        avg_win = float(wins.mean()) if len(wins) else 0.0
        avg_loss = float(losses.mean()) if len(losses) else 0.0  # negative

        expectancy = float(pnl.mean())

        # Max drawdown from equity curve
        eq = equity["equity"].astype(float)
        peak = eq.cummax()
        dd = (eq / peak) - 1.0
        max_dd = float(dd.min())  # negative number

        # Approx Sharpe using bar-to-bar equity returns
        rets = eq.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
        if len(rets) > 2 and rets.std(ddof=1) > 0:
            sharpe = float((rets.mean() / rets.std(ddof=1)) * np.sqrt(252))  # rough daily scaling
        else:
            sharpe = 0.0

        return {
            "final_equity": final_equity,
            "net_profit": float(net_profit),
            "total_trades": total_trades,
            "win_rate": float(win_rate),
            "profit_factor": float(profit_factor),
            "expectancy": float(expectancy),
            "avg_win": float(avg_win),
            "avg_loss": float(avg_loss),
            "max_drawdown": float(max_dd),
            "sharpe_approx": float(sharpe),
        }

    # -----------------------------
    # Output
    # -----------------------------

    def _persist_outputs(
        self,
        df: pd.DataFrame,
        sig: pd.Series,
        equity: pd.DataFrame,
        trades: pd.DataFrame,
    ) -> Dict[str, Any]:
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Save equity curve
        equity_path = self.results_dir / "equity_curve.csv"
        equity.to_csv(equity_path)

        # Save trades
        trades_path = self.results_dir / "trades.csv"
        trades.to_csv(trades_path, index=False)

        # Optional: save signals alongside close for debugging
        signals_path = self.results_dir / "signals.csv"
        out = pd.DataFrame(
            {
                "close": df["close"],
                "signal": sig.astype(int),
            },
            index=df.index,
        )
        out.to_csv(signals_path)

        return {
            "equity_curve_path": str(equity_path),
            "trades_path": str(trades_path),
            "signals_path": str(signals_path),
        }
