# src/backtesting/engine.py
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union, List

import numpy as np
import pandas as pd


# -----------------------------
# Config / Specs
# -----------------------------

@dataclass(frozen=True)
class ContractSpecs:
    """
    Keep specs separate from data source.
    You can run NQ-price history while applying MNQ specs via tick_value.
    """
    tick_size: float = 0.25
    tick_value: float = 0.50              # MNQ = $0.50 per tick (0.25 point)
    commission_per_side: float = 0.35     # per contract per side ($)


@dataclass(frozen=True)
class ExecutionConfig:
    """
    Event-driven, bar-by-bar.
    fill_mode:
      - "next_open" (recommended): decision on bar t executes at bar t+1 open
    slippage_ticks: adverse ticks per fill
    allow_flip_same_bar: if signal flips, allow exit+entry on same next_open
    """
    fill_mode: str = "next_open"
    slippage_ticks: float = 1.0
    allow_flip_same_bar: bool = True


@dataclass(frozen=True)
class RiskConfig:
    """
    Guardrails.
    equity_floor: stop entering new trades if equity <= equity_floor (and flatten then stop)
    max_contracts: hard cap on position size
    max_daily_loss: realized PnL threshold; stop entering new trades for that day
    max_drawdown: peak-to-trough equity drawdown threshold; stop trading entirely
    """
    equity_floor: float = 0.0
    max_contracts: int = 10
    max_daily_loss: Optional[float] = None
    max_drawdown: Optional[float] = None


@dataclass(frozen=True)
class SizingConfig:
    """
    Position sizing modes:
      - mode="fixed": use fixed_qty
      - mode="percent_risk": risk_pct_of_equity with stop_loss_ticks required
    """
    mode: str = "fixed"              # "fixed" | "percent_risk"
    fixed_qty: int = 1
    risk_pct_of_equity: Optional[float] = None  # e.g. 0.01
    stop_loss_ticks_for_sizing: Optional[float] = None


@dataclass(frozen=True)
class TradeManagementConfig:
    """
    Optional trade management. All are in ticks.
    If both stop and target hit in the same bar, we assume STOP first (conservative).

    exit_on_flat_signal:
      - True  (default): if signal goes to 0, exit position (old behavior)
      - False: ignore signal==0 while in position; exits are via TM rules and/or opposite signal flips

    no_progress_bars + no_progress_ticks:
      Objective early-exit: if after N bars, MFE < X ticks, exit at current open.

    add_on_after_ticks / add_on_qty / max_adds:
      Optional add-on/pyramid: once MFE >= threshold, add qty at current open (risk-capped). Off by default.
    """
    # Existing
    stop_loss_ticks: Optional[float] = None
    take_profit_ticks: Optional[float] = None
    trailing_stop_ticks: Optional[float] = None
    breakeven_after_ticks: Optional[float] = None
    time_stop_bars: Optional[int] = None

    # New: signal exit behavior
    exit_on_flat_signal: bool = True

    # New: early exit if no progress
    no_progress_bars: Optional[int] = None
    no_progress_ticks: Optional[float] = None

    # New: optional add-on/pyramid
    add_on_after_ticks: Optional[float] = None
    add_on_qty: int = 0
    max_adds: int = 0


# -----------------------------
# Engine
# -----------------------------

class BacktestEngine:
    """
    Event-driven futures backtester:
      - strict {-1,0,1} signals (shifted to prevent lookahead)
      - next-bar open fills
      - slippage + commission per side
      - optional stops/targets/trailing/breakeven/time-stop/no-progress/add-on
      - risk gates: equity floor, max daily loss, max drawdown
      - metrics + trade ledger
      - optional artifact export to results/
    """

    def __init__(
        self,
        data: pd.DataFrame,
        strategy: Any,
        initial_capital: float = 50_000.0,
        specs: Optional[ContractSpecs] = None,
        execution: Optional[ExecutionConfig] = None,
        risk: Optional[RiskConfig] = None,
        sizing: Optional[SizingConfig] = None,
        tm: Optional[TradeManagementConfig] = None,
        results_dir: Union[str, Path] = "results",
        signal_col: str = "signal",
        export_artifacts: bool = True,
    ):
        self.data = data.copy()
        self.strategy = strategy
        self.initial_capital = float(initial_capital)
        self.specs = specs or ContractSpecs()
        self.execution = execution or ExecutionConfig()
        self.risk = risk or RiskConfig()
        self.sizing = sizing or SizingConfig()
        self.tm = tm or TradeManagementConfig()
        self.results_dir = Path(results_dir)
        self.signal_col = signal_col
        self.export_artifacts = export_artifacts

        if self.execution.fill_mode != "next_open":
            raise ValueError("Only fill_mode='next_open' is supported in this engine build.")
        if self.risk.max_contracts <= 0:
            raise ValueError("risk.max_contracts must be >= 1")

    # -----------------------------
    # Public API
    # -----------------------------

    def run(self) -> Dict[str, Any]:
        df = self._prepare_market_data(self.data)
        sig = self._get_signal_series(df)

        # Prevent lookahead: signal at t acts at t+1 open
        sig = sig.shift(1).fillna(0).astype(int)
        sig = sig.reindex(df.index).fillna(0).astype(int)

        equity, trades = self._simulate(df, sig)
        metrics = self._compute_metrics(equity, trades)

        out = {
            "metrics": metrics,
            "equity": equity,
            "trades": trades,
            "configs": {
                "specs": asdict(self.specs),
                "execution": asdict(self.execution),
                "risk": asdict(self.risk),
                "sizing": asdict(self.sizing),
                "trade_management": asdict(self.tm),
            }
        }

        if self.export_artifacts:
            out.update(self._persist_outputs(df, sig, equity, trades))

        return out

    # -----------------------------
    # Data prep / signals
    # -----------------------------

    def _prepare_market_data(self, df: pd.DataFrame) -> pd.DataFrame:
        required = {"open", "high", "low", "close"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        out = df.copy()
        # normalize column case
        out = out.rename(columns={c: c.lower() for c in out.columns})
        if not isinstance(out.index, pd.DatetimeIndex):
            out.index = pd.to_datetime(out.index, utc=True)
        out = out.sort_index()
        return out

    def _get_signal_series(self, df: pd.DataFrame) -> pd.Series:
        # Strategy expected to create a 'signal' series or return one
        sig = None
        if hasattr(self.strategy, "generate_signals"):
            sig = self.strategy.generate_signals(df)
        elif hasattr(self.strategy, "signal"):
            sig = getattr(self.strategy, "signal")
        if sig is None:
            raise ValueError("Strategy must implement generate_signals(df) returning a Series of {-1,0,1}.")
        sig = pd.Series(sig, index=df.index).fillna(0).astype(int)
        sig = sig.clip(-1, 1)
        return sig

    # -----------------------------
    # Sizing
    # -----------------------------

    def _determine_qty(self, equity: float) -> int:
        mode = self.sizing.mode

        if mode == "fixed":
            qty = int(self.sizing.fixed_qty)

        elif mode == "percent_risk":
            rp = self.sizing.risk_pct_of_equity
            st = self.sizing.stop_loss_ticks_for_sizing
            if rp is None or st is None:
                raise ValueError("percent_risk requires risk_pct_of_equity and stop_loss_ticks_for_sizing")

            risk_dollars = max(0.0, equity) * float(rp)
            dollars_per_contract = float(st) * self.specs.tick_value
            qty = int(risk_dollars // dollars_per_contract) if dollars_per_contract > 0 else 0
            qty = max(1, qty) if risk_dollars > 0 else 0

        else:
            raise ValueError("Unknown sizing.mode")

        qty = max(0, min(qty, self.risk.max_contracts))
        return qty

    # -----------------------------
    # Execution helpers
    # -----------------------------

    def _slip(self, price: float, side: int, is_entry: bool) -> float:
        slip = float(self.execution.slippage_ticks) * self.specs.tick_size
        if slip <= 0:
            return float(price)

        # Adverse slippage
        if is_entry:
            return float(price + slip) if side > 0 else float(price - slip)
        else:
            # exiting a long sells worse (down), exiting a short buys worse (up)
            return float(price - slip) if side > 0 else float(price + slip)

    # -----------------------------
    # Simulation
    # -----------------------------

    def _simulate(self, df: pd.DataFrame, sig: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame]:
        idx = df.index
        o = df["open"].to_numpy(float)
        h = df["high"].to_numpy(float)
        l = df["low"].to_numpy(float)
        c = df["close"].to_numpy(float)
        s = sig.to_numpy(int)

        tick = self.specs.tick_size
        tick_value = self.specs.tick_value
        comm_side = self.specs.commission_per_side

        cash = self.initial_capital
        pos = 0
        qty = 0
        entry_px = np.nan
        entry_i = -1
        entry_time = None
        entry_commission = 0.0

        # For trailing/breakeven/MFE
        best_favorable = 0.0  # in ticks (MFE)
        stop_px = None
        target_px = None

        # Add-on tracking
        adds_done = 0

        peak_equity = self.initial_capital
        daily_realized = 0.0
        current_day = idx[0].date()
        stop_all_trading = False

        trades: List[Dict[str, Any]] = []
        equity_rows: List[Tuple[pd.Timestamp, float, float, float, int]] = []

        def pnl_ticks(side: int, entry: float, exit: float) -> float:
            return ((exit - entry) / tick) * side

        def pnl_dollars(side: int, q: int, entry: float, exit: float) -> float:
            return pnl_ticks(side, entry, exit) * tick_value * q

        def update_trade_levels_for_entry(side: int, ep: float):
            nonlocal stop_px, target_px, best_favorable
            best_favorable = 0.0
            stop_px = None
            target_px = None

            if self.tm.stop_loss_ticks is not None:
                st = float(self.tm.stop_loss_ticks) * tick
                stop_px = ep - st if side > 0 else ep + st

            if self.tm.take_profit_ticks is not None:
                tp = float(self.tm.take_profit_ticks) * tick
                target_px = ep + tp if side > 0 else ep - tp

        def maybe_update_trailing_and_breakeven(side: int, ep: float, bar_high: float, bar_low: float):
            nonlocal stop_px, best_favorable

            # Update MFE in ticks
            if side > 0:
                mfe = (bar_high - ep) / tick
            else:
                mfe = (ep - bar_low) / tick
            if mfe > best_favorable:
                best_favorable = float(mfe)

            # Breakeven move
            if self.tm.breakeven_after_ticks is not None and stop_px is not None:
                if best_favorable >= float(self.tm.breakeven_after_ticks):
                    # Move stop to entry (or better)
                    if side > 0:
                        stop_px = max(float(stop_px), float(ep))
                    else:
                        stop_px = min(float(stop_px), float(ep))

            # Trailing stop (simple)
            if self.tm.trailing_stop_ticks is not None and stop_px is not None:
                tr = float(self.tm.trailing_stop_ticks) * tick
                if side > 0:
                    new_stop = bar_high - tr
                    stop_px = max(float(stop_px), float(new_stop))
                else:
                    new_stop = bar_low + tr
                    stop_px = min(float(stop_px), float(new_stop))

        def intrabar_exit_check(side: int, bar_high: float, bar_low: float):
            """
            Approx: check whether stop/target levels were touched during bar.
            Conservative: if both hit, assume STOP first.
            """
            if stop_px is None and target_px is None:
                return None

            stop_hit = False
            target_hit = False

            if stop_px is not None:
                if side > 0 and bar_low <= float(stop_px):
                    stop_hit = True
                if side < 0 and bar_high >= float(stop_px):
                    stop_hit = True

            if target_px is not None:
                if side > 0 and bar_high >= float(target_px):
                    target_hit = True
                if side < 0 and bar_low <= float(target_px):
                    target_hit = True

            if stop_hit:
                return ("stop", float(stop_px))
            if target_hit:
                return ("target", float(target_px))
            return None

        def close_position(i: int, reason: str, exit_level: float, exit_time: pd.Timestamp):
            nonlocal cash, daily_realized, pos, qty, entry_px, entry_i, entry_time, stop_px, target_px, best_favorable, entry_commission, adds_done

            if pos == 0:
                return

            exit_fill = self._slip(exit_level, side=pos, is_entry=False)
            gross = pnl_dollars(pos, qty, float(entry_px), exit_fill)

            exit_commission = qty * comm_side
            net = gross - exit_commission  # entry commission already deducted from cash
            cash += net
            daily_realized += net

            trades.append({
                "entry_time": entry_time,
                "exit_time": exit_time,
                "reason": reason,
                "side": "LONG" if pos > 0 else "SHORT",
                "qty": qty,
                "entry_price": float(entry_px),
                "exit_price": float(exit_fill),
                "gross_pnl": float(gross),
                "commission_entry": float(entry_commission),
                "commission_exit": float(exit_commission),
                "net_pnl": float(gross - entry_commission - exit_commission),
                "mfe_ticks": float(best_favorable),
                "bars_held": int(i - entry_i) if entry_i >= 0 else None,
                "adds_done": int(adds_done),
            })

            pos = 0
            qty = 0
            entry_px = np.nan
            entry_i = -1
            entry_time = None
            entry_commission = 0.0
            stop_px = None
            target_px = None
            best_favorable = 0.0
            adds_done = 0

        def open_position(i: int, side: int, qty_new: int, fill_price: float, t: pd.Timestamp):
            nonlocal cash, pos, qty, entry_px, entry_i, entry_time, entry_commission, adds_done

            entry_fill = self._slip(fill_price, side=side, is_entry=True)
            entry_commission = qty_new * comm_side
            cash -= entry_commission

            pos = side
            qty = qty_new
            entry_px = entry_fill
            entry_i = i
            entry_time = t
            adds_done = 0

            update_trade_levels_for_entry(pos, float(entry_px))

        def add_on_position(i: int, add_qty: int, fill_price: float, t: pd.Timestamp):
            """
            Add-on at current open. We:
              - fill with slippage
              - charge commission for added qty
              - update average entry price (VWAP)
              - recompute stop/target off new avg entry
            """
            nonlocal cash, qty, entry_px, entry_commission, adds_done, pos

            if pos == 0 or add_qty <= 0:
                return

            fill = self._slip(fill_price, side=pos, is_entry=True)
            add_comm = add_qty * comm_side
            cash -= add_comm
            entry_commission += add_comm

            # VWAP entry update
            new_qty_total = qty + add_qty
            entry_px = (float(entry_px) * qty + float(fill) * add_qty) / new_qty_total
            qty = new_qty_total
            adds_done += 1

            update_trade_levels_for_entry(pos, float(entry_px))

        # Main loop: signal already shifted, so s[i] executes on bar i open
        for i in range(1, len(df)):
            t = idx[i]

            # Daily reset
            day = t.date()
            if day != current_day:
                current_day = day
                daily_realized = 0.0

            # Mark-to-market (close)
            unreal = 0.0
            if pos != 0:
                unreal = pnl_dollars(pos, qty, float(entry_px), float(c[i]))
            equity = cash + unreal

            # ---- Correct drawdown accounting (peak-to-trough) ----
            dd = equity - peak_equity
            if self.risk.max_drawdown is not None:
                if dd <= -abs(float(self.risk.max_drawdown)):
                    if pos != 0:
                        close_position(i, "max_drawdown_flatten", float(o[i]), t)
                    stop_all_trading = True

            peak_equity = max(peak_equity, equity)

            equity_rows.append((t, cash, unreal, equity, pos))

            if stop_all_trading:
                break

            # Equity floor enforcement
            if equity <= float(self.risk.equity_floor):
                if pos != 0:
                    close_position(i, "equity_floor_flatten", float(o[i]), t)
                break

            # Daily loss gate: blocks NEW entries only
            daily_block = False
            if self.risk.max_daily_loss is not None:
                if daily_realized <= -abs(float(self.risk.max_daily_loss)):
                    daily_block = True

            desired = int(s[i])

            # -----------------------------------------
            # In-position management
            # -----------------------------------------
            if pos != 0:
                maybe_update_trailing_and_breakeven(pos, float(entry_px), float(h[i]), float(l[i]))

                # No-progress early exit (objective)
                if self.tm.no_progress_bars is not None and self.tm.no_progress_ticks is not None and entry_i >= 0:
                    bars_in = i - entry_i
                    if bars_in >= int(self.tm.no_progress_bars):
                        if best_favorable < float(self.tm.no_progress_ticks):
                            close_position(i, "no_progress_exit", float(o[i]), t)

                # Time stop: exit at current open
                if pos != 0 and self.tm.time_stop_bars is not None and entry_i >= 0:
                    if (i - entry_i) >= int(self.tm.time_stop_bars):
                        close_position(i, "time_stop", float(o[i]), t)

                # Intrabar stop/target (approx on bar i)
                if pos != 0:
                    hit = intrabar_exit_check(pos, float(h[i]), float(l[i]))
                    if hit is not None:
                        reason, level_px = hit
                        close_position(i, reason, float(level_px), t)

                # Optional add-on/pyramid (objective)
                if pos != 0 and self.tm.add_on_after_ticks is not None and self.tm.add_on_qty and self.tm.max_adds:
                    if adds_done < int(self.tm.max_adds) and best_favorable >= float(self.tm.add_on_after_ticks):
                        room = int(self.risk.max_contracts) - int(qty)
                        add_qty = min(int(self.tm.add_on_qty), max(0, room))
                        if add_qty > 0 and (not daily_block):
                            add_on_position(i, add_qty, float(o[i]), t)

            # -----------------------------------------
            # Signal-based exit/flip at i open
            # -----------------------------------------
            if pos != 0:
                # If opposite signal -> flip/exit
                if desired != 0 and desired != pos:
                    close_position(i, "signal_flip", float(o[i]), t)

                    if self.execution.allow_flip_same_bar and (not daily_block):
                        new_qty = self._determine_qty(equity)
                        if new_qty > 0:
                            open_position(i, desired, new_qty, float(o[i]), t)

                # If signal goes flat -> only exit if configured
                elif desired == 0 and self.tm.exit_on_flat_signal:
                    close_position(i, "signal_exit", float(o[i]), t)

            # -----------------------------------------
            # Entry if flat
            # -----------------------------------------
            if pos == 0 and desired != 0 and (not daily_block):
                new_qty = self._determine_qty(equity)
                if new_qty > 0:
                    open_position(i, desired, new_qty, float(o[i]), t)

        equity_df = pd.DataFrame(
            equity_rows,
            columns=["datetime", "cash", "unrealized_pnl", "equity", "position"],
        ).set_index("datetime")

        trades_df = pd.DataFrame(trades)
        return equity_df, trades_df

    # -----------------------------
    # Metrics
    # -----------------------------

    def _compute_metrics(self, equity: pd.DataFrame, trades: pd.DataFrame) -> Dict[str, Any]:
        eq = equity["equity"].astype(float)
        final_equity = float(eq.iloc[-1])
        net_profit = final_equity - self.initial_capital

        peak = eq.cummax()
        dd_dollars = eq - peak
        max_dd_dollars = float(dd_dollars.min())
        dd_pct = (eq / peak) - 1.0
        max_dd_pct = float(dd_pct.min())

        exposure_pct = float((equity["position"] != 0).mean())

        total_trades = int(len(trades))
        if total_trades == 0:
            return {
                "final_equity": final_equity,
                "net_profit": float(net_profit),
                "total_trades": 0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "expectancy": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "max_drawdown_dollars": max_dd_dollars,
                "max_drawdown_pct": max_dd_pct,
                "annualized_sharpe": 0.0,
                "annualized_sortino": 0.0,
                "exposure_pct": exposure_pct,
            }

        pnl = trades["net_pnl"].astype(float)
        wins = pnl[pnl > 0]
        losses = pnl[pnl < 0]

        win_rate = float((pnl > 0).mean())
        gross_profit = float(wins.sum())
        gross_loss = float(abs(losses.sum()))
        profit_factor = float(gross_profit / gross_loss) if gross_loss > 0 else float("inf")

        avg_win = float(wins.mean()) if len(wins) else 0.0
        avg_loss = float(losses.mean()) if len(losses) else 0.0  # negative
        expectancy = float(pnl.mean())

        # Sharpe/Sortino on bar-to-bar equity changes (minute bars)
        d = eq.diff().fillna(0.0)
        bars_per_year = 365 * 24 * 60
        mu = float(d.mean())
        sd = float(d.std(ddof=0))
        sharpe = float((mu / sd) * np.sqrt(bars_per_year)) if sd > 0 else 0.0

        downside = d[d < 0]
        dsd = float(downside.std(ddof=0))
        sortino = float((mu / dsd) * np.sqrt(bars_per_year)) if dsd > 0 else 0.0

        return {
            "final_equity": final_equity,
            "net_profit": float(net_profit),
            "total_trades": total_trades,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "expectancy": expectancy,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "max_drawdown_dollars": max_dd_dollars,
            "max_drawdown_pct": max_dd_pct,
            "annualized_sharpe": sharpe,
            "annualized_sortino": sortino,
            "exposure_pct": exposure_pct,
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

        equity_path = self.results_dir / "equity_curve.csv"
        trades_path = self.results_dir / "trades.csv"
        signals_path = self.results_dir / "signals.csv"

        equity.to_csv(equity_path)
        trades.to_csv(trades_path, index=False)

        out = pd.DataFrame({"close": df["close"], "signal": sig.astype(int)}, index=df.index)
        out.to_csv(signals_path)

        return {
            "equity_curve_path": str(equity_path),
            "trades_path": str(trades_path),
            "signals_path": str(signals_path),
        }
