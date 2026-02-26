# src/backtesting/engine.py
"""
YAMBOT Backtesting Engine v2

Key features:
  - next_open fill mode (realistic: signal on bar close, fill on next open)
  - Per-trade SL / TP / trailing stop / breakeven / time stop
  - No-progress exit (chop filter)
  - Prop-firm daily loss + max drawdown circuit breakers
  - ATR-based or fixed contract sizing
  - Full trade blotter with entry/exit reason
  - Regime-aware performance breakdown
  - Sharpe, Sortino, Calmar, exposure %
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ContractSpecs:
    tick_size: float = 0.25
    tick_value: float = 0.50         # MNQ $0.50/tick; NQ $5.00/tick
    commission_per_side: float = 0.35


@dataclass
class ExecutionConfig:
    fill_mode: str = "next_open"     # "next_open" | "close"
    slippage_ticks: float = 1.5
    allow_flip_same_bar: bool = False


@dataclass
class RiskConfig:
    equity_floor: float = 0.0
    max_contracts: int = 10
    max_daily_loss: Optional[float] = None
    max_drawdown: Optional[float] = None


@dataclass
class SizingConfig:
    mode: str = "fixed"              # "fixed" | "atr_risk"
    fixed_qty: int = 1
    risk_per_trade: float = 0.01
    atr_window: int = 14
    atr_stop_mult: float = 2.0


@dataclass
class TradeManagementConfig:
    stop_loss_ticks: Optional[float] = 30
    take_profit_ticks: Optional[float] = 75
    trailing_stop_ticks: Optional[float] = None
    breakeven_after_ticks: Optional[float] = 25
    time_stop_bars: Optional[int] = 60
    exit_on_flat_signal: bool = False
    no_progress_bars: Optional[int] = 10
    no_progress_ticks: Optional[float] = 8.0
    add_on_after_ticks: Optional[float] = None
    add_on_qty: int = 0
    max_adds: int = 0


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class BacktestEngine:

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
        self.data = self._prepare(data)
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

        self._tick = float(self.specs.tick_size)
        self._tick_val = float(self.specs.tick_value)
        self._pt_val = self._tick_val / self._tick
        self._comm = float(self.specs.commission_per_side)
        self._slip_pts = float(self.execution.slippage_ticks) * self._tick

        if self.execution.fill_mode not in ("next_open", "close"):
            raise ValueError("fill_mode must be 'next_open' or 'close'")

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def run(self) -> Dict[str, Any]:
        df = self.data.copy()
        raw = self.strategy.generate_signals(df)
        signals = self._extract_signals(raw, df.index)
        atr_series = self._compute_atr(df)
        equity_rows, trades = self._simulate(df, signals, atr_series)

        equity_df = pd.DataFrame(
            equity_rows,
            columns=["datetime", "cash", "unrealized", "equity", "position", "contracts"],
        ).set_index("datetime")

        trades_df = pd.DataFrame(trades)
        metrics = self._compute_metrics(equity_df, trades_df)

        if self.export_artifacts:
            self._save(equity_df, trades_df, signals)

        return {"metrics": metrics, "equity": equity_df, "trades": trades_df, "signals": signals}

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------

    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]

        if not isinstance(df.index, pd.DatetimeIndex):
            if "datetime" in df.columns:
                df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
                df = df.set_index("datetime")
            else:
                df.index = pd.to_datetime(df.index, utc=True)

        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")

        df = df.sort_index()
        if df.index.has_duplicates:
            df = df[~df.index.duplicated(keep="last")]

        for col in ("open", "high", "low", "close"):
            if col not in df.columns:
                raise ValueError(f"Missing required column: '{col}'")
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=["open", "high", "low", "close"])
        if df.empty:
            raise ValueError("Data is empty after cleaning.")
        return df

    # ------------------------------------------------------------------
    # Signal extraction
    # ------------------------------------------------------------------

    def _extract_signals(self, raw: Any, index: pd.Index) -> pd.Series:
        if isinstance(raw, pd.Series):
            sig = raw.copy()
        elif isinstance(raw, pd.DataFrame):
            cols_lower = {c.lower(): c for c in raw.columns}
            key = cols_lower.get(self.signal_col.lower()) or cols_lower.get("signal")
            if key is None:
                raise ValueError(
                    f"Strategy DataFrame missing '{self.signal_col}' column. Got: {list(raw.columns)}"
                )
            sig = raw[key].copy()
        else:
            raise TypeError("strategy.generate_signals must return a Series or DataFrame.")

        sig = sig.reindex(index)
        sig = pd.to_numeric(sig, errors="coerce").fillna(0)
        return sig.clip(-1, 1).round().astype(int)

    # ------------------------------------------------------------------
    # ATR
    # ------------------------------------------------------------------

    def _compute_atr(self, df: pd.DataFrame) -> pd.Series:
        w = int(self.sizing.atr_window)
        h, l, c = df["high"], df["low"], df["close"]
        tr = pd.concat([(h - l).abs(), (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
        return tr.rolling(w).mean().bfill()

    # ------------------------------------------------------------------
    # Position sizing
    # ------------------------------------------------------------------

    def _calc_size(self, equity: float, atr_val: float) -> int:
        if self.sizing.mode == "fixed":
            qty = int(self.sizing.fixed_qty)
        elif self.sizing.mode == "atr_risk":
            if atr_val <= 0 or not np.isfinite(atr_val):
                qty = 1
            else:
                stop_pts = atr_val * float(self.sizing.atr_stop_mult)
                risk_dollars = equity * float(self.sizing.risk_per_trade)
                qty = max(1, int(risk_dollars / (stop_pts * self._pt_val)))
        else:
            raise ValueError(f"Unknown sizing mode: {self.sizing.mode}")
        return min(qty, int(self.risk.max_contracts))

    # ------------------------------------------------------------------
    # Main simulation loop
    # ------------------------------------------------------------------

    def _simulate(self, df, signals, atr):
        n = len(df)
        opens = df["open"].to_numpy(float)
        highs = df["high"].to_numpy(float)
        lows = df["low"].to_numpy(float)
        closes = df["close"].to_numpy(float)
        timestamps = df.index
        sig_arr = signals.to_numpy(int)
        atr_arr = atr.to_numpy(float)

        # Fill price series
        if self.execution.fill_mode == "next_open":
            fill_prices = np.roll(opens, -1)
            fill_prices[-1] = closes[-1]
        else:
            fill_prices = closes.copy()

        cash = self.initial_capital
        position = 0
        contracts = 0
        entry_price = np.nan
        entry_time = None
        entry_bar = -1
        sl_price = np.nan
        tp_price = np.nan
        trail_extreme = np.nan

        trades: List[Dict] = []
        equity_rows: List[Tuple] = []

        current_day = None
        day_start_equity = cash
        peak_equity = cash

        for i in range(n):
            ts = timestamps[i]
            day = ts.date()

            if day != current_day:
                current_day = day
                day_start_equity = cash

            # Mark-to-market
            unreal = 0.0
            if position != 0 and np.isfinite(entry_price):
                unreal = (closes[i] - entry_price) * self._pt_val * contracts * position

            equity = cash + unreal
            peak_equity = max(peak_equity, equity)
            equity_rows.append((ts, cash, unreal, equity, position, contracts))

            # --- Circuit breakers ---
            if equity <= self.risk.equity_floor:
                log.warning(f"Equity floor hit at {ts}.")
                break

            if self.risk.max_drawdown is not None:
                if (peak_equity - equity) >= self.risk.max_drawdown:
                    log.warning(f"Max drawdown limit hit at {ts}.")
                    break

            daily_loss_tripped = False
            if self.risk.max_daily_loss is not None:
                if (day_start_equity - equity) >= self.risk.max_daily_loss:
                    log.warning(f"Daily loss limit hit at {ts}.")
                    daily_loss_tripped = True

            # --- In-trade exits ---
            if position != 0:
                # Update trailing extreme
                if position == 1:
                    if not np.isfinite(trail_extreme):
                        trail_extreme = highs[i]
                    else:
                        trail_extreme = max(trail_extreme, highs[i])
                else:
                    if not np.isfinite(trail_extreme):
                        trail_extreme = lows[i]
                    else:
                        trail_extreme = min(trail_extreme, lows[i])

                # Update trailing stop
                if self.tm.trailing_stop_ticks and np.isfinite(trail_extreme):
                    trail_dist = float(self.tm.trailing_stop_ticks) * self._tick
                    if position == 1:
                        new_tsl = trail_extreme - trail_dist
                        sl_price = max(sl_price, new_tsl) if np.isfinite(sl_price) else new_tsl
                    else:
                        new_tsl = trail_extreme + trail_dist
                        sl_price = min(sl_price, new_tsl) if np.isfinite(sl_price) else new_tsl

                # Breakeven
                if self.tm.breakeven_after_ticks and np.isfinite(entry_price) and np.isfinite(sl_price):
                    be_dist = float(self.tm.breakeven_after_ticks) * self._tick
                    if position == 1 and closes[i] >= entry_price + be_dist:
                        sl_price = max(sl_price, entry_price)
                    elif position == -1 and closes[i] <= entry_price - be_dist:
                        sl_price = min(sl_price, entry_price)

                exited, reason, exit_px = self._check_exits(
                    i, position, entry_price, entry_bar,
                    sl_price, tp_price, highs[i], lows[i], closes[i],
                    fill_prices[i], sig_arr[i], daily_loss_tripped
                )

                if exited:
                    market_exit_reasons = {"signal_exit", "time_stop", "no_progress", "daily_loss_stop"}
                    if reason in market_exit_reasons:
                        exit_px = self._fill(exit_px, -position)
                    gross, net = self._pnl(entry_price, exit_px, position, contracts)
                    cash += net
                    trades.append(self._trade_record(
                        entry_time, ts, position, contracts,
                        entry_price, exit_px, gross, net, reason
                    ))
                    position = 0; contracts = 0
                    entry_price = np.nan; sl_price = np.nan; tp_price = np.nan
                    trail_extreme = np.nan

                    if daily_loss_tripped or not self.execution.allow_flip_same_bar:
                        continue

            # --- New entry ---
            desired = int(sig_arr[i])
            if desired != 0 and position == 0 and not daily_loss_tripped:
                fill_px = self._fill(fill_prices[i], desired)
                qty = self._calc_size(equity, float(atr_arr[i]))
                cash -= self._comm * qty
                position = desired
                contracts = qty
                entry_price = fill_px
                entry_time = ts
                entry_bar = i
                trail_extreme = fill_px
                sl_price, tp_price = self._calc_sl_tp(fill_px, desired)

        # Close open position at end
        if position != 0 and np.isfinite(entry_price):
            exit_px = self._fill(closes[-1], -position)
            gross, net = self._pnl(entry_price, exit_px, position, contracts)
            cash += net
            trades.append(self._trade_record(
                entry_time, timestamps[-1], position, contracts,
                entry_price, exit_px, gross, net, "end_of_data"
            ))

        return equity_rows, trades

    # ------------------------------------------------------------------
    # Exit checks
    # ------------------------------------------------------------------

    def _check_exits(
        self, i, position, entry_price, entry_bar,
        sl_price, tp_price, high, low, close, fill_px,
        current_signal, daily_loss_tripped
    ) -> Tuple[bool, str, float]:

        if daily_loss_tripped:
            return True, "daily_loss_stop", fill_px

        if np.isfinite(sl_price):
            if position == 1 and low <= sl_price:
                return True, "stop_loss", sl_price
            if position == -1 and high >= sl_price:
                return True, "stop_loss", sl_price

        if np.isfinite(tp_price):
            if position == 1 and high >= tp_price:
                return True, "take_profit", tp_price
            if position == -1 and low <= tp_price:
                return True, "take_profit", tp_price

        bars_in_trade = i - entry_bar
        if self.tm.time_stop_bars and bars_in_trade >= int(self.tm.time_stop_bars):
            return True, "time_stop", fill_px

        if self.tm.no_progress_bars and self.tm.no_progress_ticks:
            if bars_in_trade >= int(self.tm.no_progress_bars):
                min_move = float(self.tm.no_progress_ticks) * self._tick
                if (close - entry_price) * position < min_move:
                    return True, "no_progress", fill_px

        if self.tm.exit_on_flat_signal and current_signal == 0:
            return True, "signal_exit", fill_px

        return False, "", np.nan

    # ------------------------------------------------------------------
    # Price helpers
    # ------------------------------------------------------------------

    def _fill(self, price: float, side: int) -> float:
        return float(price) + float(side) * self._slip_pts

    def _calc_sl_tp(self, entry: float, direction: int) -> Tuple[float, float]:
        sl = (entry - direction * float(self.tm.stop_loss_ticks) * self._tick
              if self.tm.stop_loss_ticks else np.nan)
        tp = (entry + direction * float(self.tm.take_profit_ticks) * self._tick
              if self.tm.take_profit_ticks else np.nan)
        return sl, tp

    def _pnl(self, entry: float, exit_px: float, direction: int, qty: int) -> Tuple[float, float]:
        gross = (exit_px - entry) * direction * self._pt_val * qty
        net = gross - self._comm * qty
        return gross, net

    @staticmethod
    def _trade_record(entry_time, exit_time, direction, contracts,
                      entry_price, exit_price, gross_pnl, net_pnl, exit_reason):
        return {
            "entry_time": entry_time,
            "exit_time": exit_time,
            "side": "LONG" if direction == 1 else "SHORT",
            "contracts": contracts,
            "entry_price": float(entry_price),
            "exit_price": float(exit_price),
            "gross_pnl": float(gross_pnl),
            "net_pnl": float(net_pnl),
            "exit_reason": exit_reason,
        }

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def _compute_metrics(self, equity: pd.DataFrame, trades: pd.DataFrame) -> Dict[str, Any]:
        eq = equity["equity"].astype(float)
        final_equity = float(eq.iloc[-1])
        net_profit = final_equity - self.initial_capital
        total_trades = len(trades)

        empty_metrics = {
            "final_equity": final_equity, "net_profit": net_profit,
            "total_trades": 0, "win_rate": 0.0, "profit_factor": 0.0,
            "expectancy": 0.0, "avg_win": 0.0, "avg_loss": 0.0,
            "max_drawdown_dollars": 0.0, "max_drawdown_pct": 0.0,
            "annualized_sharpe": 0.0, "annualized_sortino": 0.0,
            "calmar": 0.0, "exposure_pct": 0.0,
            "exit_reasons": {}, "regime_breakdown": [],
        }
        if total_trades == 0:
            return empty_metrics

        pnl = trades["net_pnl"].astype(float)
        wins = pnl[pnl > 0]
        losses = pnl[pnl < 0]

        win_rate = float((pnl > 0).mean())
        gross_profit = float(wins.sum()) if len(wins) else 0.0
        gross_loss = float(losses.sum()) if len(losses) else 0.0
        profit_factor = float(gross_profit / abs(gross_loss)) if gross_loss != 0 else float("inf")
        avg_win = float(wins.mean()) if len(wins) else 0.0
        avg_loss = float(losses.mean()) if len(losses) else 0.0
        expectancy = float(pnl.mean())

        # Drawdown
        peak = eq.cummax()
        dd_abs = eq - peak
        dd_pct = (dd_abs / peak.replace(0, np.nan))
        max_dd_dollars = float(dd_abs.min())
        max_dd_pct = float(dd_pct.min())

        # Returns
        rets = eq.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
        annual_factor = np.sqrt(252 * 390)
        sharpe = sortino = 0.0
        if len(rets) > 2 and rets.std(ddof=1) > 0:
            sharpe = float((rets.mean() / rets.std(ddof=1)) * annual_factor)
            downside = rets[rets < 0]
            ds_std = downside.std(ddof=1) if len(downside) > 1 else np.nan
            if ds_std and ds_std > 0:
                sortino = float((rets.mean() / ds_std) * annual_factor)

        # Calmar
        n_years = (eq.index[-1] - eq.index[0]).days / 365.25
        ann_return = (final_equity / self.initial_capital) ** (1 / max(n_years, 1e-6)) - 1
        calmar = float(ann_return / abs(max_dd_pct)) if max_dd_pct != 0 else 0.0

        exposure_pct = float(equity["position"].astype(bool).mean())

        exit_reasons = {}
        if "exit_reason" in trades.columns:
            exit_reasons = trades["exit_reason"].value_counts().to_dict()

        regime_breakdown = self._regime_breakdown(trades)

        return {
            "final_equity": final_equity,
            "net_profit": net_profit,
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
            "calmar": calmar,
            "exposure_pct": exposure_pct,
            "exit_reasons": exit_reasons,
            "regime_breakdown": regime_breakdown,
        }

    def _regime_breakdown(self, trades: pd.DataFrame) -> List[Dict]:
        if "regime" not in self.data.columns or trades.empty:
            return []
        try:
            regime_at_entry = self.data["regime"].reindex(
                pd.DatetimeIndex(trades["entry_time"])
            ).values
            breakdown = []
            for regime_name in pd.Series(regime_at_entry).dropna().unique():
                mask = regime_at_entry == regime_name
                group_pnl = trades.loc[mask, "net_pnl"].astype(float)
                breakdown.append({
                    "regime": regime_name,
                    "trades": int(mask.sum()),
                    "win_rate": float((group_pnl > 0).mean()),
                    "total_pnl": float(group_pnl.sum()),
                    "avg_pnl": float(group_pnl.mean()),
                })
            return breakdown
        except Exception:
            return []

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self, equity, trades, signals):
        self.results_dir.mkdir(parents=True, exist_ok=True)
        equity.to_csv(self.results_dir / "equity_curve.csv")
        if not trades.empty:
            trades.to_csv(self.results_dir / "trades.csv", index=False)
        pd.DataFrame({"close": self.data["close"], "signal": signals}).to_csv(
            self.results_dir / "signals.csv"
        )
