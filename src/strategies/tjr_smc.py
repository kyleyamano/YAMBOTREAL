# src/strategies/tjr_smc.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, List

import pandas as pd
import numpy as np

from src.utils.smc import (
    ensure_utc_index,
    resample_ohlc,
    fractal_swings,
    daily_levels,
    daily_bias_from_range_expansion,
    ny_killzone_mask_utc,
    detect_fvg,
    FVG,
)


@dataclass
class TJRSMCConfig:
    # Sweep
    sweep_buffer_points: float = 2.0         # how far beyond PDH/PDL to count as sweep
    sweep_max_bars: int = 6                  # sweep must reclaim within N bars

    # BOS on 5m
    bos_swing_left: int = 2
    bos_swing_right: int = 2
    bos_lookback_swings: int = 8             # how many recent swing levels to consider

    # FVG entry
    fvg_max_age_bars: int = 60               # how many 1m bars FVG remains valid
    entry_requires_bias: bool = True         # only take longs if daily bias bullish, shorts if bearish

    # Time filter
    killzone_only: bool = True
    killzone_start_et: str = "09:30"
    killzone_end_et: str = "11:00"

    # Cooldown
    cooldown_bars: int = 20                  # after a trade signal, wait N bars before next

    # Safety
    max_signals_per_day: int = 2


class TJRSMCStrategy:
    """
    Testable v1: PDH/PDL sweep -> BOS (5m) -> FVG midpoint retrace entry.

    Output: pd.Series of {-1,0,1} aligned to 1m bars.
    Engine will shift signal by 1 bar automatically (next-open fill). :contentReference[oaicite:2]{index=2}
    """

    def __init__(self, **kwargs):
        self.cfg = TJRSMCConfig(**kwargs)

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]
        df = ensure_utc_index(df)

        # Required columns
        for col in ("open", "high", "low", "close"):
            if col not in df.columns:
                raise ValueError(f"TJRSMCStrategy requires column '{col}'")

        # Daily levels + daily bias
        levels = daily_levels(df)
        bias = daily_bias_from_range_expansion(df)  # +1 / -1 / 0 per day

        # Killzone mask
        if self.cfg.killzone_only:
            kz = ny_killzone_mask_utc(df.index, self.cfg.killzone_start_et, self.cfg.killzone_end_et)
        else:
            kz = pd.Series(True, index=df.index)

        # Build 5m frame for BOS detection
        df5 = resample_ohlc(df, "5T")
        sh, sl = fractal_swings(df5, left=self.cfg.bos_swing_left, right=self.cfg.bos_swing_right)

        # Track latest swing levels (forward-filled)
        last_swing_high = sh.ffill()
        last_swing_low = sl.ffill()

        # Forward-fill 5m swing levels back to 1m
        last_swing_high_1m = last_swing_high.reindex(df.index, method="ffill")
        last_swing_low_1m = last_swing_low.reindex(df.index, method="ffill")

        # Sweep detection on 1m against PDH/PDL
        hi = df["high"].astype(float)
        lo = df["low"].astype(float)
        cl = df["close"].astype(float)

        pdh = levels["pdh"].astype(float)
        pdl = levels["pdl"].astype(float)

        buf = float(self.cfg.sweep_buffer_points)

        # Sweep events:
        # - bearish sweep: take PDH liquidity then close back below PDH
        # - bullish sweep: take PDL liquidity then close back above PDL
        bearish_sweep = (hi >= (pdh + buf)) & (cl < pdh)
        bullish_sweep = (lo <= (pdl - buf)) & (cl > pdl)

        # For each sweep, we require "reclaim" within sweep_max_bars.
        # Since we already require close back through on same bar, this is mostly redundancy,
        # but we keep window gating for stability.
        sweep_max = int(self.cfg.sweep_max_bars)
        bear_ok = bearish_sweep.rolling(sweep_max, min_periods=1).max().astype(bool)
        bull_ok = bullish_sweep.rolling(sweep_max, min_periods=1).max().astype(bool)

        # BOS confirmation (simple):
        # After bullish sweep, BOS long when close > last swing high (5m structure)
        # After bearish sweep, BOS short when close < last swing low
        bos_long = cl > last_swing_high_1m
        bos_short = cl < last_swing_low_1m

        # FVG detection list (all fvgs)
        fvgs: List[FVG] = detect_fvg(df, max_age_bars=self.cfg.fvg_max_age_bars)

        # Create a time->list of fvgs created at that bar for quick access
        fvg_by_time: Dict[pd.Timestamp, List[FVG]] = {}
        for f in fvgs:
            fvg_by_time.setdefault(f.created_at, []).append(f)

        signals = pd.Series(0, index=df.index, dtype=int)

        # State machine:
        # 1) see sweep in killzone
        # 2) wait for BOS in same direction (reversal direction)
        # 3) after BOS, wait for a fresh FVG in that direction
        # 4) enter when price retraces to FVG midpoint
        waiting_dir = 0           # +1 looking long, -1 looking short
        sweep_time: Optional[pd.Timestamp] = None
        bos_time: Optional[pd.Timestamp] = None
        active_fvg: Optional[FVG] = None
        cooldown = 0
        signals_today = 0
        current_day = df.index[0].date()

        def reset_daily(t: pd.Timestamp):
            nonlocal signals_today, current_day
            day = t.date()
            if day != current_day:
                current_day = day
                signals_today = 0

        for i, t in enumerate(df.index):
            reset_daily(t)

            if cooldown > 0:
                cooldown -= 1
                continue

            if signals_today >= int(self.cfg.max_signals_per_day):
                continue

            if not bool(kz.iloc[i]):
                continue

            # Bias filter
            b = int(bias.iloc[i])
            if self.cfg.entry_requires_bias and b == 0:
                # No bias days: skip
                continue

            # ---------------------------------------------------
            # Step A: Detect sweep -> set waiting direction
            # ---------------------------------------------------
            if waiting_dir == 0:
                # bullish sweep implies long reversal setup
                if bool(bull_ok.iloc[i]):
                    if (not self.cfg.entry_requires_bias) or (b >= 0):
                        waiting_dir = 1
                        sweep_time = t
                        bos_time = None
                        active_fvg = None
                        continue

                # bearish sweep implies short reversal setup
                if bool(bear_ok.iloc[i]):
                    if (not self.cfg.entry_requires_bias) or (b <= 0):
                        waiting_dir = -1
                        sweep_time = t
                        bos_time = None
                        active_fvg = None
                        continue

            # ---------------------------------------------------
            # Step B: Wait for BOS in that direction
            # ---------------------------------------------------
            if waiting_dir != 0 and bos_time is None:
                if waiting_dir == 1 and bool(bos_long.iloc[i]):
                    bos_time = t
                    # reset active fvg; we want a fresh one after BOS
                    active_fvg = None
                    continue
                if waiting_dir == -1 and bool(bos_short.iloc[i]):
                    bos_time = t
                    active_fvg = None
                    continue

            # ---------------------------------------------------
            # Step C: After BOS, pick first fresh FVG in direction
            # ---------------------------------------------------
            if waiting_dir != 0 and bos_time is not None and active_fvg is None:
                created = fvg_by_time.get(t, [])
                # choose the first FVG in the direction of the setup
                for f in created:
                    if f.direction == waiting_dir:
                        active_fvg = f
                        break
                # if none created now, keep waiting
                continue

            # ---------------------------------------------------
            # Step D: Wait for retrace to FVG midpoint to enter
            # ---------------------------------------------------
            if active_fvg is not None:
                # expire by age in bars
                age = (df.index.get_loc(t) - df.index.get_loc(active_fvg.created_at))
                if age > int(active_fvg.expiry_bars):
                    # stale, reset and wait for next fvg
                    active_fvg = None
                    continue

                # midpoint touch check
                mid = float(active_fvg.mid)
                if (lo.iloc[i] <= mid <= hi.iloc[i]):
                    # issue entry signal
                    signals.iloc[i] = int(waiting_dir)
                    signals_today += 1
                    cooldown = int(self.cfg.cooldown_bars)

                    # reset state after signal
                    waiting_dir = 0
                    sweep_time = None
                    bos_time = None
                    active_fvg = None

        return signals
