import pandas as pd
import numpy as np


class LiquiditySweepStrategy:
    """
    Mechanical Liquidity Sweep + Displacement Strategy (research-grade, filterable)

    Outputs:
        signal in {-1, 0, 1}

    Key upgrades:
      - allowed_hours filter (default: [12])
      - regime_whitelist filter if df has 'regime'
      - displacement via ATR + quality checks (body_ratio, close_location)
      - cooldown between signals

    Notes:
      - Your engine already shifts signals by 1 bar and fills at next bar open.
      - This strategy is purely "signal generation"; stops/targets are engine-level if supported.
    """

    def __init__(
        self,
        sweep_lookback: int = 20,
        atr_window: int = 14,
        displacement_mult: float = 1.5,

        # Direction control
        long_only: bool = True,

        # Time filters
        allowed_hours=None,            # e.g. [12]
        allowed_weekdays=None,         # e.g. [0,1,2,3,4] (Mon-Fri)

        # Regime filters
        regime_whitelist=None,         # e.g. {"HIGH_VOL_TREND"} if df has 'regime'

        # Quality filters (help reduce garbage trades)
        min_atr_ticks: float = 6.0,    # ignore tiny-vol environments (in ticks)
        min_body_ratio: float = 0.25,  # body/(range) >= this
        min_close_loc: float = 0.2,    # close location away from extreme (avoid dojis); 0..1
        max_close_loc: float = 0.8,

        # Cooldown to avoid spam
        cooldown_bars: int = 5,

        # Long setup type:
        #   "sweep_high_reject" = (your current profitable version)
        #   "sweep_low_reclaim" = (more standard ICT long)
        long_setup: str = "sweep_high_reject",
    ):
        self.sweep_lookback = int(sweep_lookback)
        self.atr_window = int(atr_window)
        self.displacement_mult = float(displacement_mult)

        self.long_only = bool(long_only)

        self.allowed_hours = allowed_hours  # None or list[int]
        self.allowed_weekdays = allowed_weekdays  # None or list[int]

        self.regime_whitelist = set(regime_whitelist) if regime_whitelist else None

        self.min_atr_ticks = float(min_atr_ticks)
        self.min_body_ratio = float(min_body_ratio)
        self.min_close_loc = float(min_close_loc)
        self.max_close_loc = float(max_close_loc)

        self.cooldown_bars = int(cooldown_bars)
        self.long_setup = str(long_setup)

        if self.long_setup not in {"sweep_high_reject", "sweep_low_reclaim"}:
            raise ValueError("long_setup must be 'sweep_high_reject' or 'sweep_low_reclaim'.")

    def _atr(self, df: pd.DataFrame) -> pd.Series:
        high = df["high"]
        low = df["low"]
        close = df["close"]

        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(self.atr_window).mean()

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]

        # Require OHLC
        for col in ("open", "high", "low", "close"):
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        o = df["open"]
        h = df["high"]
        l = df["low"]
        c = df["close"]

        atr = self._atr(df)

        # Rolling sweep levels (prior window extremes)
        prior_high = h.rolling(self.sweep_lookback).max().shift(1)
        prior_low = l.rolling(self.sweep_lookback).min().shift(1)

        # Displacement + quality checks
        rng = (h - l).replace(0, np.nan)
        body = (c - o).abs()
        body_ratio = (body / rng).fillna(0.0)

        # Close location in bar: 0=low, 1=high
        close_loc = ((c - l) / rng).clip(0, 1).fillna(0.5)

        displacement = (h - l) > (atr * self.displacement_mult)

        # ATR floor in ticks: convert to "ticks" by dividing by tick size if available
        # If your df has 'tick_size', we'll use it; else assume MNQ 0.25
        tick_size = float(df["tick_size"].iloc[0]) if "tick_size" in df.columns else 0.25
        atr_ticks = (atr / tick_size).replace([np.inf, -np.inf], np.nan)

        signal = pd.Series(0, index=df.index, dtype=int)

        last_signal_i = -10**9

        for i in range(len(df)):
            # Warmup
            if np.isnan(prior_high.iloc[i]) or np.isnan(atr.iloc[i]):
                continue

            # Cooldown
            if (i - last_signal_i) < self.cooldown_bars:
                continue

            # Time filters
            ts = df.index[i]
            if self.allowed_hours is not None and ts.hour not in self.allowed_hours:
                continue
            if self.allowed_weekdays is not None and ts.weekday() not in self.allowed_weekdays:
                continue

            # Regime filter (only if column exists AND whitelist provided)
            if self.regime_whitelist is not None and "regime" in df.columns:
                if df["regime"].iloc[i] not in self.regime_whitelist:
                    continue

            # Quality filters
            if atr_ticks.iloc[i] < self.min_atr_ticks:
                continue
            if body_ratio.iloc[i] < self.min_body_ratio:
                continue
            if not (self.min_close_loc <= close_loc.iloc[i] <= self.max_close_loc):
                continue
            if not displacement.iloc[i]:
                continue

            # -------------------------
            # LONG SETUPS
            # -------------------------
            if self.long_setup == "sweep_high_reject":
                # (Your current profitable logic)
                # sweep above prior_high then close back below prior_high
                sweep_high = h.iloc[i] > prior_high.iloc[i]
                reject_back_in = c.iloc[i] < prior_high.iloc[i]

                if sweep_high and reject_back_in:
                    signal.iloc[i] = 1
                    last_signal_i = i
                    continue

            elif self.long_setup == "sweep_low_reclaim":
                # (More standard ICT long)
                # sweep below prior_low then close back above prior_low
                sweep_low = l.iloc[i] < prior_low.iloc[i]
                reclaim = c.iloc[i] > prior_low.iloc[i]

                if sweep_low and reclaim:
                    signal.iloc[i] = 1
                    last_signal_i = i
                    continue

            # -------------------------
            # SHORT SETUP (optional)
            # -------------------------
            if not self.long_only:
                # Standard short: sweep above prior_high then close back below prior_high
                sweep_high = h.iloc[i] > prior_high.iloc[i]
                reject = c.iloc[i] < prior_high.iloc[i]

                if sweep_high and reject:
                    signal.iloc[i] = -1
                    last_signal_i = i
                    continue

        return signal
