# src/utils/smc.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List
import pandas as pd
import numpy as np


@dataclass
class FVG:
    direction: int               # +1 bullish, -1 bearish
    created_at: pd.Timestamp
    low: float                   # lower bound of zone (price)
    high: float                  # upper bound of zone (price)
    mid: float                   # midpoint for entry
    expiry_bars: int


def ensure_utc_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index, utc=True)
    if out.index.tz is None:
        out.index = out.index.tz_localize("UTC")
    else:
        out.index = out.index.tz_convert("UTC")
    return out.sort_index()


def resample_ohlc(df_1m: pd.DataFrame, rule: str) -> pd.DataFrame:
    """
    Resample 1m OHLCV to higher timeframe.
    Requires columns: open, high, low, close. Volume optional.
    """
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
    }
    if "volume" in df_1m.columns:
        agg["volume"] = "sum"

    # Normalize deprecated aliases (e.g. '5T' -> '5min', '1D' stays '1D')
    rule = rule.replace("T", "min") if rule.endswith("T") or "T" in rule else rule
    out = df_1m.resample(rule, label="right", closed="right").agg(agg).dropna()
    return out


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    c = df["close"].astype(float)
    prev_c = c.shift(1)
    tr = pd.concat([(h - l).abs(), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    return tr.rolling(window).mean()


def fractal_swings(df: pd.DataFrame, left: int = 2, right: int = 2) -> Tuple[pd.Series, pd.Series]:
    """
    Returns (swing_high, swing_low) series with swing levels at the swing bar, else NaN.
    """
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)

    swing_high = np.full(len(df), np.nan, dtype=float)
    swing_low = np.full(len(df), np.nan, dtype=float)

    for i in range(left, len(df) - right):
        window_h = h[i - left : i + right + 1]
        window_l = l[i - left : i + right + 1]
        if np.argmax(window_h) == left:
            swing_high[i] = h[i]
        if np.argmin(window_l) == left:
            swing_low[i] = l[i]

    return pd.Series(swing_high, index=df.index), pd.Series(swing_low, index=df.index)


def ny_killzone_mask_utc(index_utc: pd.DatetimeIndex, start_et="09:30", end_et="11:00") -> pd.Series:
    """
    Mask for NY Open Killzone using America/New_York, returned aligned to UTC index.
    """
    # convert to NY time for filtering
    ny = index_utc.tz_convert("America/New_York")
    start_h, start_m = map(int, start_et.split(":"))
    end_h, end_m = map(int, end_et.split(":"))
    mins = ny.hour * 60 + ny.minute
    start = start_h * 60 + start_m
    end = end_h * 60 + end_m
    return pd.Series((mins >= start) & (mins <= end), index=index_utc)


def daily_levels(df_1m: pd.DataFrame) -> pd.DataFrame:
    """
    Compute prior day high/low for each timestamp.
    Returns columns: pdh, pdl, day_close_prev, day_high_prev, day_low_prev
    """
    # day boundaries in NY time are more "session-like", but for v1 we use UTC days consistently.
    df = df_1m.copy()
    daily = df.resample("1D").agg({"high": "max", "low": "min", "close": "last"})
    daily = daily.rename(columns={"high": "day_high", "low": "day_low", "close": "day_close"})

    prev = daily.shift(1)
    prev = prev.rename(columns={
        "day_high": "pdh",
        "day_low": "pdl",
        "day_close": "day_close_prev"
    })

    # forward fill prior day levels onto intraday index
    out = pd.DataFrame(index=df.index)
    out["pdh"] = prev["pdh"].reindex(out.index, method="ffill")
    out["pdl"] = prev["pdl"].reindex(out.index, method="ffill")
    out["day_close_prev"] = prev["day_close_prev"].reindex(out.index, method="ffill")

    # keep also prev day high/low if needed later
    out["day_high_prev"] = prev["pdh"].reindex(out.index, method="ffill")
    out["day_low_prev"] = prev["pdl"].reindex(out.index, method="ffill")
    return out


def daily_bias_from_range_expansion(df_1m: pd.DataFrame) -> pd.Series:
    """
    Bias rule (objective, v1):
      bullish if D-1 close > D-2 high
      bearish if D-1 close < D-2 low
      else 0
    Bias is applied to day D.
    """
    daily = df_1m.resample("1D").agg({"high": "max", "low": "min", "close": "last"}).dropna()
    d2 = daily.shift(2)  # D-2
    d1 = daily.shift(1)  # D-1

    bias_daily = pd.Series(0, index=daily.index, dtype=int)
    bias_daily[d1["close"] > d2["high"]] = 1
    bias_daily[d1["close"] < d2["low"]] = -1

    # forward fill onto intraday timestamps
    bias_1m = bias_daily.reindex(df_1m.index, method="ffill").fillna(0).astype(int)
    return bias_1m


def detect_fvg(df_1m: pd.DataFrame, max_age_bars: int = 60) -> List[FVG]:
    """
    Detects FVGs (3-candle imbalance):
      bullish: low[i] > high[i-2] => zone [high[i-2], low[i]]
      bearish: high[i] < low[i-2] => zone [high[i], low[i-2]] (note ordering)
    Returns list of FVG objects (created_at = candle i).
    """
    o = df_1m["open"].to_numpy(float)
    h = df_1m["high"].to_numpy(float)
    l = df_1m["low"].to_numpy(float)
    idx = df_1m.index

    fvgs: List[FVG] = []
    for i in range(2, len(df_1m)):
        # bullish fvg
        if l[i] > h[i - 2]:
            low_zone = float(h[i - 2])
            high_zone = float(l[i])
            mid = (low_zone + high_zone) / 2.0
            fvgs.append(FVG(direction=1, created_at=idx[i], low=low_zone, high=high_zone, mid=mid, expiry_bars=max_age_bars))

        # bearish fvg
        if h[i] < l[i - 2]:
            low_zone = float(h[i])       # lower price
            high_zone = float(l[i - 2])   # higher price
            mid = (low_zone + high_zone) / 2.0
            fvgs.append(FVG(direction=-1, created_at=idx[i], low=low_zone, high=high_zone, mid=mid, expiry_bars=max_age_bars))

    return fvgs
