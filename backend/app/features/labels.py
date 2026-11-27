"""Triple-barrier labels for long and short trades on 5-minute candles.

For a decision at the close of candle i:
- entry at the OPEN of candle i+1 (the first price we could actually get)
- long: target = entry + target_mult * ATR_i, stop = entry - stop_mult * ATR_i (short mirrored)
- the trade is closed at the day's exit time (our square-off) if neither barrier was hit
- if one candle touches both barriers we assume the stop came first (conservative)
- a candle that opens beyond a barrier fills at its open (gaps)

y_long / y_short = 1 if the target was hit first, else 0. ret_* = gross return, before costs.
Only decisions inside the entry window get labels; other rows are NaN.
"""

from __future__ import annotations

from datetime import time

import numpy as np
import pandas as pd

LABEL_COLUMNS = ["entry_price", "y_long", "y_short", "ret_long", "ret_short", "bars_long", "bars_short"]


def _first_true(mask: np.ndarray) -> np.ndarray:
    """Index of the first True in each row, or -1."""
    any_ = mask.any(axis=1)
    return np.where(any_, mask.argmax(axis=1), -1)


def _side(o, h, lo, c, entry, atr, tm, sm, valid, last, long: bool):
    n = len(o)
    out_ret = np.full(n, np.nan)
    out_y = np.full(n, np.nan)
    out_bars = np.full(n, np.nan)
    if long:
        target, stop = entry + tm * atr, entry - sm * atr
        hit_t = h[None, :] >= target[:, None]
        hit_s = lo[None, :] <= stop[:, None]
    else:
        target, stop = entry - tm * atr, entry + sm * atr
        hit_t = lo[None, :] <= target[:, None]
        hit_s = h[None, :] >= stop[:, None]
    j = np.arange(n)[None, :]
    i = np.arange(n)[:, None]
    window = (j >= i + 1) & (j <= last)          # path: entry candle .. last candle before exit time
    ft, fs = _first_true(hit_t & window), _first_true(hit_s & window)
    for k in np.nonzero(valid)[0]:
        t, s = ft[k], fs[k]
        if s >= 0 and (t < 0 or s <= t):          # stop first (ties -> stop)
            px = min(stop[k], o[s]) if long else max(stop[k], o[s])
            out_y[k], out_bars[k] = 0.0, s - k
        elif t >= 0:
            px = max(target[k], o[t]) if long else min(target[k], o[t])
            out_y[k], out_bars[k] = 1.0, t - k
        else:
            px = c[last]
            out_y[k], out_bars[k] = 0.0, last - k
        out_ret[k] = (px / entry[k] - 1) if long else (entry[k] / px - 1)
    return out_y, out_ret, out_bars


def triple_barrier(df: pd.DataFrame, exit_time: dict, entry_start: time = time(9, 30),
                   entry_end: time = time(14, 30), target_mult: float = 1.5, stop_mult: float = 1.0,
                   minutes: int = 5, unit_col: str = "atr14") -> pd.DataFrame:
    """df: sorted 5-minute frame with ts (IST), day, open/high/low/close and the barrier unit column.
    exit_time: {day: time} when open trades are closed that day (our square-off).
    unit_col: rupee distance of one barrier unit (default the 5-min ATR; any column can be used)."""
    out = pd.DataFrame(np.nan, index=df.index, columns=LABEL_COLUMNS)
    for day, g in df.groupby("day", sort=False):
        ex = exit_time.get(day)
        if ex is None or len(g) < 3:
            continue
        idx = g.index.to_numpy()
        o, h, lo, c = (g[col].to_numpy(dtype=float) for col in ("open", "high", "low", "close"))
        atr = g[unit_col].to_numpy(dtype=float)
        end_t = (g["ts"] + pd.Timedelta(minutes=minutes)).dt.time.to_numpy()
        tradable = np.array([e <= ex for e in end_t])     # candles that finish by the exit time
        if not tradable.any():
            continue
        last = int(np.nonzero(tradable)[0].max())
        n = len(g)
        entry = np.full(n, np.nan)
        entry[:-1] = o[1:]
        decision = np.array([entry_start <= e <= entry_end for e in end_t])  # candle close inside window
        valid = decision & (np.arange(n) < last) & ~np.isnan(atr) & (atr > 0) & ~np.isnan(entry)
        if not valid.any():
            continue
        yl, rl, bl = _side(o, h, lo, c, entry, atr, target_mult, stop_mult, valid, last, long=True)
        ys, rs, bs = _side(o, h, lo, c, entry, atr, target_mult, stop_mult, valid, last, long=False)
        out.loc[idx, "entry_price"] = np.where(valid, entry, np.nan)
        out.loc[idx, "y_long"], out.loc[idx, "ret_long"], out.loc[idx, "bars_long"] = yl, rl, bl
        out.loc[idx, "y_short"], out.loc[idx, "ret_short"], out.loc[idx, "bars_short"] = ys, rs, bs
    return out
