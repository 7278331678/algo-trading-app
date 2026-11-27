"""Technical indicators (pandas/numpy, causal: each value uses only current and past bars).

Conventions follow the common charting definitions:
- EMA: alpha = 2/(n+1), seeded with the first value (pandas ewm adjust=False)
- Wilder smoothing (RSI, ATR, ADX): alpha = 1/n
Intraday indicators that reset each session (VWAP, opening range) group by IST date.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def wilder(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    gain = wilder(d.clip(lower=0), n)
    loss = wilder(-d.clip(upper=0), n)
    rs = gain / loss.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.where(loss != 0, 100.0).where(gain.notna())


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    line = ema(close, fast) - ema(close, slow)
    sig = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return pd.DataFrame({"macd": line, "macd_signal": sig, "macd_hist": line - sig})


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev = close.shift(1)
    return pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    return wilder(true_range(high, low, close), n)


def adx(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.DataFrame:
    up, down = high.diff(), -low.diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    tr = wilder(true_range(high, low, close), n)
    plus_di = 100 * wilder(plus_dm, n) / tr
    minus_di = 100 * wilder(minus_dm, n) / tr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return pd.DataFrame({"adx": wilder(dx, n), "plus_di": plus_di, "minus_di": minus_di})


def supertrend(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 10, mult: float = 3.0) -> pd.DataFrame:
    """Classic Supertrend. direction +1 = uptrend (line below price), -1 = downtrend."""
    a = atr(high, low, close, n).to_numpy()
    mid = ((high + low) / 2).to_numpy()
    c = close.to_numpy()
    upper_basic, lower_basic = mid + mult * a, mid - mult * a
    size = len(c)
    upper = np.full(size, np.nan)
    lower = np.full(size, np.nan)
    line = np.full(size, np.nan)
    direction = np.full(size, np.nan)
    for i in range(size):
        if np.isnan(a[i]):
            continue
        if i == 0 or np.isnan(upper[i - 1]):
            upper[i], lower[i] = upper_basic[i], lower_basic[i]
            direction[i] = 1.0 if c[i] >= mid[i] else -1.0
        else:
            upper[i] = upper_basic[i] if (upper_basic[i] < upper[i - 1] or c[i - 1] > upper[i - 1]) else upper[i - 1]
            lower[i] = lower_basic[i] if (lower_basic[i] > lower[i - 1] or c[i - 1] < lower[i - 1]) else lower[i - 1]
            if direction[i - 1] == -1.0 and c[i] > upper[i - 1]:
                direction[i] = 1.0
            elif direction[i - 1] == 1.0 and c[i] < lower[i - 1]:
                direction[i] = -1.0
            else:
                direction[i] = direction[i - 1]
        line[i] = lower[i] if direction[i] == 1.0 else upper[i]
    return pd.DataFrame({"st_line": line, "st_dir": direction}, index=close.index)


def bollinger_width(close: pd.Series, n: int = 20, k: float = 2.0) -> pd.Series:
    mid = close.rolling(n, min_periods=n).mean()
    sd = close.rolling(n, min_periods=n).std(ddof=0)
    return (2 * k * sd) / mid


def session_key(ts: pd.Series) -> pd.Series:
    """IST trading date of each candle (ts must be tz-aware)."""
    return ts.dt.tz_convert("Asia/Kolkata").dt.date


def vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, day: pd.Series) -> pd.Series:
    """Session VWAP, reset each day. NaN until the session has traded volume."""
    tp = (high + low + close) / 3
    pv = (tp * volume).groupby(day).cumsum()
    vv = volume.groupby(day).cumsum()
    return pv / vv.replace(0, np.nan)


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    sign = np.sign(close.diff()).fillna(0)
    return (sign * volume).cumsum()


def relative_volume(volume: pd.Series, ts: pd.Series, sessions: int = 20) -> pd.Series:
    """Volume vs the average volume at the same time of day over the previous `sessions` days."""
    tod = ts.dt.tz_convert("Asia/Kolkata").dt.strftime("%H:%M")
    past_avg = volume.groupby(tod).transform(
        lambda v: v.shift(1).rolling(sessions, min_periods=max(5, sessions // 2)).mean()
    )
    return volume / past_avg.replace(0, np.nan)


def opening_range(high: pd.Series, low: pd.Series, ts: pd.Series, day: pd.Series,
                  minutes: int = 15) -> pd.DataFrame:
    """High/low of the first `minutes` of each session, known only once that window has closed."""
    local = ts.dt.tz_convert("Asia/Kolkata")
    mins = (local.dt.hour * 60 + local.dt.minute) - (9 * 60 + 15)
    in_window = mins < minutes
    or_high = high.where(in_window).groupby(day).transform("max")
    or_low = low.where(in_window).groupby(day).transform("min")
    known = ~in_window  # rows inside the window cannot know the final range yet
    return pd.DataFrame({"or_high": or_high.where(known), "or_low": or_low.where(known)})


def cpr(prev_high: pd.Series, prev_low: pd.Series, prev_close: pd.Series) -> pd.DataFrame:
    """Central Pivot Range from the previous session."""
    pivot = (prev_high + prev_low + prev_close) / 3
    bc = (prev_high + prev_low) / 2
    tc = 2 * pivot - bc
    return pd.DataFrame({"cpr_pivot": pivot, "cpr_top": np.maximum(tc, bc), "cpr_bottom": np.minimum(tc, bc)})
