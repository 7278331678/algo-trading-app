"""Feature frames on the 5-minute base timeframe.

Every row is one 5-minute candle; `decision_time` is when that candle closes, i.e. the
moment a decision could be made. Every feature in a row uses only data that was
available at decision_time:
- 15-minute features come from the last 15-minute candle that had CLOSED by then
- daily features come from the previous session
- opening range is known only after its window closes
The truncation test in tests/test_features.py checks this on synthetic data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import indicators as ind

FEATURE_VERSION = "v1"
BASE_MINUTES = 5

STOCK_FEATURES = [
    "ret_1", "ret_3", "ret_6", "ret_open", "minutes_since_open",
    "ema9_21_spread", "close_vs_ema21", "rsi14", "macd_hist_pct", "atr14_pct",
    "adx14", "di_spread", "st_dir", "st_dist", "bb_width", "vwap_dist", "obv_slope", "rvol",
    "or_breakout", "or_high_dist", "or_low_dist",
    "ema50_15m_dist", "ema50_15m_slope",
    "gap_pct", "prev_high_dist", "prev_low_dist", "cpr_pos", "cpr_width", "daily_atr_pct",
]
CONTEXT_FEATURES = [
    "nifty_ret_6", "nifty_ret_open", "nifty_ema50_15m_dist",
    "vix", "vix_chg", "sector_ret_6", "sector_ret_open", "rel_strength", "breadth_above_vwap",
]
ALL_FEATURES = STOCK_FEATURES + CONTEXT_FEATURES


def frame(rows: list[dict] | pd.DataFrame) -> pd.DataFrame:
    """Candles -> sorted frame with IST timestamps and session date."""
    df = pd.DataFrame(rows).copy()
    if df.empty:
        return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume", "day"])
    df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_convert("Asia/Kolkata")
    df = df.sort_values("ts").drop_duplicates("ts").reset_index(drop=True)
    for c in ("open", "high", "low", "close"):
        df[c] = df[c].astype(float)
    df["volume"] = df["volume"].astype(float)
    df["day"] = df["ts"].dt.date
    return df


def _prev_session(daily: pd.DataFrame) -> pd.DataFrame:
    """Per session D: values from the last daily candle strictly before D."""
    d = daily[["day", "high", "low", "close"]].copy()
    d["daily_atr"] = ind.atr(daily["high"], daily["low"], daily["close"], 14)
    d = d.rename(columns={"high": "prev_high", "low": "prev_low", "close": "prev_close"})
    d["day"] = pd.to_datetime(d["day"])
    return d.sort_values("day")


def _attach_prev_session(m5: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    left = m5.assign(_day=pd.to_datetime(m5["day"])).sort_values("_day")
    merged = pd.merge_asof(left, _prev_session(daily), left_on="_day", right_on="day",
                           direction="backward", allow_exact_matches=False, suffixes=("", "_d"))
    return merged.drop(columns=["_day", "day_d"]).sort_values("ts").reset_index(drop=True)


def _attach_15m(m5: pd.DataFrame, m15: pd.DataFrame) -> pd.DataFrame:
    if m15.empty:
        return m5.assign(ema50_15m=np.nan, ema50_15m_prev=np.nan)
    f = pd.DataFrame({
        "available_at": m15["ts"] + pd.Timedelta(minutes=15),  # a 15-min candle is known when it closes
        "ema50_15m": ind.ema(m15["close"], 50),
    })
    f["ema50_15m_prev"] = f["ema50_15m"].shift(3)
    f = f.sort_values("available_at")
    return pd.merge_asof(m5.sort_values("decision_time"), f, left_on="decision_time",
                         right_on="available_at", direction="backward").drop(columns=["available_at"])


def stock_features(m5: pd.DataFrame, m15: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    """Per-stock features on 5-minute candles (inputs from frame())."""
    df = m5.copy()
    df["decision_time"] = df["ts"] + pd.Timedelta(minutes=BASE_MINUTES)
    c, h, lo, v = df["close"], df["high"], df["low"], df["volume"]
    logc = np.log(c)
    df["ret_1"], df["ret_3"], df["ret_6"] = logc.diff(1), logc.diff(3), logc.diff(6)
    day_open = df.groupby("day")["open"].transform("first")
    df["ret_open"] = c / day_open - 1
    local = df["ts"]
    df["minutes_since_open"] = (local.dt.hour * 60 + local.dt.minute + BASE_MINUTES) - (9 * 60 + 15)

    e9, e21 = ind.ema(c, 9), ind.ema(c, 21)
    df["ema9_21_spread"] = (e9 - e21) / c
    df["close_vs_ema21"] = c / e21 - 1
    df["rsi14"] = ind.rsi(c, 14)
    df["macd_hist_pct"] = ind.macd(c)["macd_hist"] / c
    df["atr14"] = ind.atr(h, lo, c, 14)
    df["atr14_pct"] = df["atr14"] / c
    a = ind.adx(h, lo, c, 14)
    df["adx14"], df["di_spread"] = a["adx"], a["plus_di"] - a["minus_di"]
    st = ind.supertrend(h, lo, c, 10, 3.0)
    df["st_dir"], df["st_dist"] = st["st_dir"], (c - st["st_line"]) / c
    df["bb_width"] = ind.bollinger_width(c, 20, 2.0)
    df["vwap"] = ind.vwap(h, lo, c, v, df["day"])
    df["vwap_dist"] = c / df["vwap"] - 1
    avg_v = v.rolling(20, min_periods=10).mean().replace(0, np.nan)
    df["obv_slope"] = ind.obv(c, v).diff(6) / (6 * avg_v)
    df["rvol"] = ind.relative_volume(v, df["ts"], sessions=20)

    orr = ind.opening_range(h, lo, df["ts"], df["day"], minutes=15)
    df["or_high_dist"] = c / orr["or_high"] - 1
    df["or_low_dist"] = c / orr["or_low"] - 1
    df["or_breakout"] = np.select([c > orr["or_high"], c < orr["or_low"]], [1.0, -1.0], 0.0)
    df.loc[orr["or_high"].isna(), "or_breakout"] = np.nan

    df = _attach_15m(df, m15)
    df["ema50_15m_dist"] = df["close"] / df["ema50_15m"] - 1
    df["ema50_15m_slope"] = (df["ema50_15m"] - df["ema50_15m_prev"]) / df["close"]

    df = _attach_prev_session(df, daily)
    first_open = df.groupby("day")["open"].transform("first")
    df["gap_pct"] = first_open / df["prev_close"] - 1
    df["prev_high_dist"] = df["close"] / df["prev_high"] - 1
    df["prev_low_dist"] = df["close"] / df["prev_low"] - 1
    cp = ind.cpr(df["prev_high"], df["prev_low"], df["prev_close"])
    df["cpr_pos"] = df["close"] / cp["cpr_pivot"] - 1
    df["cpr_width"] = (cp["cpr_top"] - cp["cpr_bottom"]) / cp["cpr_pivot"]
    df["daily_atr_pct"] = df["daily_atr"] / df["prev_close"]
    return df.sort_values("ts").reset_index(drop=True)


def index_context(m5: pd.DataFrame, m15: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Index features by timestamp (indices have no volume, so no VWAP)."""
    df = m5[["ts", "day", "open", "close"]].copy()
    df["decision_time"] = df["ts"] + pd.Timedelta(minutes=BASE_MINUTES)
    df[f"{prefix}_ret_6"] = np.log(df["close"]).diff(6)
    df[f"{prefix}_ret_open"] = df["close"] / df.groupby("day")["open"].transform("first") - 1
    df = _attach_15m(df, m15)
    df[f"{prefix}_ema50_15m_dist"] = df["close"] / df["ema50_15m"] - 1
    return df[["ts", f"{prefix}_ret_6", f"{prefix}_ret_open", f"{prefix}_ema50_15m_dist"]]


def vix_context(m5: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    df = _attach_prev_session(m5[["ts", "day", "open", "high", "low", "close", "volume"]].copy(), daily)
    return pd.DataFrame({"ts": df["ts"], "vix": df["close"], "vix_chg": df["close"] / df["prev_close"] - 1})


def add_context(stock: pd.DataFrame, nifty: pd.DataFrame, sector: pd.DataFrame | None,
                vix: pd.DataFrame | None, breadth: pd.Series | None) -> pd.DataFrame:
    """Join index/VIX/breadth context on the candle timestamp (same 5-minute grid)."""
    df = stock.merge(nifty, on="ts", how="left")
    if sector is not None:  # built with index_context(..., prefix="sector")
        df = df.merge(sector[["ts", "sector_ret_6", "sector_ret_open"]], on="ts", how="left")
    else:
        df["sector_ret_6"], df["sector_ret_open"] = df["nifty_ret_6"], df["nifty_ret_open"]
    if vix is not None:
        df = df.merge(vix, on="ts", how="left")
    else:
        df["vix"], df["vix_chg"] = np.nan, np.nan
    df["rel_strength"] = df["ret_open"] - df["nifty_ret_open"]
    if breadth is not None:
        df = df.merge(breadth.rename("breadth_above_vwap"), left_on="ts", right_index=True, how="left")
    else:
        df["breadth_above_vwap"] = np.nan
    return df


def above_vwap(m5: pd.DataFrame) -> pd.Series:
    """1/0 per timestamp: close above session VWAP (input for market breadth)."""
    vw = ind.vwap(m5["high"], m5["low"], m5["close"], m5["volume"], m5["day"])
    return pd.Series((m5["close"] > vw).astype(float).where(vw.notna()).to_numpy(), index=m5["ts"])
