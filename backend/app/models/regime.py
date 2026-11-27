"""Bad-day regime gate (rules from the plan), evaluated from features known at decision time.

A day turns "bad" from the first candle where any condition holds, and stays bad for the rest of
the day: Nifty down more than 1% from the open, India VIX up more than 8% from the previous close,
or fewer than 30% of index stocks above their VWAP (after 10:00, once breadth is meaningful).
On bad days only longs with relative strength and a stricter threshold are allowed.
"""

from __future__ import annotations

import pandas as pd


def bad_day_flag(df: pd.DataFrame, nifty_drop: float = -0.01, vix_jump: float = 0.08,
                 min_breadth: float = 0.30) -> pd.Series:
    after_10 = df["minutes_since_open"] >= 45
    now_bad = (df["nifty_ret_open"] <= nifty_drop) | (df["vix_chg"] >= vix_jump) | \
        (after_10 & (df["breadth_above_vwap"] < min_breadth))
    now_bad = now_bad.fillna(False).astype(int)
    # once bad, bad for the rest of the day (cumulative max per symbol-day, in time order)
    order = df.sort_values("ts").index
    flag = now_bad.loc[order].groupby([df.loc[order, "symbol"], df.loc[order, "day"]]).cummax()
    return flag.reindex(df.index).astype(bool)
