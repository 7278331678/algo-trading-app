"""Event features at each decision time, using only what was public before it.

For a row of stock S deciding at time D (candle close) on day T:
- "since close" means announcements in (previous session's 15:30, D - LATENCY]: the overnight and
  morning news a trader could have read before deciding.
- an announcement counts only if public_at <= D - LATENCY (one minute to read it).
- a board meeting counts only once known_at <= D (its intimation time, or a late estimate).

Everything is vectorised per stock with sorted arrays and searchsorted, so 4 million rows take
seconds. Missing values are filled with neutral caps (not NaN) so the logistic model can use them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .classify import MATERIAL

LATENCY = pd.Timedelta(minutes=1)
FLAG_CATEGORIES = ["results", "order", "rating", "mna", "corp_action", "holder", "regulatory",
                   "clarification", "kmp", "guidance"]
HOURS_CAP = 168.0            # one week
DAYS_TO_CAP, DAYS_SINCE_CAP = 30.0, 120.0

EVENT_FEATURES = (["ev_n_since_close", "ev_material_since_close", "ev_material_7d", "ev_hours_since_material"]
                  + [f"ev_{c}_since_close" for c in FLAG_CATEGORIES]
                  + ["ev_results_today", "ev_results_prev_day", "ev_days_to_results", "ev_days_since_results"])


def _ns(x) -> np.ndarray:
    """tz-aware datetimes -> int64 nanoseconds UTC."""
    s = pd.to_datetime(pd.Series(x), utc=True)
    return s.dt.tz_localize(None).astype("datetime64[ns]").to_numpy().astype("int64")


def _count_le(events: np.ndarray, t: np.ndarray) -> np.ndarray:
    return np.searchsorted(events, t, side="right") if len(events) else np.zeros(len(t), dtype=np.int64)


def add_features(df: pd.DataFrame, ann: pd.DataFrame, meetings: pd.DataFrame) -> pd.DataFrame:
    """Adds EVENT_FEATURES (float32) to df in place and returns it."""
    out = {c: np.zeros(len(df), dtype=np.float32) for c in EVENT_FEATURES}
    ann_by = {}
    if len(ann):
        ann = ann.copy()
        ann["is_material"] = ann["category"].isin(MATERIAL)
        ann_by = {s: g for s, g in ann.groupby("symbol")}
    bm = meetings[meetings["is_results"].astype(bool)] if len(meetings) else meetings
    bm_by = {s: g for s, g in bm.groupby("symbol")} if len(bm) else {}

    for sym, idx in df.groupby("symbol", sort=False).indices.items():
        rows = df.iloc[idx]
        dec = _ns(rows["decision_time"]) - LATENCY.value
        day = pd.to_datetime(rows["day"]).to_numpy()
        # previous session close for each row: 15:30 IST on this stock's previous trading day
        days = np.unique(day)
        prev = np.r_[days[0] - np.timedelta64(3, "D"), days[:-1]]
        prev_of = dict(zip(days, prev, strict=True))
        prev_day = np.array([prev_of[d] for d in day], dtype="datetime64[ns]")
        close_prev = _ns(pd.DatetimeIndex(prev_day).tz_localize("Asia/Kolkata") + pd.Timedelta(hours=15, minutes=30))
        week_ago = dec - pd.Timedelta(days=7).value

        g = ann_by.get(sym)
        if g is not None:
            t_all = np.sort(_ns(g["public_at"]))
            t_mat = np.sort(_ns(g.loc[g["is_material"], "public_at"]))
            out["ev_n_since_close"][idx] = _count_le(t_all, dec) - _count_le(t_all, close_prev)
            out["ev_material_since_close"][idx] = _count_le(t_mat, dec) - _count_le(t_mat, close_prev)
            out["ev_material_7d"][idx] = _count_le(t_mat, dec) - _count_le(t_mat, week_ago)
            k = _count_le(t_mat, dec)
            last = np.where(k > 0, t_mat[np.maximum(k - 1, 0)] if len(t_mat) else 0, np.iinfo(np.int64).min)
            hours = np.where(k > 0, (dec - last) / 3.6e12, HOURS_CAP)
            out["ev_hours_since_material"][idx] = np.minimum(hours, HOURS_CAP)
            for c in FLAG_CATEGORIES:
                tc = np.sort(_ns(g.loc[g["category"] == c, "public_at"]))
                if len(tc):
                    out[f"ev_{c}_since_close"][idx] = ((_count_le(tc, dec) - _count_le(tc, close_prev)) > 0)
        else:
            out["ev_hours_since_material"][idx] = HOURS_CAP

        to_days = np.full(len(idx), DAYS_TO_CAP)
        since_days = np.full(len(idx), DAYS_SINCE_CAP)
        today = np.zeros(len(idx), bool)
        prev_flag = np.zeros(len(idx), bool)
        m = bm_by.get(sym)
        if m is not None:
            dec_full = dec + LATENCY.value
            for md, kn in zip(pd.to_datetime(m["meeting_date"]).to_numpy(), _ns(m["known_at"]), strict=True):
                md = md.astype("datetime64[D]").astype("datetime64[ns]")
                known = kn <= dec_full
                gap = (md - day.astype("datetime64[D]").astype("datetime64[ns]")) / np.timedelta64(1, "D")
                ahead = known & (gap >= 0)
                to_days = np.where(ahead, np.minimum(to_days, gap), to_days)
                today |= known & (gap == 0)
                prev_flag |= md == prev_day.astype("datetime64[D]").astype("datetime64[ns]")
                past = gap < 0           # a meeting in the past is public by definition
                since_days = np.where(past, np.minimum(since_days, -gap), since_days)
        out["ev_days_to_results"][idx] = np.minimum(to_days, DAYS_TO_CAP)
        out["ev_days_since_results"][idx] = np.minimum(since_days, DAYS_SINCE_CAP)
        out["ev_results_today"][idx] = today
        out["ev_results_prev_day"][idx] = prev_flag

    for c, v in out.items():
        df[c] = v.astype("float32")
    return df
