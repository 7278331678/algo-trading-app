"""Signal generators. Each adds `signal` (+1 long, -1 short, 0 none) and `score` (higher first)
to the feature frame. They only read feature columns, which are known at decision time.

Rows that must not be traded (exclude, not in the index that day) never get a signal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _tradable(df: pd.DataFrame) -> pd.Series:
    return (~df["exclude"].astype(bool)) & df["in_universe"].astype(bool)


def rules_baseline(df: pd.DataFrame, min_rvol: float = 1.5, min_adx: float = 20.0,
                   allow_short: bool = True) -> pd.DataFrame:
    """Opening-range breakout confirmed by VWAP, 15-min trend, volume and trend strength.

    long:  close above the opening range high, above VWAP, above the 15-min EMA50,
           relative volume >= min_rvol, ADX >= min_adx, Supertrend up, Nifty not falling > 0.5%
    short: the mirror image
    score: relative volume (strongest participation first)
    """
    out = df.copy()
    common = _tradable(df) & (df["rvol"] >= min_rvol) & (df["adx14"] >= min_adx)
    long_ = (common & (df["or_breakout"] == 1) & (df["vwap_dist"] > 0) & (df["ema50_15m_dist"] > 0)
             & (df["st_dir"] == 1) & (df["nifty_ret_open"] > -0.005))
    short = (common & (df["or_breakout"] == -1) & (df["vwap_dist"] < 0) & (df["ema50_15m_dist"] < 0)
             & (df["st_dir"] == -1) & (df["nifty_ret_open"] < 0.005))
    if not allow_short:
        short &= False
    out["signal"] = np.select([long_, short], [1, -1], 0)
    out["score"] = df["rvol"].fillna(0.0)
    return out


def random_baseline(df: pd.DataFrame, rate: float = 0.002, seed: int = 42) -> pd.DataFrame:
    """Random entries: the sanity check. With no edge it should lose roughly its costs."""
    out = df.copy()
    rng = np.random.default_rng(seed)
    pick = _tradable(df).to_numpy() & (rng.random(len(df)) < rate)
    out["signal"] = np.where(pick, rng.choice([1, -1], len(df)), 0)
    out["score"] = rng.random(len(df))
    return out


STRATEGIES = {"rules": rules_baseline, "random": random_baseline}

# stop/target sizes compared in the barrier study: (name, stop_basis, stop_mult, target_mult)
BARRIER_GRID = [
    ("atr5_1.0x1.5", "atr5", 1.0, 1.5),
    ("atr5_2.0x3.0", "atr5", 2.0, 3.0),
    ("daily_0.25x0.375", "daily", 0.25, 0.375),
    ("daily_0.4x0.6", "daily", 0.4, 0.6),
]
